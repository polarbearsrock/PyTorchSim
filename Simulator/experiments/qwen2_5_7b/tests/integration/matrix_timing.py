"""End-to-end BF16/FP32 GEMM timing regression with exact K-tile cancellation.

Exercises frontend -> corrected TOG extraction -> gem5 -> TOGSim, with Spike
output checks. Small matrices keep this separate from the full Qwen experiment.
"""
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

from Simulator.experiments.qwen2_5_7b.config import configure_simulator
from Simulator.experiments.qwen2_5_7b.tests.integration.tog_regions import audit_regions, check_graph_regions


def check_lowered_regions(text, regions):
    """Check the post-legalization LLVM IR actually passed to llc for gem5."""
    lowered, active = [], None
    for line in text.splitlines():
        if ".insn r CUSTOM_3, 0, 0x40" in line:
            if active is not None:
                raise AssertionError("Nested LLVM timing markers")
            active = [0, 0]
        elif ".insn r CUSTOM_3, 0, 0x41" in line:
            if active is None:
                raise AssertionError("Unmatched LLVM timing marker")
            lowered.append(tuple(active))
            active = None
        else:
            push = bool(re.search(r'call .*@llvm\.riscv\.sf\.vc\.iv[^\n]*\(i64 0,', line))
            read = bool(re.search(r'call .*@llvm\.riscv\.sf\.vc\.v\.i[^\n]*\(i64 2,', line))
            if active is None and (push or read):
                raise AssertionError("Matrix transfer outside LLVM timing markers")
            if active is not None:
                active[0] += int(push)
                active[1] += int(read)
    if active is not None or lowered != [(r["pushes"], r["reads"]) for r in regions]:
        raise AssertionError("LLVM lowering changed matrix timing-region coverage")


def main():
    scratch = Path(os.environ["TMPDIR"])
    configure_simulator(SimpleNamespace(mode="timing", validate_timing=True, output_dir=scratch))
    import torch
    from PyTorchSimFrontend import extension_config
    from Simulator.simulator import TOGSimulator

    mapping = scratch / "matrix-mapping.json"
    mapping.write_text(json.dumps({
        f"{m}_128_384": {"TILE_M": m, "TILE_N": 128, "TILE_K": 128}
        for m in (8, 128)
    }))
    extension_config.codegen_mapping_strategy = "external-then-heuristic"
    extension_config.codegen_external_mapping_file = str(mapping)
    cases = []
    with torch.no_grad(), TOGSimulator(config_path=os.environ["TOGSIM_CONFIG"]):
        for m, dtype in ((8, torch.bfloat16), (128, torch.bfloat16),
                         (128, torch.float16), (128, torch.float32)):
            a = torch.ones(m, 384, dtype=dtype)
            b = torch.zeros(384, 128, dtype=dtype)
            b[0], b[128], b[256] = 256, 1, -256
            expected = (a.float() @ b.float()).to(dtype)
            compiled = torch.compile(lambda x, w: x @ w, fullgraph=True, dynamic=False)
            actual = torch.npu.launch_model(compiled, a.to("npu:0"), b.to("npu:0"),
                                            stream_index=0, timestamp=0).cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            cases.append(dict(m=m, n=128, k=384, tile_k=128, dtype=str(dtype),
                              elements=actual.numel(), numerical_status="bit_exact"))
            print(f"M={m}, {dtype}: Spike cancellation output is bit-exact", flush=True)
        torch.npu.synchronize()

    artifacts = []
    samples = sorted(Path(os.environ["TORCHSIM_DUMP_PATH"]).glob("*/*_sample_llvm.mlir.tog_post.mlir"))
    if len(samples) != 4:
        raise AssertionError(f"Expected four freshly compiled timing kernels, got {len(samples)}")
    for sample in samples:
        regions = audit_regions(sample.read_text())
        source = sample.with_name(sample.name.removesuffix("_sample_llvm.mlir.tog_post.mlir"))
        check_graph_regions(source.with_name(source.name + "_tog.py").read_text(), regions)
        lowered = source.with_name(source.name + "_sample.ll.bf16_memory.ll")
        if not lowered.exists():
            lowered = source.with_name(source.name + "_sample.ll")
        check_lowered_regions(lowered.read_text(), regions)
        stats = (sample.parent / "m5out/stats.txt").read_text()
        cycles = [int(value) for value in re.findall(r"^system.cpu.numCycles\s+(\d+)", stats, re.M)][:-1]
        if len(cycles) != len(regions) or not all(cycle > 0 for cycle in cycles):
            raise AssertionError("gem5 did not return one positive latency per timing region")
        matrix = [r for r in regions if r["kind"] == "MatmulCompute"]
        artifacts.append(dict(sample=str(sample), matrix_regions=matrix, region_latencies=cycles))
    counts = sorted((r["pushes"], r["reads"]) for a in artifacts for r in a["matrix_regions"])
    if counts != [(1, 1), (8, 8), (8, 16), (16, 16)]:
        raise AssertionError(f"Unexpected push/read coverage: {counts}")
    compiler = Path(extension_config.CONFIG_TORCHSIM_LLVM_PATH) / "mlir-opt"
    report = dict(status="passed", cases=cases, artifacts=artifacts,
                  compiler=str(compiler), compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest())
    (scratch / "matrix-timing-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: compiler regions, gem5 latencies, TOGSim completion, and exact Spike outputs", flush=True)


if __name__ == "__main__":
    main()
