"""Compiler integration tests for matrix timing attribution, not numerical values.

Runs the real bundled and patched MLIR passes. Optional saved-run replay uses
the original post-BF16-lowering IR read-only; all outputs go under TMPDIR.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def audit_regions(text):
    regions, active = [], None
    for line in text.splitlines():
        if ".insn r CUSTOM_3, 0, 0x40" in line:
            if active is not None:
                raise ValueError("Nested timing markers")
            kind = re.search(r'compute_type = "([^"]+)"', line)
            if kind is None:
                raise ValueError("Timing marker has no compute_type")
            active = dict(kind=kind[1], pushes=0, reads=0, stores=0, roles=[])
        elif ".insn r CUSTOM_3, 0, 0x41" in line:
            if active is None:
                raise ValueError("Unmatched timing end marker")
            regions.append(active)
            active = None
        else:
            is_read = '"vcix.v.i"' in line and "opcode = 2 :" in line
            if is_read and (active is None or active["kind"] != "MatmulCompute"):
                raise ValueError("Matrix result read attributed outside MatmulCompute")
            if active is None:
                continue
            active["reads"] += int(is_read)
            active["pushes"] += int('"vcix.iv"' in line and "opcode = 0 :" in line)
            active["stores"] += int('"vector.transfer_write"' in line)
            role = re.search(r'test.role = "([^"]+)"', line)
            if role:
                active["roles"].append(role[1])
                expected = "MatmulCompute" if role[1].startswith("accum") else "VectorCompute"
                if active["kind"] != expected:
                    raise ValueError(f"{role[1]} attributed to {active['kind']}, expected {expected}")
    if active is not None:
        raise ValueError("Unclosed timing marker")
    for region in regions:
        if region["kind"] == "MatmulCompute":
            if not region["pushes"] or not region["reads"]:
                raise ValueError("Incomplete matrix region")
            if region["stores"] != region["reads"]:
                raise ValueError("Matrix region does not include all accumulator writes")
    return regions


def check_graph_regions(graph_text, regions):
    # Do not execute generated Python. Traverse graph children in source order;
    # dictionary insertion order is BFS and is not the timing-marker order.
    tree = ast.parse(graph_text)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        raise ValueError("Expected one emitted timing graph")
    graph = ast.literal_eval(tree.body[0].value)
    kinds = []
    def walk(identity):
        node = graph[identity]
        if node["node_name"] == "ComputeNode":
            kinds.append(node["compute_type"])
        for child in node["children"]:
            walk(child)
    walk(0)
    values = {"VectorCompute": 0, "MatmulCompute": 1, "MatmulPreload": 2}
    if kinds != [values[region["kind"]] for region in regions]:
        raise ValueError("Graph unit labels disagree with timing markers")


def without_markers(text):
    return "\n".join(line for line in text.splitlines() if ".insn r CUSTOM_3" not in line)


def fixture(input_lanes, output_lanes, input_dtype, output_dtype, segments, orphan=False):
    feed_type = f"vector<{input_lanes}x{input_dtype}>"
    out_type = f"vector<{output_lanes}x{output_dtype}>"
    acc_type = f"memref<256x{output_dtype}>"
    lines = ["module {", f"  func.func @kernel(%feed: {feed_type}, %acc: {acc_type}) {{",
             "    %c0 = arith.constant 0 : index",
             f"    %vl = arith.constant {input_lanes} : i64",
             f"    %zero = arith.constant 0.0 : {output_dtype}",
             "    affine.for %i = 0 to 2 {"]
    def push(opcode):
        lines.append(f'      "vcix.iv"(%feed, %vl) <{{imm = 0 : i64, opcode = {opcode} : i64, rd = 0 : i64}}> : ({feed_type}, i64) -> ()')
    for segment, (pushes, reads) in enumerate(segments):
        if not orphan:
            push(1)
            for _ in range(pushes):
                push(0)
            lines.append('      "vcix.i"(%vl) <{imm = 4 : i64, lmul = 0 : i64, opcode = 1 : i64, rd = 0 : i64, rs2 = 0 : i64, sew = 16 : i64}> : (i64) -> ()')
        for index in range(reads):
            suffix = f"{segment}_{index}"
            lines.extend([
                f'      %pop{suffix} = "vcix.v.i"(%vl) <{{imm = 0 : i64, opcode = 2 : i64, rs2 = 0 : i64}}> : (i64) -> {out_type}',
                f"      %old{suffix} = vector.transfer_read %acc[%c0], %zero : {acc_type}, {out_type}",
                f"      %sum{suffix} = arith.addf %old{suffix}, %pop{suffix} : {out_type}",
                f'      vector.transfer_write %sum{suffix}, %acc[%c0] {{test.role = "accum-{suffix}"}} : {out_type}, {acc_type}',
            ])
        lines.extend([
            f"      %tail{segment} = arith.negf %sum{suffix} : {out_type}",
            f'      vector.transfer_write %tail{segment}, %acc[%c0] {{test.role = "epilogue-{segment}"}} : {out_type}, {acc_type}',
        ])
    lines.extend(["    } {inner_loop = true}", "    return", "  }", "}"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-run", type=Path)
    parser.add_argument("--original-mlir-opt", type=Path, default=Path("/riscv-llvm/bin/mlir-opt"),
                        help="Unchanged image compiler, used only as a negative/control reference")
    args = parser.parse_args()
    output = Path(os.environ["TMPDIR"]) / "tog-regions"
    output.mkdir()
    llvm = Path(os.environ.get("TORCHSIM_LLVM_PATH", "/riscv-llvm/bin"))
    from PyTorchSimFrontend.toolchain import require_mixed_width_tog
    require_mixed_width_tog(llvm)
    bf16_plugin = os.environ["TORCHSIM_BF16_PLUGIN"]
    dialect = f"--load-dialect-plugin={bf16_plugin}"
    results = []

    def run_pass(source, name, sample=True, patched=True, fail=False):
        destination = output / f"{name}.mlir"
        options = f"vectorlane=128 sample-mode={int(sample)}"
        compiler = llvm / "mlir-opt" if patched else args.original_mlir_opt
        command = [str(compiler), dialect, f"-test-tile-operation-graph={options}",
                   "--mlir-print-op-generic", str(source), "-o", str(destination)]
        result = subprocess.run(command, text=True, capture_output=True, timeout=60)
        (output / f"{name}.tog.txt").write_text(result.stdout)
        (output / f"{name}.stderr").write_text(result.stderr)
        if fail:
            if result.returncode == 0 or "matrix result read is outside" not in result.stderr:
                raise AssertionError("Orphan matrix read was not rejected")
            return None
        if result.returncode:
            raise RuntimeError(f"Compiler failed; see {name}.stderr: {result.stderr[:1500]}")
        text = destination.read_text()
        if patched:
            check_graph_regions(result.stdout, audit_regions(text))
        return text

    cases = {
        "bf16_m128": (16, 8, "i16", "f32", [(8, 16)]),
        "bf16_m8": (8, 8, "i16", "f32", [(1, 1)]),
        "bf16_m16": (16, 8, "i16", "f32", [(1, 2)]),
        "bf16_m32": (16, 8, "i16", "f32", [(2, 4)]),
        "bf16_padded_tail": (16, 8, "i16", "f32", [(2, 3)]),
        "fp16_equal_width": (16, 16, "f16", "f16", [(8, 8)]),
        "fp32_equal_width": (8, 8, "f32", "f32", [(16, 16)]),
        "wider_output_vector": (8, 16, "f32", "f32", [(4, 2)]),
        "adjacent_segments": (16, 8, "i16", "f32", [(8, 16), (1, 2), (2, 3)]),
    }
    for name, spec in cases.items():
        source = output / f"{name}.input.mlir"
        source.write_text(fixture(*spec))
        for sample in (True, False):
            label = f"{name}-sample-{int(sample)}"
            text = run_pass(source, label, sample=sample)
            if f"step = {2 if sample else 1} : index" not in text:
                raise AssertionError(f"Sampling mode was not honored in {label}")
            regions = audit_regions(text)
            matrix = [r for r in regions if r["kind"] == "MatmulCompute"]
            if [(r["pushes"], r["reads"]) for r in matrix] != spec[-1]:
                raise AssertionError(f"Incorrect matrix segmentation in {label}: {matrix}")
            if sum(len(r["roles"]) for r in regions) != sum(n + 1 for _, n in spec[-1]):
                raise AssertionError(f"Missing accumulator/epilogue coverage in {label}")
            results.append(dict(case=label, status="passed", regions=regions))
        if name in ("bf16_m8", "fp16_equal_width", "fp32_equal_width"):
            old = run_pass(source, name + "-original", patched=False)
            old_regions = audit_regions(old)
            new_regions = results[-2]["regions"]
            if old_regions != new_regions:
                raise AssertionError(f"Equal-width control changed: {name}")
            new = (output / f"{name}-sample-1.mlir").read_text()
            if without_markers(old) != without_markers(new):
                raise AssertionError(f"Non-marker operations changed: {name}")
        if name == "bf16_m128":
            old = run_pass(source, name + "-original", patched=False)
            new = (output / f"{name}-sample-1.mlir").read_text()
            if without_markers(old) != without_markers(new):
                raise AssertionError("BF16 M=128 non-marker operations changed")
            try:
                audit_regions(old)
            except ValueError as error:
                if "Matrix result read attributed outside" not in str(error):
                    raise
                results.append(dict(case="original-pass-negative-control", status="failed_as_expected", error=str(error)))
            else:
                raise AssertionError("Regression did not reproduce the original attribution bug")

    source = output / "orphan.input.mlir"
    source.write_text(fixture(16, 8, "i16", "f32", [(0, 1)], orphan=True))
    run_pass(source, "orphan", fail=True)
    results.append(dict(case="orphan-result-read", status="rejected_as_expected"))

    if args.saved_run:
        sources = sorted(args.saved_run.glob("generated/*/*_sample_llvm.mlir.bf16_post.mlir"))
        if not sources:
            raise RuntimeError("Saved run has no post-BF16 lowering IR")
        for index, source in enumerate(sources):
            name = f"saved-{index:03d}-{source.parent.name}"
            prepared = output / f"{name}.input.mlir"
            preparation = subprocess.run([
                str(llvm / "mlir-opt"), dialect,
                "-arith-emulate-unsupported-floats=source-types=bf16 target-type=f32",
                "-test-pytorchsim-to-vcix=systolic-array-size=128 vlen=256",
                "--mlir-print-op-generic", str(source), "-o", str(prepared),
            ], text=True, capture_output=True, timeout=60)
            (output / f"{name}.prepare.stderr").write_text(preparation.stderr)
            preparation.check_returncode()
            regions = audit_regions(run_pass(prepared, name))
            expected_reads = sum('"vcix.v.i"' in line and "opcode = 2 :" in line
                                 for line in prepared.read_text().splitlines())
            if sum(r["reads"] for r in regions) != expected_reads:
                raise AssertionError(f"Missing result-read coverage: {source}")
            results.append(dict(case=name, status="passed", source=str(source),
                                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), regions=regions))
    compiler = llvm / "mlir-opt"
    report = dict(status="passed", compiler=str(compiler),
                  compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(), cases=results)
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(results)} checks; results: {output / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
