"""CLI for offline Qwen workloads; configure the backend before importing PyTorch.

The existing result.json schema is retained, including legacy probe-named
metadata keys, so recorded-run analysis remains compatible.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from .analysis.memory_audit import memory_audit
from .config import MODEL_PATH, configure_simulator, load_manifest
from .provenance import transformers_provenance
from .workloads.cpu_reference import cpu_reference


def run_workload(args, manifest, torch):
    if args.component == "decoder":
        from .workloads.decoder import run_decoder
        return run_decoder(args, manifest, torch)
    if args.component == "attention":
        from .workloads.attention import run_attention
        return run_attention(args, manifest, torch)
    if args.component == "attention_parts":
        from .tests.integration.attention_components import run_attention_components
        return run_attention_components(args, manifest, torch)
    from .workloads.components import run_components
    return run_components(args, manifest, torch)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "cpu", "functional", "timing"), default="audit")
    parser.add_argument("--component", choices=("decoder", "rmsnorm", "q_proj", "mlp", "attention", "attention_parts"), default="decoder")
    parser.add_argument("--attention-case", choices=("all", "lookup", "rotate", "rope_math", "rope", "gqa", "scores", "softmax", "context", "cache"), default="all")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mapping-file", type=Path, help="Replay a recorded tile mapping instead of autotuning")
    parser.add_argument("--validate-timing", action="store_true", help="Decoder/attention: run Spike and CPU checks alongside heuristic timing, supplying real indirect-address indices")
    parser.add_argument("--allow-numerical-mismatch", action="store_true",
                        help="Exploratory decoder only: record finite CPU tolerance mismatches and continue; never marks numerical validation as passed")
    args = parser.parse_args()
    if args.validate_timing and (args.mode != "timing" or args.component not in ("decoder", "attention", "attention_parts")):
        parser.error("--validate-timing requires timing mode and a decoder/attention component")
    if args.validate_timing and args.mapping_file:
        parser.error("--validate-timing currently requires the heuristic mapping, without --mapping-file")
    if args.component == "decoder" and args.mode == "timing" and not args.validate_timing:
        parser.error("The decoder baseline requires --validate-timing; unvalidated timing is not a baseline")
    if args.allow_numerical_mismatch and (args.component != "decoder" or args.mode not in ("functional", "timing")):
        parser.error("--allow-numerical-mismatch requires a functional/timing decoder run")
    if min(args.batch, args.seq_len, args.context_tokens, args.decode_steps) < 1:
        parser.error("All shape/count arguments must be positive")
    if args.mode != "audit" and args.output_dir is None:
        parser.error("--output-dir is required outside audit mode")
    if args.output_dir:
        args.output_dir = args.output_dir.resolve()
        tmpdir = Path(os.environ["TMPDIR"]).resolve()
        if not args.output_dir.is_relative_to(tmpdir):
            parser.error("Transient outputs must be under TMPDIR")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def main():
    args = parse_args()
    manifest = load_manifest()
    result = {
        "model_id": manifest["model_id"], "revision": manifest["revision"],
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "audit": memory_audit(manifest, args.batch, args.context_tokens),
        "checkpoint_loaded": False,
        "baseline_dtype": "bfloat16; float32 probes are diagnostic controls only",
        "probe_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_manifest_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
    }
    started = time.monotonic()
    try:
        if args.mode in ("functional", "timing"):
            result["simulator_config"] = configure_simulator(args)
        if args.mode != "audit":
            if args.component == "decoder":
                result["transformers_source"] = transformers_provenance()
                (args.output_dir / "transformers_source.json").write_text(
                    json.dumps(result["transformers_source"], indent=2) + "\n")
            sys.path.insert(0, os.environ["TORCHSIM_DIR"])
            import torch
            import transformers

            result["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__}
            # Record the binaries actually selected; a path alone is not enough
            # when testing locally rebuilt toolchains.
            result["toolchain"] = {}
            selected = {
                "spike": os.environ.get("TORCHSIM_SPIKE") or shutil.which("spike"),
                "mlir_bf16_plugin": os.environ.get("TORCHSIM_BF16_PLUGIN"),
            }
            if args.mode in ("functional", "timing"):
                from PyTorchSimFrontend import extension_config
                selected["gem5"] = extension_config.CONFIG_GEM5_PATH
                for name in ("mlir-opt", "mlir-translate", "llc", "opt"):
                    selected[name] = str(Path(extension_config.CONFIG_TORCHSIM_LLVM_PATH) / name)
            if selected["mlir_bf16_plugin"]:
                selected["llvm_bf16_memory_plugin"] = str(Path(selected["mlir_bf16_plugin"]).with_name("libPyTorchSimBF16Memory.so"))
            for name, location in selected.items():
                if location:
                    with Path(location).open("rb") as binary:
                        digest = hashlib.file_digest(binary, "sha256").hexdigest()
                    result["toolchain"][name] = {"path": location, "sha256": digest}
            result["toolchain"]["bf16_max_lmul"] = 4 if args.dtype == "bfloat16" else None
            result["frontend_sha256"] = {
                str(path.relative_to(os.environ["TORCHSIM_DIR"])): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    Path(os.environ["TORCHSIM_DIR"]) / "PyTorchSimFrontend/extension_codecache.py",
                    Path(os.environ["TORCHSIM_DIR"]) / "PyTorchSimFrontend/mlir/mlir_ops.py",
                    Path(os.environ["TORCHSIM_DIR"]) / "PyTorchSimFrontend/mlir/mlir_common.py",
                    Path(os.environ["TORCHSIM_DIR"]) / "PyTorchSimFrontend/mlir/mlir_gemm_template.py",
                    Path(os.environ["TORCHSIM_DIR"]) / "PyTorchSimFrontend/mlir/mlir_bmm_template.py",
                    Path(os.environ["TORCHSIM_DIR"]) / "Simulator/simulator.py",
                    *[Path(__file__).parent / name for name in (
                        "runner.py", "config.py", "validation.py", "submission.py", "provenance.py",
                        "analysis/memory_audit.py", "workloads/attention.py",
                        "workloads/common.py", "workloads/components.py",
                        "workloads/cpu_reference.py", "tests/integration/attention_components.py",
                        "workloads/decoder.py",
                    )],
                ]
            }
            torch.manual_seed(args.seed)
            torch.set_num_threads(4)
            with torch.no_grad():
                operation = cpu_reference if args.mode == "cpu" and args.component not in ("decoder", "attention", "attention_parts") else run_workload
                result["probe"] = operation(args, manifest, torch)
        result["status"] = ("completed_with_numerical_mismatch"
                            if result.get("probe", {}).get("numerical_status") == "failed" else "passed")
    except Exception as error:
        result["status"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
    result["wall_seconds"] = time.monotonic() - started
    if args.output_dir:
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    # Keep terminal summaries short; full errors and manifests remain in the run directory.
    print(json.dumps({key: result[key] for key in ("model_id", "status", "wall_seconds")}, indent=2))
    if args.mode == "audit":
        print(json.dumps(result["audit"], indent=2))
    # Zero means the requested execution completed, not that warnings passed.
    # A timing run still has to pass the launcher's separate dependency audit.
    return int(result["status"] not in ("passed", "completed_with_numerical_mismatch"))


if __name__ == "__main__":
    raise SystemExit(main())
