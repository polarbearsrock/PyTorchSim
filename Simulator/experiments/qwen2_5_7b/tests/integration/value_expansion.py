"""Exact upstream repeat_kv/cast regressions for the decode value-cache path.

Run through run.sh toolchain; all generated kernels and reports stay in TMPDIR.
This is an isolated compiler test, not a replacement model implementation.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from ...config import configure_simulator


def check_saved_run(run, output_dir):
    """Validate the actual expansion buffers from a complete decoder trace."""
    import numpy as np
    import torch
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv
    from ...analysis.kernel_inventory import inventory

    phases = json.loads((run / "decoder_phases.json").read_text())
    expected_phases = {phase["phase"] for phase in phases if phase["phase"].startswith("decode_")}
    results = []
    for row in inventory(run):
        if row["phase"] not in expected_phases or row["origins"] != ["convert_element_type_default_2"]:
            continue
        if len(row["arguments"]) != 2:
            raise ValueError("Unexpected value-expansion operands")
        operands = []
        for (name, (mode, dtype, count, shape, stride)), expected_mode, expected_dtype in zip(
            row["arguments"], (1, 2), ("torch.bfloat16", "torch.float32")
        ):
            if (mode, dtype) != (expected_mode, expected_dtype):
                raise ValueError("Unexpected value-expansion operand role or dtype")
            raw = torch.from_numpy(np.fromfile(
                Path(row["runtime_dir"]) / name / "0.raw",
                dtype=np.uint16 if dtype == "torch.bfloat16" else np.float32))
            if dtype == "torch.bfloat16":
                raw = raw.view(torch.bfloat16)
            if raw.numel() != count:
                raise ValueError("Saved operand size does not match wrapper metadata")
            operands.append(raw.as_strided(shape, stride))
        values, actual = operands
        if values.ndim != 4 or actual.ndim != 4 or values.shape[1] != 4 or actual.shape[1] != 28:
            raise ValueError("Unexpected Qwen value-head expansion shape")
        expected = repeat_kv(values, 7).float()
        if actual.shape != expected.shape:
            raise ValueError("Expanded value shape mismatch")
        mismatch = actual.view(torch.int32) != expected.view(torch.int32)
        mismatches = int(mismatch.sum())
        record = {
            "phase": row["phase"], "kernel_id": row["kernel_id"],
            "source_path": row["source_path"], "source_sha256": row["source_sha256"],
            "input_shape": list(values.shape), "output_shape": list(actual.shape),
            "max_abs_error": float((actual - expected).abs().max()),
            "mismatched_elements": mismatches, "elements": actual.numel(),
            "mismatched_elements_per_head": mismatch.sum(dim=(0, 2, 3)).tolist(),
            "status": "passed" if mismatches == 0 else "failed",
        }
        results.append(record)
        print(json.dumps(record), flush=True)
    if not expected_phases or len(results) != len(expected_phases) or {row["phase"] for row in results} != expected_phases:
        raise ValueError("Expected exactly one identified value expansion per decode phase")
    failed = any(row["status"] != "passed" for row in results)
    report = {"source_run": str(run), "status": "failed" if failed else "passed",
              "expected_operation": "Transformers repeat_kv(saved_input, 7).float()",
              "comparison": "bit-exact FP32", "checks": results}
    (output_dir / "value_expansion_saved.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--case", default="all")
    mode.add_argument("--saved-run", type=Path, help="Check saved decoder operands without rerunning kernels")
    args = parser.parse_args()
    output_dir = Path(os.environ["TMPDIR"])
    if args.saved_run is not None:
        check_saved_run(args.saved_run, output_dir)
    configure_simulator(SimpleNamespace(mode="functional", output_dir=output_dir))

    import torch
    from transformers.models.qwen2.modeling_qwen2 import repeat_kv

    torch.manual_seed(7)
    torch.set_num_threads(4)
    # name: batch, KV heads, cache length, head dimension, repetitions, layout, cast
    cases = {
        "prefill": (1, 4, 8, 128, 7, "contiguous", True),
        "decode_1": (1, 4, 9, 128, 7, "contiguous", True),
        "decode_2": (1, 4, 10, 128, 7, "contiguous", True),
        "longer_cache": (1, 4, 17, 128, 7, "contiguous", True),
        "projection_layout": (1, 4, 9, 128, 7, "transposed", True),
        "batch_two": (2, 4, 9, 128, 7, "contiguous", True),
        "two_repetitions": (1, 4, 9, 128, 2, "contiguous", True),
        "no_replication": (1, 4, 9, 128, 1, "contiguous", True),
        "bf16_copy": (1, 4, 9, 128, 7, "contiguous", False),
    }
    if args.case != "all" and args.case not in cases:
        parser.error(f"Unknown case: {args.case}; choose from {', '.join(cases)}")
    report = {}
    with torch.no_grad():
        for name, (batch, heads, length, dim, groups, layout, cast) in cases.items():
            if args.case not in ("all", name):
                continue
            if layout == "transposed":
                values = torch.randn(batch, length, heads, dim, dtype=torch.bfloat16).transpose(1, 2)
            else:
                values = torch.randn(batch, heads, length, dim, dtype=torch.bfloat16)

            def expand(value):
                expanded = repeat_kv(value, groups)
                return expanded.float() if cast else expanded

            expected = expand(values)
            # Independent CPU check of the required head-replication ordering.
            reference = torch.repeat_interleave(values, groups, dim=1).to(expected.dtype)
            torch.testing.assert_close(expected, reference, rtol=0, atol=0)
            entry = {"shape": list(values.shape), "stride": list(values.stride()),
                     "repetitions": groups, "output_dtype": str(expected.dtype)}
            try:
                # Each shape/layout is an independent test, not repeated
                # specialization of one production graph with a cache limit.
                torch.compiler.reset()
                compiled = torch.compile(expand, fullgraph=True, dynamic=False)
                actual = compiled(values.to("npu:0")).cpu()
                entry.update({
                    "max_abs_error": float((actual.float() - expected.float()).abs().max()),
                    "mismatched_elements": int((actual != expected).sum()),
                    "elements": expected.numel(),
                })
                # Include sign bits: this is a copy and an exact widening cast.
                word_type = torch.int32 if cast else torch.int16
                torch.testing.assert_close(actual.view(word_type), expected.view(word_type), rtol=0, atol=0)
                entry["status"] = "passed"
            except Exception as error:
                entry.update(status="failed", error=f"{type(error).__name__}: {error}")
            report[name] = entry
            (output_dir / "value_expansion.json").write_text(json.dumps(report, indent=2) + "\n")
            print(name, entry["status"], json.dumps(entry), flush=True)
    raise SystemExit(any(case["status"] != "passed" for case in report.values()))


if __name__ == "__main__":
    main()
