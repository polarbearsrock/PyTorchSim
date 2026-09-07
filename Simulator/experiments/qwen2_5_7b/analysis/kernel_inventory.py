"""Match wrapper calls to launched MLIR and attach audited per-kernel occupancy.

Labels are compiler origins, not guessed semantic attention/MLP operator names.
Generated Python is parsed as data, never imported or executed.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re

from .dependency_audit import audit_run
from .trace_summary import check_run_status, merge, overlap


def metadata_literal(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "torch":
        return f"torch.{node.attr}"
    if isinstance(node, (ast.List, ast.Tuple)):
        return [metadata_literal(value) for value in node.elts]
    return ast.literal_eval(node)


def wrapper_inventory(path):
    tree = ast.parse(path.read_text())
    definitions = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        name = ast.unparse(node.targets[0])
        if not re.fullmatch(r"(?:mlir|extension)_kernel_\d+", name):
            continue
        keywords = {item.arg: item.value for item in node.value.keywords}
        definitions[name] = {
            "source": ast.literal_eval(node.value.args[0]),
            "origins": sorted(ast.literal_eval(keywords["origins"])),
            "arguments": metadata_literal(keywords["arg_attributes"]),
        }
    call_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "call")
    calls = []
    for node in ast.walk(call_function):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name.startswith(("aten.", "extern_kernels.", "torch.ops.")):
                raise ValueError(f"Unaccounted direct operator call in wrapper: {name}")
    for node in call_function.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name = ast.unparse(node.value.func)
            if name in definitions:
                entry = dict(definitions[name], wrapper=str(path), wrapper_call_line=node.lineno,
                             function=name, call=ast.unparse(node.value),
                             argument_expressions=[ast.unparse(value) for value in node.value.args])
                calls.append(entry)
    if not calls:
        raise ValueError(f"No submitted kernels in wrapper {path}")
    return calls


def inventory(run):
    calls, phase = [], None
    for line in (run / "console.log").read_text().splitlines():
        if match := re.search(r"Starting decoder (\w+):", line):
            phase = match[1]
        if match := re.search(r"Wrapper Codegen Path = (\S+)", line):
            if phase is None:
                raise ValueError("Wrapper appeared without a decoder phase")
            calls.extend(dict(call, phase=phase) for call in wrapper_inventory(Path(match[1])))
    trace_path, = (run / "logs").glob("*.trace")
    launches = [row for row in csv.reader(trace_path.read_text().splitlines())
                if row and row[0] == "LAUNCH_KERNEL"]
    if len(calls) != len(launches):
        raise ValueError(f"Wrapper/launch count mismatch: {len(calls)} / {len(launches)}")
    for call, launch in zip(calls, launches):
        if launch[2:4] != ["0", "0"]:
            raise ValueError("Inventory supports device 0, stream 0")
        source_path, = [path for path in Path(launch[4]).parent.glob("*.mlir")
                       if re.fullmatch(r"c[a-z0-9]+\.mlir", path.name)]
        source = call.pop("source")
        if source.strip() != source_path.read_text().strip():
            raise ValueError(f"Wrapper/launch source mismatch at kernel {launch[1]}")
        dimensions = {key: int(value) for key, value in re.findall(
            r"^// (M|N|K|TILE_M|TILE_N|TILE_K|SUB_TILE_M|SUB_TILE_N) = (\d+)$", source, re.M)}
        call.update(kernel_id=int(launch[1]), source_path=str(source_path),
                    source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    runtime_dir=str(Path(launch[5]).parent.parent), dimensions=dimensions,
                    contains_matrix_operation="linalg.matmul" in source or "linalg.batch_matmul" in source)
    return calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("analysis", type=Path, help="Existing trace_summary output directory")
    parser.add_argument("--allow-numerical-mismatch", action="store_true")
    args = parser.parse_args()
    output = args.analysis.resolve()
    if not output.is_relative_to(Path(os.environ["TMPDIR"]).resolve()):
        parser.error("Outputs must be under TMPDIR")
    run_result = json.loads((args.run / "result.json").read_text())
    check_run_status(run_result, args.allow_numerical_mismatch)
    summary = json.loads((output / "summary.json").read_text())
    audit, instructions = audit_run(args.run)
    assert audit["status"] == "passed" and audit["log_sha256"] == summary["log_sha256"]
    kernels = inventory(args.run)
    assert [k["kernel_id"] for k in kernels] == [k["kernel_id"] for k in audit["kernels"]]
    phase_ids = [(p["phase"], identity) for p in json.loads((args.run / "decoder_phases.json").read_text())
                 for identity in p["kernel_ids"]]
    assert phase_ids == [(k["phase"], k["kernel_id"]) for k in kernels]
    with (output / "compute_intervals.csv").open() as stream:
        jobs = list(csv.DictReader(stream))
    by_kernel = {kernel["kernel_id"]: [] for kernel in kernels}
    for job in jobs:
        identity = instructions[int(job["instruction_id"])]["kernel_id"]
        by_kernel[identity].append(job)
    rows = []
    for kernel, timing in zip(kernels, audit["kernels"]):
        begin, end = timing["dispatch_start"], timing["completion"]
        occupied = {unit: merge((int(job["issue_cycle"]), int(job["finish_cycle"]))
                               for job in by_kernel[kernel["kernel_id"]] if job["unit"] == unit)
                    for unit in ("VPU", "MXU0", "MXU1")}
        kernel.update(start_cycle=begin, end_cycle=end, cycles=end - begin,
                      queue_occupied_cycles={unit: overlap(intervals, begin, end) for unit, intervals in occupied.items()})
        rows.append({"phase": kernel["phase"], "kernel_id": kernel["kernel_id"],
                     "origins": ";".join(kernel["origins"]), "start_cycle": begin,
                     "end_cycle": end, "cycles": end - begin,
                     **kernel["queue_occupied_cycles"], "source_path": kernel["source_path"]})
    result = {"source_run": str(args.run), "run_status": run_result["status"],
              "log_sha256": summary["log_sha256"], "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "wrapper_launch_source_coverage": "all calls matched in order, with byte-matched MLIR; no direct ATen/extern wrapper calls",
              "labels": "compiler origins; no semantic operator names inferred from ordinal positions",
              "kernels": kernels}
    (output / "kernel_inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "kernel_queue_occupancy.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"kernels_checked": len(kernels), "status": "passed", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
