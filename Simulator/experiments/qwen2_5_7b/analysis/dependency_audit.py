"""Check single-stream kernel completion against every instruction and DMA response.

Uses launch attributes and all parser-constructor categories to map global
instruction IDs to kernels; never executes generated wrappers. This checks
inter-kernel ordering, not numerical correctness or physical TPU accuracy.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

CREATION = re.compile(r"\[TOGParser\] (?:(Load|Store) Node (\S+) Numa_id: \d+:|(Compute|DMA Wait) Node :)(.*)")
ATTRIBUTE = re.compile(r'\[LoadConfig\] Loaded configuration file "([^"]+/runtime_\d+/attribute/\d+)"')
EVENT = re.compile(r"\[(\d+)\]\[Core (\d+)\]\[([^\]]+)\]\[INST_ID=(\d+)\] (MOVIN|MOVOUT|COMP|BAR)\b")
COMPLETE = re.compile(r"Kernel (\d+) has completed .* operation: (\S+) finished at cycle (\d+)")
START = re.compile(r"Kernel (\d+) execution summary - Started at: (\d+) cycles")


def audit(log, launches):
    lookup = {str(row["attribute"]): row["kernel_id"] for row in launches}
    order = [row["kernel_id"] for row in launches]
    if len(set(order)) != len(order):
        raise ValueError("Duplicate kernel launch ID")
    kernels = {identity: dict(kernel_id=identity, first_issue=None, last_event=0,
                              completion=None, dispatch_start=None, operation=None)
               for identity in order}
    instructions, errors, seen = [], [], []
    current = None
    for line_number, line in enumerate(log.splitlines(), 1):
        if match := ATTRIBUTE.search(line):
            current = lookup[match[1]]
            seen.append(current)
        if match := CREATION.search(line):
            kind = match[1] or match[3]
            if current is None:
                raise ValueError("Instruction constructor before kernel attribute load")
            instructions.append(dict(instruction_id=len(instructions), kernel_id=current,
                                     opcode={"Load": "MOVIN", "Store": "MOVOUT", "Compute": "COMP", "DMA Wait": "BAR"}[kind],
                                     address_argument=match[2], events={}, parser_line=line_number))
        if match := EVENT.search(line):
            cycle, core, tag, identity, opcode = match.groups()
            cycle, core, identity = map(int, (cycle, core, identity))
            if core != 0:
                raise ValueError("Audit supports the single-core baseline only")
            inst = instructions[identity]
            if inst["opcode"] != opcode:
                raise ValueError(f"Constructor/trace opcode mismatch at line {line_number}")
            if opcode in ("MOVIN", "MOVOUT") and (address := re.search(r"addr_name=(\w+)", line)):
                if address[1] != inst["address_argument"]:
                    raise ValueError(f"Memory argument mismatch at line {line_number}")
            tag = tag.strip()
            if tag in inst["events"]:
                raise ValueError(f"Duplicate {tag} for instruction {identity}")
            inst["events"][tag] = cycle
            if tag == "INST_ISSUED":
                inst["async"] = "async=true" in line
                if address := re.search(r"dram=(0x[0-9a-f]+)", line):
                    inst["address"] = address[1]
            kernel = kernels[inst["kernel_id"]]
            kernel["last_event"] = max(kernel["last_event"], cycle)
            if tag in ("INST_ISSUED", "INST_SKIP"):
                kernel["first_issue"] = cycle if kernel["first_issue"] is None else min(kernel["first_issue"], cycle)
        if match := COMPLETE.search(line):
            identity, operation, cycle = match.groups()
            kernel = kernels[int(identity)]
            if kernel["completion"] is not None:
                raise ValueError("Duplicate kernel completion")
            kernel.update(completion=int(cycle), operation=operation)
        if match := START.search(line):
            kernels[int(match[1])]["dispatch_start"] = int(match[2])
    if seen != order or not instructions:
        raise ValueError("Parser kernel order does not match complete launch trace")
    store_responses = dma_transfers = 0
    for inst in instructions:
        events = inst["events"]
        if not events:
            errors.append(f"Instruction {inst['instruction_id']}: no events (unsupported skipped node or incomplete trace)")
            continue
        if "INST_ISSUED" not in events:
            if "INST_SKIP" not in events:
                errors.append(f"Instruction {inst['instruction_id']}: no issue/skip event")
            continue
        required = "INST_FINISHED"
        if inst["opcode"] == "MOVOUT" or (inst["opcode"] == "MOVIN" and inst["async"]):
            required = "DRAM_RESP_DONE"
        if required not in events:
            errors.append(f"Instruction {inst['instruction_id']} {inst['opcode']}: missing {required}")
        elif events[required] < events["INST_ISSUED"]:
            errors.append(f"Instruction {inst['instruction_id']}: completion before issue")
        if inst["opcode"] in ("MOVIN", "MOVOUT"):
            dma_transfers += 1
        store_responses += inst["opcode"] == "MOVOUT" and "DRAM_RESP_DONE" in events
    previous = None
    for identity in order:
        kernel = kernels[identity]
        if kernel["completion"] is None or kernel["dispatch_start"] is None:
            errors.append(f"Kernel {identity}: missing completion/start")
        elif kernel["completion"] < kernel["last_event"]:
            errors.append(f"Kernel {identity}: completed at {kernel['completion']} before last event at {kernel['last_event']}")
        if previous is not None and previous["completion"] is not None:
            for key in ("dispatch_start", "first_issue"):
                if kernel[key] is not None and kernel[key] < previous["completion"]:
                    errors.append(f"Kernel {identity}: {key}={kernel[key]} before kernel {previous['kernel_id']} completion={previous['completion']}")
        previous = kernel
    return dict(status="failed" if errors else "passed", errors=errors,
                kernel_count=len(order), instruction_count=len(instructions),
                inter_kernel_boundaries_checked=max(0, len(order) - 1),
                dma_transfers_checked=dma_transfers, store_response_events_checked=store_responses,
                kernels=list(kernels.values())), instructions


def audit_run(run, log_path=None):
    trace_path, = (run / "logs").glob("*.trace")
    launches = []
    for row in csv.reader(trace_path.read_text().splitlines()):
        if row and row[0] == "LAUNCH_KERNEL":
            if row[2:4] != ["0", "0"]:
                raise ValueError("Audit supports device 0, stream 0 only")
            launches.append(dict(kernel_id=int(row[1]), attribute=row[5]))
    log_path = log_path or run / "console.log"
    raw = log_path.read_bytes()
    report, instructions = audit(raw.decode(), launches)
    report["log_sha256"] = hashlib.sha256(raw).hexdigest()
    report["log_path"] = str(log_path)
    report["audit_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["scope"] = "All single-stream kernel boundaries and logged DMA responses; not intra-kernel memory hazards or hardware calibration"
    return report, instructions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--log", type=Path, help="Audit a standalone replay of the saved launch trace")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path(os.environ["TMPDIR"]).resolve()):
        parser.error("Output must be under TMPDIR")
    output.mkdir(parents=True, exist_ok=True)
    report, instructions = audit_run(args.run.resolve(), args.log)
    (output / "dependency_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "instruction_events.json").write_text(json.dumps(instructions, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("kernels", "errors")}, indent=2))
    for error in report["errors"][:10]:
        print(error)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
