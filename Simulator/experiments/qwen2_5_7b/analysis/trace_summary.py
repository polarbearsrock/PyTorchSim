"""Reconstruct single-core/two-MXU queue occupancy, not useful-FLOP utilization."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

from .dependency_audit import audit_run


EVENT = re.compile(r"\[(\d+)\]\[Core (\d+)\]\[(INST_ISSUED|INST_FINISHED)\s*\]\[INST_ID=(\d+)\] COMP \(compute_type=(\d+) compute_cycle=(\d+) overlapping_cycle=(\d+)\)")
COMPLETE = re.compile(r"Kernel (\d+) has completed .* operation: (\S+) finished at cycle (\d+)")


def phase_file(result):
    name = result.get("probe", {}).get("phase_file", "attention_phases.json")
    if name not in ("attention_phases.json", "decoder_phases.json"):
        raise ValueError(f"Unsupported phase metadata file: {name}")
    return name


def check_run_status(result, allow_numerical_mismatch=False):
    if result["status"] == "passed":
        return
    if (allow_numerical_mismatch and result["status"] == "completed_with_numerical_mismatch"
            and result.get("arguments", {}).get("allow_numerical_mismatch")
            and result.get("probe", {}).get("numerical_status") == "failed"):
        return
    raise ValueError("Run did not pass; a completed exploratory run requires --allow-numerical-mismatch")


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def overlap(intervals, start, end):
    return sum(max(0, min(end, b) - max(start, a)) for a, b in intervals)


def activity_state_cycles(occupied, start, end):
    """Exact simultaneous queue states; idle means all three compute queues empty."""
    units = ("VPU", "MXU0", "MXU1")
    labels = ["+".join(unit for bit, unit in enumerate(units) if mask & (1 << bit)) or "none"
              for mask in range(8)]
    events = {start: [], end: []}
    for bit, unit in enumerate(units):
        for a, b in merge(occupied[unit]):
            a, b = max(start, a), min(end, b)
            if a < b:
                events.setdefault(a, []).append((bit, 1))
                events.setdefault(b, []).append((bit, -1))
    result = dict.fromkeys(labels, 0)
    active, previous = 0, start
    for cycle in sorted(events):
        result[labels[active]] += cycle - previous
        for bit, delta in events[cycle]:
            active += delta * (1 << bit)
        previous = cycle
    assert active == 0 and sum(result.values()) == end - start
    return result


def check_matrix_nodes(run):
    # A zero-cycle matrix node advances Core's round-robin selector but has no
    # issue event. Refuse to guess the MXU assignment if any such node exists.
    import onnx
    from onnx import helper

    count = 0

    def walk(graph):
        nonlocal count
        for node in graph.node:
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if attrs.get("torchsim_compute_type", 0):
                count += 1
                assert attrs.get("torchsim_cycle", 0) > 0, "Zero-cycle matrix node: cannot infer MXU assignment"
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    walk(attribute.g)

    paths = list((run / "generated").glob("*/tile_graph.onnx"))
    assert paths, "No final tile graphs to check"
    for path in paths:
        walk(onnx.load(str(path)).graph)
    return count


def analyze(log):
    jobs, pending, kernels = [], {}, {}
    native, totals, reset_points = {}, [], set()
    tail, rr = {}, 0
    for line in log.splitlines():
        if match := EVENT.search(line):
            cycle, core, event, identity, kind, latency, hidden = match.groups()
            cycle, core, identity, kind, latency, hidden = map(int, (cycle, core, identity, kind, latency, hidden))
            assert core == 0 and kind in (0, 1, 2), "Only this single-core/two-MXU configuration is supported"
            if event == "INST_ISSUED":
                assert identity not in pending and latency > 0
                unit = "VPU" if kind == 0 else f"MXU{rr % 2}"
                rr += kind != 0
                queued = max(0, tail.get(unit, 0) - cycle)
                finish = cycle + queued + latency - min(queued, hidden)
                job = dict(instruction_id=identity, unit=unit, compute_type=kind,
                           issue_cycle=cycle, finish_cycle=finish, compute_cycle=latency,
                           overlapping_cycle=hidden, bubble_cycle=max(0, hidden - queued))
                pending[identity] = job
                jobs.append(job)
                tail[unit] = finish
            else:
                job = pending.pop(identity)
                assert job["finish_cycle"] == cycle, (identity, job["finish_cycle"], cycle)
        if match := COMPLETE.search(line):
            identity, operation, cycle = match.groups()
            kernels[int(identity)] = {"operation": operation, "finish_cycle": int(cycle)}
        if match := re.search(r"Systolic array \[(\d+)\] utilization\(%\): [\d.]+, active_cycles: (\d+)", line):
            native[f"MXU{match[1]}"] = int(match[2])
        if match := re.search(r"Vector unit utilization\(%\): [\d.]+, active cycle: (\d+)", line):
            native["VPU"] = int(match[1])
        if match := re.search(r"Core \[0\] : Total_cycles: (\d+)", line):
            reset_points.add(int(match[1]))
        if match := re.search(r"Total execution cycles: (\d+)", line):
            totals.append(int(match[1]))
    assert not pending and len(totals) == 1 and jobs, "Incomplete or mixed simulation log"
    assert set(native) == {"MXU0", "MXU1", "VPU"}
    total = totals[0]
    occupied = {unit: merge((j["issue_cycle"], j["finish_cycle"]) for j in jobs if j["unit"] == unit) for unit in native}
    # Replay Core.cc's retirement/bubble accounting, including periodic counter
    # resets and saturation at zero. Its VPU counter adds one per retirement;
    # both matrix and vector counters subtract the modeled bubble term.
    replayed = {}
    for unit, intervals in occupied.items():
        flags = bytearray(total)
        for start, end in intervals:
            flags[start:end] = b"\1" * (end - start)
        retirements = {}
        for job in jobs:
            if job["unit"] == unit:
                retirements.setdefault(job["finish_cycle"], []).append(job["bubble_cycle"])
        accumulated = counter = 0
        for cycle, busy in enumerate(flags):
            for bubble in retirements.get(cycle, ()):
                counter = max(0, counter + (unit == "VPU") - bubble)
            counter += busy
            if cycle + 1 in reset_points:
                accumulated += counter
                counter = 0
        replayed[unit] = accumulated + counter
    assert replayed == native, ("Native counter reconciliation failed", replayed, native)
    return jobs, occupied, kernels, total, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-last-kernels", help="Explicit phase endpoints for older runs without recorded kernel_ids")
    parser.add_argument("--window-cycles", type=int, default=5000)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--allow-numerical-mismatch", action="store_true",
                        help="Analyze an explicitly completed exploratory run, preserving its failed numerical status")
    args = parser.parse_args()
    assert args.window_cycles > 0
    args.output_dir = args.output_dir.resolve()
    assert args.output_dir.is_relative_to(Path(os.environ["TMPDIR"]).resolve()), "Outputs must be under TMPDIR"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_result = json.loads((args.run / "result.json").read_text())
    config = run_result["simulator_config"]
    assert config["num_cores"] == 1 and config["num_systolic_array_per_core"] == 2
    check_run_status(run_result, args.allow_numerical_mismatch)
    raw_log = (args.run / "console.log").read_bytes()
    log = raw_log.decode()
    assert "[Indirect Access] Failed" not in log and "[Indirect Access] Invalid" not in log
    matrix_nodes = check_matrix_nodes(args.run)
    dependency_report, _ = audit_run(args.run)
    (args.output_dir / "dependency_audit.json").write_text(json.dumps(dependency_report, indent=2) + "\n")
    assert dependency_report["status"] == "passed", "Inter-kernel dependency audit failed; see dependency_audit.json"
    jobs, occupied, kernels, total, native = analyze(log)
    phases = json.loads((args.run / phase_file(run_result)).read_text())
    if args.phase_last_kernels:
        last_ids = list(map(int, args.phase_last_kernels.split(",")))
        assert len(last_ids) == len(phases)
    else:
        last_ids = [phase["kernel_ids"][-1] for phase in phases]
        phase_ids = [identity for phase in phases for identity in phase["kernel_ids"]]
        assert phase_ids == list(kernels), "Phase metadata must cover each completed kernel exactly once, in order"
    assert last_ids == sorted(last_ids) and last_ids[-1] == max(kernels)
    units = ("VPU", "MXU0", "MXU1")
    result = {"total_cycles": total, "matrix_graph_nodes_checked": matrix_nodes,
              "run_status": run_result["status"],
              "measurement_class": run_result.get("probe", {}).get("measurement_class", "validation-gated"),
              "numerical_validation": run_result.get("probe", {}).get("numerical_validation", "see source run"),
              "core_freq_mhz": config["core_freq_mhz"],
              "trace_analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "log_sha256": hashlib.sha256(raw_log).hexdigest(),
              "occupancy_definition": "union of half-open [issue, finish) intervals: nonempty modeled compute queue, including queued/pipelined work",
              "mxu_assignment": "inferred from verified Core.cc round-robin rule; every predicted finish matches trace; no zero-cycle matrix graph nodes",
              "native_counter_reconciliation": "exact replay including bubbles, VPU retirement increments, and periodic resets",
              "phase_boundary_semantics": "kernel retirement plus DMA-response drain; checked against every instruction event and subsequent kernel dispatch",
              "dependency_audit": "passed for all single-stream kernel boundaries and logged DMA responses",
              "queue_occupied_cycles": {u: overlap(occupied[u], 0, total) for u in units},
              "simultaneous_queue_state_cycles": activity_state_cycles(occupied, 0, total),
              "native_active_cycles": native, "phases": []}
    start = 0
    for phase, last in zip(phases, last_ids):
        end = kernels[last]["finish_cycle"]
        result["phases"].append({"phase": phase["phase"], "start_cycle": start, "end_cycle": end,
                                  "cycles": end - start, "last_kernel_id": last,
                                  "simultaneous_queue_state_cycles": activity_state_cycles(occupied, start, end),
                                  "queue_occupied_fraction": {u: overlap(occupied[u], start, end) / (end - start) for u in units}})
        start = end
    result["post_completion_tail_cycles"] = total - start
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "compute_intervals.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(jobs[0]))
        writer.writeheader()
        writer.writerows(jobs)
    rows = []
    for start in range(0, total, args.window_cycles):
        end = min(total, start + args.window_cycles)
        rows.append({"start_cycle": start, "end_cycle": end, **{u: overlap(occupied[u], start, end) / (end - start) for u in units}})
    with (args.output_dir / "occupancy_windows.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.plot:
        os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / "matplotlib-config"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
        for axis, unit, color in zip(axes, units, ("#8b5cf6", "#0891b2", "#e58b19")):
            axis.stairs([100 * row[unit] for row in rows], [rows[0]["start_cycle"]] + [row["end_cycle"] for row in rows], fill=True, color=color)
            axis.set_ylabel(f"{unit}\nqueue busy (%)")
            axis.set_ylim(0, 105)
            axis.grid(axis="y", alpha=.2)
            for phase in result["phases"]:
                axis.axvline(phase["end_cycle"], color="#555555", linewidth=.7, linestyle="--")
        for phase in result["phases"]:
            axes[0].text((phase["start_cycle"] + phase["end_cycle"]) / 2, 108, phase["phase"], ha="center", fontsize=10)
        axes[-1].set_xlabel(f"Simulated core cycle — {args.window_cycles:,}-cycle averaging windows")
        shape = run_result["arguments"]
        scope = "one decoder layer" if shape.get("component") == "decoder" else "attention only"
        warning = " · EXPLORATORY: CPU numerical check failed" if run_result["status"] != "passed" else ""
        figure.suptitle(f"Qwen2.5-7B {scope} · {shape['dtype']} · {shape['seq_len']}-token prefill + {shape['decode_steps']} decode steps\nQueue occupancy, not useful FLOPs{warning}", fontsize=11)
        figure.savefig(args.output_dir / "queue_occupancy.png", dpi=160)
        plt.close(figure)
        # Preserve a direct interval view as well as the averaged overview.
        # These are queue-residence intervals, not physical MAC activity.
        colors = {"VPU": "#8b5cf6", "MXU0": "#0891b2", "MXU1": "#e58b19"}
        for phase in result["phases"]:
            begin, end = phase["start_cycle"], phase["end_cycle"]
            figure, axis = plt.subplots(figsize=(12, 3.5), layout="constrained")
            for row, unit in enumerate(units):
                intervals = [(max(a, begin), min(b, end)) for a, b in occupied[unit]
                             if a < end and b > begin]
                axis.broken_barh([((a - begin) / config["core_freq_mhz"],
                                  (b - a) / config["core_freq_mhz"]) for a, b in intervals],
                                (row - .25, .5), facecolors=colors[unit], linewidth=0)
            axis.set_yticks(range(3), units)
            axis.set_ylim(2.7, -.7)
            axis.set_xlim(0, (end - begin) / config["core_freq_mhz"])
            axis.set_xlabel(f"Microseconds from {phase['phase']} start (cycle {begin:,}); {config['core_freq_mhz']} MHz")
            axis.grid(axis="x", alpha=.2)
            axis.set_title(f"Qwen2.5-7B {scope} · {phase['phase']} · exact trace intervals\n"
                           f"Colored = nonempty compute queue, not useful arithmetic{warning}", fontsize=10)
            figure.savefig(args.output_dir / f"unit_activity_{phase['phase']}.png", dpi=180)
            plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
