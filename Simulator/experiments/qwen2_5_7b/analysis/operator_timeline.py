"""Plot source-attributed attention operators after the dependency audit passes."""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re

from .dependency_audit import audit_run
from .trace_summary import merge


PREFILL = [
    ("Q projection", "addmm"), ("Negate Q half", "neg"),
    ("Copy Q half", "cat"), ("Concatenate rotated Q", "cat"),
    ("K projection", "addmm_1"), ("Negate K half", "neg_1"),
    ("Copy K half", "cat_1"), ("Concatenate rotated K", "cat_1"),
    ("K RoPE", "add_1"), ("Q RoPE", "add"),
    ("K GQA expansion", "clone"), ("Attention scores: Q K^T", "bmm"),
    ("Softmax: scale/mask/max", "amax"), ("Softmax: exp/sum", "sum_1"),
    ("Softmax: normalize", "div_1"), ("V projection", "addmm_1"),
    ("V GQA expansion", "clone"), ("Weighted values: P V", "bmm_1"),
    ("Head merge", "clone_3"), ("Output projection", "mm"),
]
DECODE = (PREFILL[:9] + [("K cache append", "cat_2")] + PREFILL[9:16] +
          [("V cache layout", "cat_3"), ("V cache append", "cat_2")] +
          PREFILL[16:18] + PREFILL[19:])


def wrapper_calls(path):
    module = ast.parse(path.read_text())
    definitions = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        name = ast.unparse(node.targets[0])
        if not re.fullmatch(r"(?:mlir|extension)_kernel_\d+", name):
            continue
        keywords = {k.arg: k.value for k in node.value.keywords}
        definitions[name] = dict(source=ast.literal_eval(node.value.args[0]),
                                 origins=ast.literal_eval(keywords["origins"]))
    call_fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "call")
    calls = []
    for node in call_fn.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id in definitions:
                calls.append(dict(definitions[call.func.id], call=ast.unparse(call), line=node.lineno))
    return calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("analysis", type=Path, help="trace_summary.py output directory")
    args = parser.parse_args()
    if (args.run / "decoder_phases.json").exists():
        parser.error("The fixed operator labels here describe only the historical attention diagnostic. "
                     "Use trace_summary for decoder occupancy; decoder operator attribution is not implemented.")
    output = args.analysis.resolve()
    if not output.is_relative_to(Path(os.environ["TMPDIR"]).resolve()):
        parser.error("Output must be under TMPDIR")
    report, instructions = audit_run(args.run)
    assert report["status"] == "passed", report["errors"][:5]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["log_sha256"] == report["log_sha256"]
    phases = json.loads((args.run / "attention_phases.json").read_text())
    log = (args.run / "console.log").read_text()
    wrappers = [Path(p) for p in re.findall(r"Wrapper Codegen Path = (\S+)", log)]
    assert len(wrappers) == len(phases)
    trace_path, = (args.run / "logs").glob("*.trace")
    launches = [row for row in csv.reader(trace_path.read_text().splitlines()) if row and row[0] == "LAUNCH_KERNEL"]
    kernel_map = {k["kernel_id"]: k for k in report["kernels"]}
    cursor = 0
    for phase, wrapper in zip(phases, wrappers):
        labels = PREFILL if phase["phase"] == "prefill" else DECODE
        calls = wrapper_calls(wrapper)
        assert len(calls) == len(labels) == len(phase["kernel_ids"])
        for identity, call, (label, origin) in zip(phase["kernel_ids"], calls, labels):
            launch = launches[cursor]
            cursor += 1
            assert int(launch[1]) == identity and origin in call["origins"]
            source_path, = [p for p in Path(launch[4]).parent.glob("*.mlir")
                            if re.fullmatch(r"c[a-z0-9]+\.mlir", p.name)]
            assert call["source"].strip() == source_path.read_text().strip()
            kernel_map[identity].update(phase=phase["phase"], label=label,
                                       wrapper=str(wrapper), wrapper_call_line=call["line"],
                                       call=call["call"], origins=sorted(call["origins"]),
                                       source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
    assert cursor == len(launches)
    with (output / "compute_intervals.csv").open() as stream:
        jobs = list(csv.DictReader(stream))
    for job in jobs:
        for key in ("instruction_id", "issue_cycle", "finish_cycle"):
            job[key] = int(job[key])
        job["kernel_id"] = instructions[job["instruction_id"]]["kernel_id"]
        job["operator"] = kernel_map[job["kernel_id"]]["label"]
        job["phase"] = kernel_map[job["kernel_id"]]["phase"]
    with (output / "operator_compute_intervals.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(jobs[0]))
        writer.writeheader()
        writer.writerows(jobs)
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib-config"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"VPU": "#8b5cf6", "MXU0": "#0891b2", "MXU1": "#e58b19", "DMA lifetime": "#c5cbd3"}
    plots = {}
    for phase_window in summary["phases"]:
        phase = phase_window["phase"]
        kernels = [k for k in kernel_map.values() if k["phase"] == phase]
        begin, end = phase_window["start_cycle"], phase_window["end_cycle"]
        figure, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=True,
                                    gridspec_kw={"width_ratios": [1.1, 1]}, layout="constrained")
        for axis in axes:
            for row, kernel in enumerate(kernels):
                identity = kernel["kernel_id"]
                if row % 2 == 0:
                    axis.axhspan(row - .47, row + .47, color="#f5f6f7", zorder=0)
                for unit_index, unit in enumerate(("VPU", "MXU0", "MXU1")):
                    intervals = merge((j["issue_cycle"], j["finish_cycle"]) for j in jobs
                                      if j["kernel_id"] == identity and j["unit"] == unit)
                    axis.broken_barh([((a - begin) / 1000, (b - a) / 1000) for a, b in intervals],
                                    (row - .33 + .18 * unit_index, .15), facecolors=colors[unit], linewidth=0)
                memory = [i for i in instructions if i["kernel_id"] == identity
                          and i["opcode"] in ("MOVIN", "MOVOUT") and "INST_ISSUED" in i["events"]]
                intervals = merge((i["events"]["INST_ISSUED"],
                                   i["events"].get("DRAM_RESP_DONE", i["events"].get("INST_FINISHED")))
                                  for i in memory)
                axis.broken_barh([((a - begin) / 1000, (b - a) / 1000) for a, b in intervals],
                                (row + .24, .09), facecolors=colors["DMA lifetime"], linewidth=0)
            axis.set_ylim(len(kernels) - .4, -.7)
            axis.grid(axis="x", alpha=.2)
            axis.set_xlabel(f"Thousands of core cycles from {begin:,}")
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].set_yticks(range(len(kernels)), [f"K{k['kernel_id']}  {k['label']}" for k in kernels])
        axes[0].set_xlim(0, (end - begin) / 1000)
        axes[0].set_title("Whole phase")
        axes[1].set_xlim((kernels[1]["dispatch_start"] - begin) / 1000,
                        (kernels[-2]["completion"] - begin) / 1000)
        axes[1].set_title("Between Q projection and output projection")
        figure.suptitle(f"Qwen2.5-7B attention only · {phase} · corrected kernel completion\n"
                       "Colored: compute queue residence, not useful FLOPs. Gray: DMA lifetime through response, not bus utilization.", fontsize=11)
        axes[0].legend(handles=[Patch(color=color, label=label) for label, color in colors.items()],
                       loc="lower left", bbox_to_anchor=(0, -.14), ncol=4, frameon=False, fontsize=9)
        path = output / f"attention_operators_{phase}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        plots[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(path)
    (output / "operator_attribution.json").write_text(json.dumps(
        dict(log_sha256=report["log_sha256"], script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             kernels=list(kernel_map.values()), plots=plots), indent=2) + "\n")


if __name__ == "__main__":
    main()
