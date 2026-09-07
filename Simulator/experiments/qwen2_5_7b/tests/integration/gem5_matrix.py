"""Bounded gem5 matrix-queue regressions and replay of saved timing binaries.

These are readiness-accounting tests, not numerical GEMM validation. Every
case must terminate and account for each push/pop's actual VL exactly.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


STUDY = Path(__file__).resolve().parents[2]
NORMAL_EXIT = "exiting with last active thread context"
LMUL = {"mf2": 0.5, "m1": 1, "m2": 2, "m4": 4}


@dataclass(frozen=True)
class Case:
    name: str
    feed_bits: int
    feed_lmul: str
    feed_avl: int
    pushes: int
    pop_lmul: str
    pop_avl: int
    pops: int

    @property
    def feed_vl(self):
        return min(self.feed_avl, int(256 * LMUL[self.feed_lmul] / self.feed_bits))

    @property
    def pop_vl(self):
        return min(self.pop_avl, int(256 * LMUL[self.pop_lmul] / 32))


CASES = (
    Case("bf16_vl0", 16, "m1", 0, 1, "m1", 0, 1),
    Case("bf16_vl2", 16, "m1", 2, 1, "m1", 2, 1),
    Case("bf16_vl7", 16, "m1", 7, 1, "m1", 7, 1),
    Case("bf16_s8", 16, "m1", 8, 1, "m1", 8, 1),
    Case("bf16_s16", 16, "m1", 16, 1, "m1", 8, 2),
    Case("bf16_s128", 16, "m1", 16, 8, "m1", 16, 16),
    Case("bf16_mf2_clamped", 16, "mf2", 16, 1, "m1", 16, 1),
    Case("bf16_m2", 16, "m2", 32, 4, "m1", 8, 16),
    Case("bf16_m4", 16, "m4", 64, 2, "m1", 8, 16),
    Case("fp32_s128", 32, "m1", 8, 16, "m1", 8, 16),
    Case("fp32_m2", 32, "m2", 16, 8, "m2", 16, 8),
)


def assembly(case):
    # Encodings match the fork's VCIX decoder. The 16-bit feed selector is
    # rs1=1 (BF16); results always use FP32. No numerical result is consumed.
    selector = 0x8000 if case.feed_bits == 16 else 0
    return f""".section .text
.globl _start
.type _start, @function
_start:
    li t0, {256 // case.feed_bits}
    vsetvli zero, t0, e{case.feed_bits}, m1, ta, ma
    vmv.v.i v8, 0
    .rept {128 // (256 // case.feed_bits)}
    .word {0x2680305b | selector:#x}
    .endr
    li t0, {case.feed_avl}
    vsetvli zero, t0, e{case.feed_bits}, {case.feed_lmul}, ta, ma
    vmv.v.i v8, 0
    .rept {case.pushes}
    .word {0x2280305b | selector:#x}
    .endr
    .word 0x0602305b
    li t0, {case.pop_avl}
    vsetvli zero, t0, e32, {case.pop_lmul}, ta, ma
    .rept {case.pops}
    .word 0x0800345b
    .endr
    li a0, 0
    li a7, 93
    ecall
.size _start, .-_start
"""


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def queue_counts(trace):
    return {
        "input_sizes": [int(x) for x in re.findall(r"pushInput: input size: (\d+)", trace)],
        "pop_sizes": [int(x) for x in re.findall(r"Size before vpop\((\d+)\)", trace)],
        "remaining": [int(x) for x in re.findall(r"Size after vpop: (\d+)", trace)],
    }


def run_probe(gem5, binary, output, ticks, debug=False):
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, QWEN_GEM5_PROBE_BINARY=str(binary),
               QWEN_GEM5_PROBE_TICKS=str(ticks),
               QWEN_GEM5_PROBE_RESULT=str(output / "event.json"))
    command = [gem5, "-d", str(output / "m5out")]
    if debug:
        command += ["--debug-flags=SystolicArray", "--debug-end=2000000", "--debug-file=systolic.log"]
    command.append(str(STUDY / "tools/gem5/bounded.py"))
    result = {"binary": str(binary), "binary_sha256": sha256(binary), "tick_limit": ticks}
    try:
        with (output / "console.log").open("w") as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=90)
        result["returncode"] = process.returncode
    except subprocess.TimeoutExpired:
        result["error"] = "host wall-clock limit exceeded"
    event_path = output / "event.json"
    if event_path.exists():
        result.update(json.loads(event_path.read_text()))
    result["completed"] = result.get("returncode") == 0 and result.get("cause") == NORMAL_EXIT
    stats_path = output / "m5out/stats.txt"
    if stats_path.exists():
        result["cycle_checkpoints"] = [int(x) for x in re.findall(r"^system.cpu.numCycles\s+(\d+)", stats_path.read_text(), re.M)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gem5", default=os.environ.get("GEM5_PATH", "/gem5/release/gem5.opt"))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ["TMPDIR"]) / "gem5-matrix")
    parser.add_argument("--replay-binary", type=Path, action="append", default=[])
    parser.add_argument("--replay-ticks", type=int, default=5000000000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(Path(os.environ["TMPDIR"]).resolve()) or args.replay_ticks <= 0:
        parser.error("Output must be under TMPDIR and the tick limit must be positive")
    output.mkdir(parents=True, exist_ok=False)
    result = {"gem5": {"path": args.gem5, "sha256": sha256(Path(args.gem5))}, "cases": []}
    for case in CASES:
        source = output / f"{case.name}.S"
        binary = output / f"{case.name}.bin"
        source.write_text(assembly(case))
        subprocess.run(["riscv64-unknown-elf-gcc", "-march=rv64gcv", "-mabi=lp64d", "-nostdlib", "-static",
                        "-Wl,-e,_start", str(source), "-o", str(binary)], check=True)
        measured = run_probe(args.gem5, binary, output / case.name, 20000000, debug=True)
        trace_path = output / case.name / "m5out/systolic.log"
        measured.update(queue_counts(trace_path.read_text() if trace_path.exists() else ""))
        measured["case"] = asdict(case)
        measured["passed"] = (measured["completed"] and
                              measured["input_sizes"] == [case.feed_vl] * case.pushes and
                              measured["pop_sizes"] == [case.pop_vl] * case.pops and
                              measured["remaining"][-1:] == [0])
        result["cases"].append(measured)
        print(case.name, "PASS" if measured["passed"] else "FAIL", measured.get("cause"), flush=True)
    result["replays"] = []
    for index, binary in enumerate(args.replay_binary):
        measured = run_probe(args.gem5, binary.resolve(), output / f"replay-{index}", args.replay_ticks)
        result["replays"].append(measured)
        print("replay", binary, "PASS" if measured["completed"] else "FAIL", flush=True)
    result["passed"] = all(x["passed"] for x in result["cases"]) and all(x["completed"] for x in result["replays"])
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
