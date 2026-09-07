"""Execute an unchanged gem5 configuration with an explicit diagnostic tick cap.

This is a gem5 script, not a host Python program. A limit exit is a failure,
never a usable latency sample. Binary and result paths are set by the harness.
"""
import json
import os
from pathlib import Path
import runpy
import sys

import m5

real_simulate = m5.simulate


def bounded_simulate(*_args, **_kwargs):
    ticks = int(os.environ["QWEN_GEM5_PROBE_TICKS"])
    if ticks <= 0:
        raise ValueError("Diagnostic tick limit must be positive")
    event = real_simulate(ticks)
    result = {"final_tick": int(m5.curTick()), "cause": event.getCause()}
    Path(os.environ["QWEN_GEM5_PROBE_RESULT"]).write_text(json.dumps(result, indent=2) + "\n")
    print("GEM5_PROBE", json.dumps(result), flush=True)
    m5.stats.dump()
    return event


m5.simulate = bounded_simulate
script = str(Path(os.environ["TORCHSIM_DIR"]) / "gem5_script/script_systolic.py")
sys.argv = [script, "-c", os.environ["QWEN_GEM5_PROBE_BINARY"], "--vlane", "128", "--vlen", "256"]
runpy.run_path(script, run_name="__main__")
