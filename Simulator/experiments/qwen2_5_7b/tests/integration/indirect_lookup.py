"""Exact dynamic table-lookup checks: offsets, repeated rows, and decode."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from Simulator.experiments.qwen2_5_7b.config import configure_simulator

parser = argparse.ArgumentParser()
parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
args = parser.parse_args()
configure_simulator(SimpleNamespace(mode="functional", output_dir=Path(os.environ["TMPDIR"])))

import torch

torch.manual_seed(31)
torch.set_num_threads(4)
table = torch.randn(32, 128, dtype=getattr(torch, args.dtype))
compiled = torch.compile(lambda values, indices: values[indices], fullgraph=True, dynamic=False)
with torch.no_grad():
    for case, positions in (
        ("offset", torch.arange(3, 11)[None]),
        ("nonmonotonic_repeated", torch.tensor([[31, 7, 7, 0, 2, 19, 1, 8], [3, 4, 0, 3, 9, 20, 31, 30]])),
        ("decode", torch.tensor([[27]])),
    ):
        actual = compiled(table.to("npu:0"), positions.to("npu:0")).cpu()
        torch.testing.assert_close(actual, table[positions], rtol=0, atol=0)
        print(case, "PASS", flush=True)
