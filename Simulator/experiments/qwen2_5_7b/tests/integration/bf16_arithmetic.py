"""Compile and numerically check elementary BF16 operations before model layers."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from Simulator.experiments.qwen2_5_7b.config import configure_simulator

parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=("all", "up", "down", "roundtrip", "mul", "add", "neg", "bitpatterns"), default="all")
args = parser.parse_args()
configure_simulator(SimpleNamespace(mode="functional", output_dir=Path(os.environ["TMPDIR"])))

import torch

torch.manual_seed(7)
torch.set_num_threads(4)
values = torch.randn(128, 8).to(torch.bfloat16)
values[0] = torch.tensor([1, -1, 2, -2, 0, 0.125, 1024, 1e10], dtype=torch.bfloat16)
down_values = torch.randn(128, 8)
# Rounding ties for both parities and signs, subnormals, infinity, NaN.
down_values[0] = torch.tensor([
    0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000,
    0x00008000, 0x00018000, 0x7F800000, 0x7FC00000,
], dtype=torch.int64).to(torch.int32).view(torch.float32)
all_words = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
cases = {
    "up": (lambda x: x.float(), values),
    "down": (lambda x: x.bfloat16(), down_values),
    "roundtrip": (lambda x: x.float().bfloat16(), values),
    "mul": (lambda x: x * x, values),
    "add": (lambda x: x + x, values),
    "neg": (lambda x: -x, values),
    "bitpatterns": (lambda x: x.float(), all_words.view(torch.bfloat16).reshape(128, 512)),
}
failures = []
with torch.no_grad():
    torch.testing.assert_close(values.to("npu:0").cpu(), values, rtol=0, atol=0)
    for name, (function, cpu_input) in cases.items():
        if args.case not in ("all", name):
            continue
        try:
            expected = function(cpu_input)
            compiled = torch.compile(function, fullgraph=True, dynamic=False)
            actual = compiled(cpu_input.to("npu:0")).cpu()
            print(name, "actual", actual.flatten()[:8].tolist(), "expected", expected.flatten()[:8].tolist(), flush=True)
            if name == "bitpatterns":
                # Verify payload bits as well as values, including every NaN.
                expected_bits = (all_words.to(torch.int32) << 16).reshape(128, 512)
                torch.testing.assert_close(actual.view(torch.int32), expected_bits, rtol=0, atol=0)
            else:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            print(name, "PASS", flush=True)
        except Exception as error:
            failures.append(name)
            print(name, "FAIL", str(error), flush=True)
print("Failed cases:", failures)
raise SystemExit(bool(failures))
