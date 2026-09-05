"""Small compiled matrix tests, including accumulation across three K tiles."""
import argparse
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace

from Simulator.experiments.qwen2_5_7b.config import configure_simulator

parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=("all", "gemm", "bias", "decode", "cancellation"), default="all")
parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
args = parser.parse_args()
scratch = Path(os.environ["TMPDIR"])
configure_simulator(SimpleNamespace(mode="functional", output_dir=scratch))

import torch
from PyTorchSimFrontend import extension_config

torch.manual_seed(19)
torch.set_num_threads(4)
dtype = getattr(torch, args.dtype)
mapping = scratch / "matrix_mapping.json"
# Three distinct K tiles: retaining FP32 partial sums is essential here.
mapping.write_text(json.dumps({
    "16_128_384": {"TILE_M": 16, "TILE_N": 128, "TILE_K": 128},
    "1_128_384": {"TILE_M": 1, "TILE_N": 128, "TILE_K": 128},
}))
extension_config.codegen_mapping_strategy = "external-then-heuristic"
extension_config.codegen_external_mapping_file = str(mapping)

failures = []
with torch.no_grad():
    for name in ("gemm", "bias", "decode", "cancellation"):
        if args.case not in ("all", name):
            continue
        m = 1 if name == "decode" else 16
        a = torch.randn(m, 384, dtype=dtype)
        b = torch.randn(384, 128, dtype=dtype)
        bias = torch.randn(128, dtype=dtype)
        if name == "cancellation":
            a.fill_(1)
            b.zero_()
            b[0] = 256
            b[128] = 1
            b[256] = -256
        function = (lambda x, w, z: torch.nn.functional.linear(x, w.t(), z)) if name == "bias" else (lambda x, w, z: x @ w)
        expected = a.float() @ b.float()
        if name == "bias":
            expected += bias.float()
        expected = expected.to(dtype)
        if dtype == torch.float16:
            # Regression control for the existing FP16 path, which rounds each
            # matrix pop AND running partial sum to FP16. This is intentionally
            # different from the new BF16-input / FP32-accumulator contract.
            expected = bias.expand(m, 128).clone() if name == "bias" else torch.zeros(m, 128, dtype=dtype)
            for start in range(0, 384, 128):
                partial = (a[:, start:start + 128].float() @ b[start:start + 128].float()).half()
                expected = expected + partial
        try:
            actual = torch.compile(function, fullgraph=True, dynamic=False)(a.to("npu:0"), b.to("npu:0"), bias.to("npu:0")).cpu()
            tolerance = 0 if name == "cancellation" else (0.016 if dtype == torch.bfloat16 else 0.002 if dtype == torch.float16 else 1e-4)
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            print(name, "PASS", "max_abs_error", (actual.float() - expected.float()).abs().max().item(), flush=True)
        except Exception as error:
            failures.append(name)
            traceback.print_exc()
            print(name, "FAIL", str(error), flush=True)
print("Failed cases:", failures, flush=True)
raise SystemExit(bool(failures))
