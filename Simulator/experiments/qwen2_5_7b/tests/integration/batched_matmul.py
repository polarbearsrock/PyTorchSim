"""Numerical BMM regressions with a test-only forced three-K-tile mapping."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from Simulator.experiments.qwen2_5_7b.config import configure_simulator

parser = argparse.ArgumentParser()
parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
parser.add_argument("--case", choices=("all", "bmm", "transposed", "prologue", "decode", "cancellation", "narrow8", "narrow9", "shortk8"), default="all")
args = parser.parse_args()
configure_simulator(SimpleNamespace(mode="functional", output_dir=Path(os.environ["TMPDIR"])))

import torch
from PyTorchSimFrontend.mlir.mlir_bmm_template import MLIRBMMTemplate

torch.manual_seed(29)
torch.set_num_threads(4)
dtype = getattr(torch, args.dtype)


def forced_tile(_template, _kernel, m, n, k, *_args, **_kwargs):
    # BMM does not currently consult the external GEMM mapping file. Patch only
    # this test process, so all cases really exercise three separate K tiles.
    assert n in (8, 9, 128) and k in (8, 128, 384)
    tile_k = min(k, 128)
    return [(8, n, tile_k, 8, n, tile_k)]


with torch.no_grad(), patch.object(MLIRBMMTemplate, "select_tile", forced_tile):
    cases = ("bmm", "transposed", "prologue", "decode", "cancellation")
    # The old dtype controls retain their original five-case scope. Additional
    # small-tile cases validate the BF16-specific plugin, not the legacy plugin.
    if dtype == torch.bfloat16 or args.case != "all":
        cases += ("narrow8", "narrow9", "shortk8")
    for case in cases:
        if args.case not in ("all", case):
            continue
        m = 1 if case == "decode" else 8
        k = 128 if case.startswith("narrow") else 384
        if case == "shortk8":
            k = 8
        n = int(case.removeprefix("narrow")) if case.startswith("narrow") else 128
        a = torch.randn(3, m, k, dtype=dtype)
        b = torch.randn(3, k, n, dtype=dtype)
        if case == "transposed":
            b = b.transpose(1, 2).contiguous().transpose(1, 2)
        if case == "cancellation":
            a.fill_(1)
            b.zero_()
            b[:, 0] = 256
            b[:, 128] = 1
            b[:, 256] = -256
        function = (lambda x, w: torch.bmm(x * 0.5, w)) if case == "prologue" else torch.bmm
        effective_a = a * 0.5 if case == "prologue" else a
        expected = torch.bmm(effective_a.float(), b.float()).to(dtype)
        if dtype == torch.float16:
            expected = torch.zeros(3, m, n, dtype=dtype)
            for start in range(0, k, 128):
                expected += torch.bmm(effective_a[:, :, start:start + 128].float(), b[:, start:start + 128].float()).half()
        actual = torch.compile(function, fullgraph=True, dynamic=False)(a.to("npu:0"), b.to("npu:0")).cpu()
        tolerance = 0 if case == "cancellation" else (0.016 if dtype == torch.bfloat16 else 0.002 if dtype == torch.float16 else 1e-4)
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        print(case, "PASS", "max_abs_error", (actual.float() - expected.float()).abs().max().item(), flush=True)
