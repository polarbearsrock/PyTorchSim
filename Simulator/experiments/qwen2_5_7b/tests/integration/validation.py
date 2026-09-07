"""CPU-only tests of exploratory reporting; no simulator or compiler mutation."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2Config

from ...validation import NumericalMismatch, compare_tensors, numerical_tolerances
from ...workloads import decoder


class NumericalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(self.scratch.cleanup)
        self.output = Path(self.scratch.name)
        self.reference = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

    def test_default_tolerances_and_pass_unchanged(self):
        self.assertEqual(numerical_tolerances("bfloat16"), {"rtol": .02, "atol": .02})
        self.assertEqual(compare_tensors((self.reference,), (self.reference,), "bfloat16", ("hidden",)), {"hidden": 0.0})

    def test_both_comparison_operands_are_moved_to_cpu(self):
        # A device operand exposes only a CPU transfer here: arithmetic or
        # isfinite on the original object must not be attempted by validation.
        device_operand = SimpleNamespace(cpu=lambda: self.reference)
        self.assertEqual(compare_tensors((device_operand,), (device_operand,),
                                         "bfloat16", ("cache",), exact=True), {"cache": 0.0})

    def test_mismatch_is_still_an_assertion_and_preserves_phase_artifacts(self):
        for phase in ("prefill", "decode_1"):
            with self.assertRaises(NumericalMismatch) as failure:
                compare_tensors((self.reference + 1,), (self.reference,), "bfloat16", ("hidden",),
                                diagnostic_dir=self.output / phase)
            self.assertIsInstance(failure.exception, AssertionError)
            self.assertEqual(failure.exception.report["statistics"]["hidden"]["mismatched_elements"], 2)
        for phase in ("prefill", "decode_1"):
            saved = torch.load(self.output / phase / "comparison_failure.pt", weights_only=True)
            torch.testing.assert_close(saved["hidden"]["expected"], self.reference, rtol=0, atol=0)

    def test_structural_and_nonfinite_failures_cannot_be_waived(self):
        cases = (torch.ones(3, dtype=torch.bfloat16), self.reference.float(),
                 torch.tensor([float("nan"), 2.0], dtype=torch.bfloat16),
                 torch.tensor([float("inf"), 2.0], dtype=torch.bfloat16))
        for observed in cases:
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                compare_tensors((observed,), (self.reference,), "bfloat16", ("hidden",))

    def run_small_decoder(self, allow, broken_prefix=False):
        config = Qwen2Config(hidden_size=56, intermediate_size=64, num_attention_heads=7,
                             num_key_value_heads=1, num_hidden_layers=1, vocab_size=32,
                             max_position_embeddings=64, sliding_window=None)
        args = SimpleNamespace(mode="cpu", dtype="bfloat16", batch=1, seq_len=3,
                               decode_steps=2, validate_timing=False, output_dir=self.output,
                               allow_numerical_mismatch=allow)

        def inject_mismatch(actual, expected, dtype, names, **kwargs):
            # Change only the tensors entering the checker, never the model or cache.
            if kwargs.get("diagnostic_dir") is not None or (broken_prefix and kwargs.get("exact")):
                actual = (actual[0] + 1, *actual[1:])
            return compare_tensors(actual, expected, dtype, names, **kwargs)

        with patch.object(decoder, "compare_tensors", side_effect=inject_mismatch), torch.no_grad():
            return decoder.run_decoder(args, {"config": config.to_dict()}, torch)

    def test_exploratory_continues_but_records_failed_validation(self):
        result = self.run_small_decoder(True)
        self.assertEqual(result["numerical_status"], "failed")
        self.assertEqual([p["phase"] for p in result["phases"]], ["prefill", "decode_1", "decode_2"])
        self.assertTrue(all(p["numerical_status"] == "failed" for p in result["phases"]))
        self.assertTrue(all("old_cache_prefix" in p for p in result["phases"][1:]))
        self.assertEqual(len(json.loads((self.output / "decoder_phases.json").read_text())), 3)

    def test_default_still_stops_and_cache_checks_remain_strict(self):
        with self.assertRaises(NumericalMismatch):
            self.run_small_decoder(False)
        with self.assertRaises(NumericalMismatch):
            self.run_small_decoder(True, broken_prefix=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
