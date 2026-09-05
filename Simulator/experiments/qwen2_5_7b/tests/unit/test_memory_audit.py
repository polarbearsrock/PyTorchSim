"""Standard-library tests for the model and memory inventory."""

import unittest

from Simulator.experiments.qwen2_5_7b.config import load_manifest
from Simulator.experiments.qwen2_5_7b.analysis.memory_audit import memory_audit, parameter_breakdown


class AuditTests(unittest.TestCase):
    def test_parameter_inventory(self):
        manifest = load_manifest()
        self.assertEqual(parameter_breakdown(manifest["config"])["total"], 7615616512)

    def test_kv_storage(self):
        result = memory_audit(load_manifest(), 1, 2048)
        self.assertEqual(result["kv_bytes_per_token_per_sequence_bf16"], 56 * 1024)
        self.assertEqual(result["kv_bytes_bf16"], 112 * 1024**2)
        self.assertFalse(result["fit_verified"])
        self.assertGreater(result["bytes_left_before_workspace"], 0)

    def test_batch_scaling_and_capacity(self):
        one = memory_audit(load_manifest(), 1, 2048)
        two = memory_audit(load_manifest(), 2, 2048)
        self.assertEqual(two["kv_bytes_bf16"], 2 * one["kv_bytes_bf16"])
        self.assertEqual(two["weight_bytes_bf16"], one["weight_bytes_bf16"])
        large = memory_audit(load_manifest(), 1, 65536)
        self.assertLess(large["bytes_left_before_workspace"], 0)

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            memory_audit(load_manifest(), 0, 2048)
        config = dict(load_manifest()["config"], num_attention_heads=27)
        with self.assertRaises(ValueError):
            parameter_breakdown(config)


if __name__ == "__main__":
    unittest.main()
