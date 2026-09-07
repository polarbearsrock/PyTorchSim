"""Small CPU protocol tests for direct upstream calls, not performance results."""
import copy
import unittest

import torch
from transformers import Qwen2Config, Qwen2Model
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2DecoderLayer

from ...provenance import transformers_provenance
from ...workloads.decoder import build_decoder, layer_inputs, output_tensors


class TransformersDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = Qwen2Config(hidden_size=56, intermediate_size=64,
                                 num_attention_heads=7, num_key_value_heads=1,
                                 num_hidden_layers=1, vocab_size=32,
                                 max_position_embeddings=64, sliding_window=None)
        self.config._attn_implementation = "eager"

    def test_installed_package_and_unmodified_class(self):
        self.assertEqual(transformers_provenance()["version"], "4.43.4")
        _, layer = build_decoder(self.config, torch.bfloat16)
        self.assertIs(type(layer), Qwen2DecoderLayer)
        self.assertIs(type(layer.self_attn), Qwen2Attention)
        self.assertIs(layer.forward.__func__, Qwen2DecoderLayer.forward)
        self.assertTrue(all(p.device.type == "cpu" and p.dtype == torch.bfloat16 for p in layer.parameters()))

    def test_matches_layer_called_by_upstream_model(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype), torch.no_grad():
                parent, layer = build_decoder(self.config, dtype)
                model = Qwen2Model(self.config).eval().to(dtype=dtype)
                model.layers[0].load_state_dict(layer.state_dict())
                captures = []
                handle = model.layers[0].register_forward_hook(lambda _module, _args, output: captures.append(output[0]))
                hidden = torch.randn(2, 5, self.config.hidden_size, dtype=dtype)
                model_cache, layer_cache = DynamicCache(), DynamicCache()
                try:
                    for start, end in ((0, 3), (3, 4), (4, 5)):
                        inputs = hidden[:, start:end].contiguous()
                        expected = model(inputs_embeds=inputs, past_key_values=model_cache,
                                         use_cache=True, output_attentions=False)
                        actual = layer(**layer_inputs(parent, inputs, layer_cache, start, "cpu"))
                        torch.testing.assert_close(actual[0], captures[-1], rtol=0, atol=0)
                        for observed, reference in zip(actual[1][0], expected.past_key_values[0]):
                            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
                        self.assertEqual(actual[1].get_seq_length(), end)
                finally:
                    handle.remove()

    def test_fullgraph_capture_preserves_cache_mutation(self):
        with torch.no_grad():
            parent, layer = build_decoder(self.config, torch.bfloat16)
            compiled = torch.compile(copy.deepcopy(layer), backend="eager", fullgraph=True, dynamic=False)
            reference_cache, compiled_cache = DynamicCache(), DynamicCache()
            for start, count in ((0, 3), (3, 1), (4, 1)):
                hidden = torch.randn(1, count, self.config.hidden_size, dtype=torch.bfloat16)
                expected = layer(**layer_inputs(parent, hidden, reference_cache, start, "cpu"))
                actual = compiled(**layer_inputs(parent, hidden, compiled_cache, start, "cpu"))
                self.assertIs(actual[1], compiled_cache)
                self.assertEqual(compiled_cache.get_seq_length(), start + count)
                for observed, reference in zip(output_tensors(actual), output_tensors(expected)):
                    torch.testing.assert_close(observed, reference, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
