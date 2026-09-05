"""Reduced two-layer CPU reference; never a full-width performance proxy."""
import copy

from ..validation import numerical_tolerances


def cpu_reference(args, manifest, torch):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from transformers.cache_utils import DynamicCache

    full_config = Qwen2Config(**manifest["config"])
    full_config._attn_implementation = "eager"
    with torch.device("meta"):
        full_model = Qwen2ForCausalLM(full_config)
    actual_count = sum(p.numel() for p in full_model.parameters())
    assert actual_count == manifest["checkpoint_parameters"], actual_count
    del full_model

    # Preserve head_dim=128 and the 7:1 GQA ratio, but reduce size for correctness.
    tiny = copy.deepcopy(full_config)
    tiny.hidden_size = 896
    tiny.intermediate_size = 1024
    tiny.num_attention_heads = 7
    tiny.num_key_value_heads = 1
    tiny.num_hidden_layers = 2
    tiny.vocab_size = 256
    tiny.max_position_embeddings = max(128, args.seq_len + args.decode_steps)
    tiny.bos_token_id = 1
    tiny.eos_token_id = 2
    dtype = getattr(torch, args.dtype)
    model = Qwen2ForCausalLM(tiny).eval().to(dtype=dtype)
    tokens = torch.randint(3, tiny.vocab_size, (args.batch, args.seq_len + args.decode_steps))
    cache = DynamicCache()
    with torch.no_grad():
        prefill = model(tokens[:, :args.seq_len], past_key_values=cache, use_cache=True)
        assert cache.get_seq_length() == args.seq_len
        assert torch.isfinite(prefill.logits).all()
        errors = []
        for step in range(args.decode_steps):
            index = args.seq_len + step
            cached = model(tokens[:, index:index + 1], past_key_values=cache, use_cache=True)
            full = model(tokens[:, :index + 1], use_cache=False)
            actual, expected = cached.logits[:, -1], full.logits[:, -1]
            torch.testing.assert_close(actual, expected, **numerical_tolerances(args.dtype))
            assert cache.get_seq_length() == index + 1
            errors.append((actual - expected).abs().max().item())
    return {
        "full_model_meta_parameter_count": actual_count,
        "reference_shape": "reduced two-layer model; not a 7B timing measurement",
        "reference_config": {key: getattr(tiny, key) for key in (
            "hidden_size", "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "num_hidden_layers", "vocab_size")},
        "decode_max_abs_errors": errors,
        "final_cache_length": cache.get_seq_length(),
        "numerical_validation": "passed on CPU only",
    }
