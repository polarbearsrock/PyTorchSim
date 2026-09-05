"""Numerical attention-operator checks, selected with --component attention_parts."""
import json

import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RotaryEmbedding, apply_rotary_pos_emb, repeat_kv, rotate_half,
)

from ...validation import compare_tensors
from ...workloads.common import causal_mask, simulator_session


def run_attention_components(args, manifest, _torch):
    config = Qwen2Config(**manifest["config"])
    dtype = getattr(torch, args.dtype)
    numerical = args.mode != "timing" or args.validate_timing
    batch, query_length, heads, kv_heads = args.batch, args.seq_len, config.num_attention_heads, config.num_key_value_heads
    dim = config.hidden_size // heads
    groups = heads // kv_heads
    key_length = query_length + 1
    # Match projection -> view -> transpose layouts used by real Qwen attention.
    q = torch.randn(batch, query_length, heads, dim, dtype=dtype).transpose(1, 2)
    k = torch.randn(batch, key_length, kv_heads, dim, dtype=dtype).transpose(1, 2)
    v = torch.randn_like(k)
    expanded_k, expanded_v = repeat_kv(k, groups), repeat_kv(v, groups)
    scores = torch.randn(batch, heads, query_length, key_length, dtype=dtype)
    mask = causal_mask(batch, query_length, key_length, dtype)
    probabilities = torch.softmax(scores.float() + mask.float(), -1).to(dtype)
    rotary = Qwen2RotaryEmbedding(dim, max_position_embeddings=key_length + 3, base=config.rope_theta).to(dtype)
    positions = torch.arange(3, 3 + query_length)[None].expand(batch, -1).contiguous()
    cases = {
        "lookup": (lambda c, s, p: (c[p], s[p]), (rotary.cos_cached, rotary.sin_cached, positions)),
        "rotate": (lambda x: rotate_half(x), (q,)),
        "rope_math": (lambda x, c, s: x * c + rotate_half(x) * s,
                      (q, rotary.cos_cached[positions].unsqueeze(1), rotary.sin_cached[positions].unsqueeze(1))),
        "rope": (lambda q, k, c, s, p: apply_rotary_pos_emb(q, k, c, s, p), (q, k[..., :query_length, :].contiguous(), rotary.cos_cached, rotary.sin_cached, positions)),
        "gqa": (lambda k, v: (repeat_kv(k, groups), repeat_kv(v, groups)), (k, v)),
        "scores": (lambda q, k: torch.matmul(q, k.transpose(-2, -1)), (q, expanded_k)),
        "softmax": (lambda x, mask: torch.softmax(x + mask, dim=-1, dtype=torch.float32).to(x.dtype), (scores, mask)),
        "context": (lambda p, v: torch.matmul(p, v), (probabilities, expanded_v)),
        "cache": (lambda k, v, nk, nv: (torch.cat((k, nk), -2), torch.cat((v, nv), -2)),
                  (k, v, k[..., :1, :].contiguous(), v[..., :1, :].contiguous())),
    }
    results, failures = {}, {}
    with torch.no_grad(), simulator_session(args.mode):
        for name, (function, inputs) in cases.items():
            if args.attention_case not in ("all", name):
                continue
            try:
                expected = function(*inputs)
                expected = expected if isinstance(expected, tuple) else (expected,)
                if args.mode == "cpu":
                    observed = function(*inputs)
                else:
                    compiled = torch.compile(function, fullgraph=True, dynamic=False)
                    device_inputs = tuple(value.to("npu:0") for value in inputs)
                    observed = torch.npu.launch_model(compiled, *device_inputs) if args.mode == "timing" else compiled(*device_inputs)
                observed = observed if isinstance(observed, tuple) else (observed,)
                results[name] = compare_tensors(observed, expected, args.dtype, tuple(f"output_{i}" for i in range(len(expected))), exact=name in ("lookup", "rotate", "gqa", "cache")) if numerical else {"timing_only": True}
                print(name, "PASS", results[name], flush=True)
            except Exception as error:
                import traceback
                traceback.print_exc()
                failures[name] = f"{type(error).__name__}: {error}"
                print(name, "FAIL", flush=True)
            (args.output_dir / "attention_parts.json").write_text(json.dumps({"passed": results, "failed": failures}, indent=2) + "\n")
        if args.mode == "timing":
            torch.npu.synchronize()
    if failures:
        raise RuntimeError("Attention parts failed: " + ", ".join(failures))
    return {"parts": results, "numerical_validation": "passed against CPU" if numerical else "not performed"}
