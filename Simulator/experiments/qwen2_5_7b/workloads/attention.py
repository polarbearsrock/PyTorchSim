"""Full-width upstream Qwen2Attention with explicit input/output KV tensors.

This is an attention workload, not a complete decoder layer or a replacement
for Transformers attention. Weights are synthetic; math stays upstream.
"""
import copy
import json

import torch
from transformers import Qwen2Config
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

from ..validation import compare_tensors
from ..submission import phase_kernel_ids
from .common import causal_mask, simulator_session


class CachedAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Qwen2Attention(config, layer_idx=0)

    def forward(self, hidden, mask, positions, previous_k, previous_v):
        cache = DynamicCache()
        if previous_k.shape[-2]:
            cache.key_cache.append(previous_k)
            cache.value_cache.append(previous_v)
            cache._seen_tokens = previous_k.shape[-2]
        output, probabilities, cache = self.attention(
            hidden, attention_mask=mask, position_ids=positions,
            past_key_value=cache, output_attentions=True, use_cache=True,
        )
        return output, probabilities, cache.key_cache[0], cache.value_cache[0]


def run_attention(args, manifest, _torch):
    config = Qwen2Config(**manifest["config"])
    config._attn_implementation = "eager"
    dtype = getattr(torch, args.dtype)
    numerical = args.mode != "timing" or args.validate_timing
    cpu_module = CachedAttention(config).eval().to(dtype=dtype)
    device = "cpu" if args.mode == "cpu" else "npu:0"
    target = copy.deepcopy(cpu_module).to(device=device)
    if args.mode != "cpu":
        target = torch.compile(target, fullgraph=True, dynamic=False)
    total_length = args.seq_len + args.decode_steps
    hidden = torch.randn(args.batch, total_length, config.hidden_size, dtype=dtype)
    empty = torch.empty(args.batch, config.num_key_value_heads, 0,
                        config.hidden_size // config.num_attention_heads, dtype=dtype)
    cpu_k, cpu_v = empty, empty.clone()
    device_k, device_v = empty.to(device), empty.clone().to(device)
    phases = []
    with torch.no_grad(), simulator_session(args.mode) as session:
        for step in range(args.decode_steps + 1):
            query_length = args.seq_len if step == 0 else 1
            length = args.seq_len + step
            phase = "prefill" if step == 0 else f"decode_{step}"
            inputs = hidden[:, length - query_length:length].contiguous()
            mask = causal_mask(args.batch, query_length, length, dtype)
            positions = torch.arange(length - query_length, length)[None].expand(args.batch, -1).contiguous()
            record = {"phase": phase, "query_length": query_length, "cache_length": length}
            if numerical:
                expected = cpu_module(inputs, mask, positions, cpu_k, cpu_v)
                # Independently recompute the whole prefix: catches mask, offset,
                # head replication, and cache-append mistakes in the wrapper.
                full = cpu_module(hidden[:, :length].contiguous(),
                                  causal_mask(args.batch, length, length, dtype),
                                  torch.arange(length)[None].expand(args.batch, -1), empty, empty)
                record["cpu_cached_vs_full_prefix"] = compare_tensors((expected[0], expected[2], expected[3]),
                                (full[0][:, -query_length:], full[2], full[3]), args.dtype,
                                ("cached_output_vs_full_prefix", "keys_vs_full_prefix", "values_vs_full_prefix"))
                cpu_k, cpu_v = expected[2:]
            device_args = (inputs.to(device), mask.to(device), positions.to(device), device_k, device_v)
            print(f"Starting {phase}: queries={query_length}, cache={length}", flush=True)
            if args.mode == "timing":
                command_offset = len(session.trace_log)
                observed = torch.npu.launch_model(target, *device_args, stream_index=0, timestamp=0)
                record["kernel_ids"] = phase_kernel_ids(
                    session, command_offset, torch.npu.default_stream().launch_kernel)
            else:
                observed = target(*device_args)
            if numerical:
                record["max_abs_errors"] = compare_tensors(observed, expected, args.dtype,
                                                           ("output", "probabilities", "keys", "values"))
                probabilities = observed[1].cpu()
                torch.testing.assert_close(probabilities.float().sum(-1), torch.ones_like(probabilities[..., 0]).float(), rtol=0.01, atol=0.01)
                blocked = mask.expand_as(probabilities) != 0
                assert torch.count_nonzero(probabilities.masked_select(blocked)).item() == 0
                if device_k.shape[-2]:
                    torch.testing.assert_close(observed[2].cpu()[..., :-1, :], device_k.cpu(), rtol=0, atol=0)
                    torch.testing.assert_close(observed[3].cpu()[..., :-1, :], device_v.cpu(), rtol=0, atol=0)
            device_k, device_v = observed[2:]
            assert tuple(device_k.shape) == (args.batch, config.num_key_value_heads, length, config.hidden_size // config.num_attention_heads)
            assert tuple(device_v.shape) == tuple(device_k.shape)
            phases.append(record)
            (args.output_dir / "attention_phases.json").write_text(json.dumps(phases, indent=2) + "\n")
            print(f"Completed {phase}", flush=True)
        if args.mode == "timing":
            torch.npu.synchronize()
    return {"numerical_validation": "passed, including cache and full-prefix checks" if numerical else "not performed; timing-only outputs are invalid",
            "attention_implementation": "Transformers Qwen2Attention (eager)",
            "num_query_heads": config.num_attention_heads, "num_kv_heads": config.num_key_value_heads,
            "phases": phases}
