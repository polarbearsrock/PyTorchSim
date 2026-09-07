"""Drive an unmodified Transformers decoder layer; no local model forward.

Only inputs, validation, and simulator submission live here. The package owns
normalization, attention, RoPE, MLP, residuals, mask construction, and KV updates.
Mask construction is host-side setup, outside this layer-only timing boundary.
"""
import copy
import json

import torch
from transformers import Qwen2Config, Qwen2Model
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from ..submission import phase_kernel_ids
from ..validation import NumericalMismatch, compare_tensors
from .common import simulator_session


def build_decoder(config, dtype):
    # A metadata-only parent gives us the *actual* upstream initializer and mask
    # helper without allocating the other 27 layers or the embedding table.
    # No parent forward runs, and no meta tensors enter the measured layer.
    with torch.device("meta"):
        parent = Qwen2Model(config).eval()
    layer = Qwen2DecoderLayer(config, layer_idx=0)
    layer.apply(parent._init_weights)
    layer.eval().to(dtype=dtype)
    return parent, layer


def layer_inputs(parent, hidden, cache, start, device):
    positions = torch.arange(start, start + hidden.shape[1])
    mask = parent._update_causal_mask(None, hidden, positions, cache, False)
    return {
        "hidden_states": hidden.to(device),
        "attention_mask": mask.to(device),
        "position_ids": positions[None].to(device),
        "past_key_value": cache,
        "output_attentions": False,
        "use_cache": True,
        "cache_position": positions.to(device),
    }


def output_tensors(output):
    hidden, cache = output
    keys, values = cache[0]
    return hidden, keys, values


def run_decoder(args, manifest, _torch):
    config = Qwen2Config(**manifest["config"])
    config._attn_implementation = "eager"
    dtype = getattr(torch, args.dtype)
    parent, reference = build_decoder(config, dtype)
    if type(reference.self_attn).__name__ != "Qwen2Attention":
        raise RuntimeError("The decoder baseline requires the upstream eager Qwen2Attention class")
    device = "cpu" if args.mode == "cpu" else "npu:0"
    target = copy.deepcopy(reference).to(device=device)
    if args.mode != "cpu":
        target = torch.compile(target, fullgraph=True, dynamic=False)
    numerical = args.mode != "timing" or args.validate_timing
    exploratory = getattr(args, "allow_numerical_mismatch", False)
    hidden = torch.randn(args.batch, args.seq_len + args.decode_steps,
                         config.hidden_size, dtype=dtype)
    reference_cache, target_cache = DynamicCache(), DynamicCache()
    phases = []
    contract = {
        "scope": "one complete decoder layer; no embedding, final norm, or LM head",
        "model_class": f"{Qwen2DecoderLayer.__module__}.{Qwen2DecoderLayer.__name__}",
        "model_forward": "unmodified installed Transformers implementation",
        "attention_implementation": config._attn_implementation,
        "attention_class": f"{type(reference.self_attn).__module__}.{type(reference.self_attn).__name__}",
        "output_attentions": False,
        "cache_class": f"{DynamicCache.__module__}.{DynamicCache.__name__}",
        "mask_implementation": "Transformers Qwen2Model._update_causal_mask (host setup, not timed)",
        "weights": "synthetic, upstream Qwen2 initializer; no checkpoint loaded",
        "config": config.to_dict(),
        "measured_layers": 1,
        "measured_parameters": sum(p.numel() for p in reference.parameters()),
        "compilation": {"enabled": args.mode != "cpu", "fullgraph": True, "dynamic": False,
                        "note": "fullgraph rejects graph breaks; generated-code/device coverage still needs auditing"},
        "phase_file": "decoder_phases.json",
        "numerical_policy": "report finite tolerance mismatches" if exploratory else "strict tolerance gate",
        "measurement_class": "exploratory" if exploratory else "validation-gated",
    }
    # Persist scope even when compilation or simulation fails later.
    (args.output_dir / "workload.json").write_text(json.dumps(contract, indent=2) + "\n")
    with torch.no_grad(), simulator_session(args.mode) as session:
        for step in range(args.decode_steps + 1):
            query_length = args.seq_len if step == 0 else 1
            length = args.seq_len + step
            start = length - query_length
            phase = "prefill" if step == 0 else f"decode_{step}"
            inputs = hidden[:, start:length].contiguous()
            record = {"phase": phase, "query_length": query_length, "cache_length": length}
            if numerical:
                expected = reference(**layer_inputs(parent, inputs, reference_cache, start, "cpu"))
                reference_cache = expected[1]
                full = reference(**layer_inputs(parent, hidden[:, :length].contiguous(),
                                                 DynamicCache(), 0, "cpu"))
                full_hidden, full_k, full_v = output_tensors(full)
                record["cpu_cached_vs_full_prefix"] = compare_tensors(
                    output_tensors(expected), (full_hidden[:, -query_length:], full_k, full_v),
                    args.dtype, ("hidden", "keys", "values"))
            # Validation is host work. Snapshot before the upstream cache update
            # so an in-place mutation cannot also alter the expected old prefix.
            previous = (tuple(tensor.cpu().clone() for tensor in target_cache[0])
                        if numerical and len(target_cache) else None)
            arguments = layer_inputs(parent, inputs, target_cache, start, device)
            print(f"Starting decoder {phase}: queries={query_length}, cache={length}", flush=True)
            if args.mode == "timing":
                offset = len(session.trace_log)
                observed = torch.npu.launch_model(target, **arguments, stream_index=0, timestamp=0)
                record["kernel_ids"] = phase_kernel_ids(
                    session, offset, torch.npu.default_stream().launch_kernel)
            else:
                observed = target(**arguments)
            target_cache = observed[1]
            assert isinstance(target_cache, DynamicCache)
            assert len(target_cache) == 1 and target_cache.get_seq_length() == length
            observed_hidden, keys, values = output_tensors(observed)
            assert tuple(observed_hidden.shape) == (args.batch, query_length, config.hidden_size)
            assert tuple(keys.shape) == (args.batch, config.num_key_value_heads, length,
                                         config.hidden_size // config.num_attention_heads)
            assert tuple(values.shape) == tuple(keys.shape)
            if numerical:
                try:
                    record["max_abs_errors"] = compare_tensors(
                        output_tensors(observed), output_tensors(expected), args.dtype,
                        ("hidden", "keys", "values"),
                        diagnostic_dir=args.output_dir / "numerical" / phase)
                    record["numerical_status"] = "passed"
                except NumericalMismatch as error:
                    if not exploratory:
                        raise
                    record["numerical_status"] = "failed"
                    record["max_abs_errors"] = error.report["max_abs_errors"]
                    record["numerical_comparison"] = error.report
                    print(f"Exploratory warning: {phase} failed CPU tolerance checks; "
                          f"details: {error.report['diagnostic_dir']}", flush=True)
                if previous is not None:
                    compare_tensors((keys.cpu()[..., :start, :], values.cpu()[..., :start, :]), previous,
                                    args.dtype, ("unchanged_keys", "unchanged_values"), exact=True)
                    record["old_cache_prefix"] = "passed bit-exact host snapshot check"
            phases.append(record)
            (args.output_dir / contract["phase_file"]).write_text(json.dumps(phases, indent=2) + "\n")
            print(f"Completed decoder {phase}", flush=True)
        if args.mode == "timing":
            torch.npu.synchronize()
    mismatch = any(phase.get("numerical_status") == "failed" for phase in phases)
    return {**contract, "phases": phases,
            "numerical_status": "failed" if mismatch else "passed" if numerical else "not_performed",
            "numerical_validation": "FAILED CPU tolerance checks; exploratory execution only; structural and cache checks passed"
            if mismatch else "passed against upstream CPU, including full-prefix/cache checks"
            if numerical else "not performed; timing-only outputs are invalid"}
