"""Analytical parameter and BF16 memory inventory, not a verified device fit."""

def parameter_breakdown(config):
    h = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    if h % heads or heads % kv_heads:
        raise ValueError("Head dimensions and GQA grouping must divide exactly")
    kv_width = kv_heads * (h // heads)
    # Qwen2: q/k/v biases, no o_proj or MLP biases, two RMSNorms/layer.
    attention = 2 * h * h + 2 * h * kv_width + h + 2 * kv_width
    mlp = 3 * h * config["intermediate_size"]
    norms = 2 * h
    embedding = config["vocab_size"] * h
    result = {
        "attention_per_layer": attention,
        "mlp_per_layer": mlp,
        "norms_per_layer": norms,
        "embedding": embedding,
        "lm_head": 0 if config["tie_word_embeddings"] else embedding,
        "final_norm": h,
    }
    result["total"] = (attention + mlp + norms) * config["num_hidden_layers"]
    result["total"] += result["embedding"] + result["lm_head"] + h
    return result


def memory_audit(manifest, batch, context_tokens):
    if batch < 1 or context_tokens < 1:
        raise ValueError("Batch and context length must be positive")
    config = manifest["config"]
    counts = parameter_breakdown(config)
    if counts["total"] != manifest["checkpoint_parameters"]:
        raise ValueError("Analytical parameter count disagrees with checkpoint")
    weights = counts["total"] * 2
    if weights != manifest["checkpoint_bytes"]:
        raise ValueError("BF16 weight bytes disagree with checkpoint")
    kv_per_token = (2 * config["num_hidden_layers"]
                    * config["num_key_value_heads"]
                    * (config["hidden_size"] // config["num_attention_heads"]) * 2)
    kv_bytes = batch * context_tokens * kv_per_token
    return {
        "parameters": counts,
        "batch": batch,
        "context_tokens": context_tokens,
        "weight_bytes_bf16": weights,
        "kv_bytes_per_token_per_sequence_bf16": kv_per_token,
        "kv_bytes_bf16": kv_bytes,
        "weights_plus_kv_bytes": weights + kv_bytes,
        "core_capacity_budget_bytes": 16 * 2**30,
        "bytes_left_before_workspace": 16 * 2**30 - weights - kv_bytes,
        "fit_verified": False,
        "exclusions": ["activations", "workspace", "padding", "runtime allocations"],
    }
