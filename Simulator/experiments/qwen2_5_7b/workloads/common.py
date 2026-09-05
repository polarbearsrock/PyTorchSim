"""Causal masks and the continuous simulator-session boundary."""
import contextlib
import os

import torch


def causal_mask(batch, query_length, key_length, dtype):
    # Decode queries occupy the final query_length positions, not the beginning.
    queries = torch.arange(key_length - query_length, key_length)
    keys = torch.arange(key_length)
    blocked = keys.unsqueeze(0) > queries.unsqueeze(1)
    mask = torch.zeros(query_length, key_length, dtype=dtype)
    mask.masked_fill_(blocked, torch.finfo(dtype).min)
    return mask[None, None].expand(batch, 1, query_length, key_length).contiguous()


def simulator_session(mode):
    if mode != "timing":
        return contextlib.nullcontext()
    from Simulator.simulator import TOGSimulator
    return TOGSimulator(config_path=os.environ["TOGSIM_CONFIG"])
