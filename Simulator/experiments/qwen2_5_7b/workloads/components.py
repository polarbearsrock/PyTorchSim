"""Isolated upstream RMSNorm, query projection, and SwiGLU workloads."""
import copy

from ..validation import numerical_tolerances


def run_components(args, manifest, torch):
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2MLP

    config = Qwen2Config(**manifest["config"])
    dtype = getattr(torch, args.dtype)
    if args.component == "rmsnorm":
        module = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    elif args.component == "q_proj":
        module = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=True)
    else:
        module = Qwen2MLP(config)
    module.eval().to(dtype=dtype)
    values = torch.randn(args.batch, args.seq_len, config.hidden_size, dtype=dtype)
    with torch.no_grad():
        expected = module(values) if args.mode == "functional" else None
        device_module = copy.deepcopy(module).to("npu:0")
        device_values = values.to("npu:0")
        compiled = torch.compile(device_module, fullgraph=True, dynamic=False)
        if args.mode == "timing":
            from Simulator.simulator import TOGSimulator

            with TOGSimulator(config_path=os.environ["TOGSIM_CONFIG"]):
                torch.npu.launch_model(compiled, device_values, stream_index=0, timestamp=0)
                torch.npu.synchronize()
            return {"numerical_validation": "not performed; timing-only outputs are invalid"}
        actual = compiled(device_values).cpu()
        max_error = (actual.float() - expected.float()).abs().max().item()
        torch.testing.assert_close(actual, expected, **numerical_tolerances(args.dtype))
    return {"numerical_validation": "passed against CPU", "max_abs_error": max_error}
