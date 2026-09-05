"""Pinned model configuration and simulator setup; safe to import without PyTorch."""
import json
import os
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "configs" / "model.json"


def load_manifest():
    return json.loads(MODEL_PATH.read_text())


def configure_simulator(args):
    # Run before importing torch: the backend reads these environment variables.
    import yaml

    repo = Path(os.environ["TORCHSIM_DIR"])
    config_path = repo / "configs/systolic_ws_128x128_c1_simple_noc_tpuv3_timing_only.yml"
    config = yaml.safe_load(config_path.read_text())
    validated_timing = getattr(args, "validate_timing", False)
    config["pytorchsim_functional_mode"] = int(args.mode == "functional" or validated_timing)
    config["pytorchsim_timing_mode"] = int(args.mode == "timing")
    # Functional validation does not need timing-based mapping selection.
    if args.mode == "functional" or validated_timing:
        config["codegen_mapping_strategy"] = "heuristic"
    if getattr(args, "mapping_file", None):
        mapping_path = args.mapping_file.resolve()
        mapping = json.loads(mapping_path.read_text())
        mapping_copy = args.output_dir / "mapping.json"
        mapping_copy.write_text(json.dumps(mapping, indent=2) + "\n")
        config["codegen_mapping_strategy"] = "external-then-heuristic"
        config["codegen_external_mapping_file"] = str(mapping_copy)
    config["ramulator_config_path"] = str(repo / "configs/ramulator2_configs/HBM2_TPUv3.yaml")
    output = args.output_dir / "simulator.yml"
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    os.environ["TOGSIM_CONFIG"] = str(output)
    return config
