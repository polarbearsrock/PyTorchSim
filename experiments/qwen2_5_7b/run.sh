#!/usr/bin/env bash
# Compatibility entry point. The Qwen study now lives under Simulator/experiments.
set -euo pipefail
QWEN_LEGACY_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
QWEN_CHECKOUT=$(cd -- "$QWEN_LEGACY_DIRECTORY/../.." && pwd)
exec bash "$QWEN_CHECKOUT/Simulator/experiments/qwen2_5_7b/run.sh" "$@"
