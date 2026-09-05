#!/usr/bin/env bash
# Host entry point; downloads and all build/install products stay in TMPDIR.
set -euo pipefail
: "${TMPDIR:?Set TMPDIR to a scratch directory with sufficient space}"
QWEN_TOOLCHAIN_SCRIPT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
QWEN_SPIKE_REV=fb28e7aa5ea26e637bf104541d65a04f8fd0847e
QWEN_BUILD_ROOT=$(mktemp -d "$TMPDIR/qwen-bf16-toolchain.XXXXXX")
printf 'Toolchain directory: %s\n' "$QWEN_BUILD_ROOT"
curl -fsSL "https://codeload.github.com/PSAL-POSTECH/riscv-isa-sim/tar.gz/$QWEN_SPIKE_REV" \
  -o "$QWEN_BUILD_ROOT/spike.tar.gz"
tar -xzf "$QWEN_BUILD_ROOT/spike.tar.gz" -C "$QWEN_BUILD_ROOT"
patch --batch --forward --directory "$QWEN_BUILD_ROOT/riscv-isa-sim-$QWEN_SPIKE_REV" \
  -p1 --input "$QWEN_TOOLCHAIN_SCRIPT/spike-bf16.patch"
QWEN_TOOLCHAIN_ROOT="$QWEN_BUILD_ROOT" QWEN_TIMEOUT_SECONDS=1200 \
  bash "$QWEN_TOOLCHAIN_SCRIPT/../../run.sh" toolchain \
  bash Simulator/experiments/qwen2_5_7b/tools/bf16/build_in_container.sh
printf '\nUse this toolchain for new, isolated runs:\nexport QWEN_TOOLCHAIN_ROOT=%q\n' "$QWEN_BUILD_ROOT"
printf 'export TORCHSIM_SPIKE=/workspace/bf16-toolchain/install/bin/spike\n'
printf 'export TORCHSIM_BF16_PLUGIN=/workspace/bf16-toolchain/libPyTorchSimBF16.so\n'
