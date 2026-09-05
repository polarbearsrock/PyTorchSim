#!/usr/bin/env bash
set -euo pipefail
QWEN_BUILD_ROOT=/workspace/bf16-toolchain
QWEN_SPIKE_REV=fb28e7aa5ea26e637bf104541d65a04f8fd0847e
QWEN_BUILD_SCRIPTS=/workspace/PyTorchSim/Simulator/experiments/qwen2_5_7b/tools/bf16
case "${1:-all}" in
  all)
    mkdir -p "$QWEN_BUILD_ROOT/spike-legacy-build"
    cd "$QWEN_BUILD_ROOT/spike-legacy-build"
    "$QWEN_BUILD_ROOT/riscv-isa-sim-$QWEN_SPIKE_REV/configure" --prefix="$QWEN_BUILD_ROOT/install"
    make -j8
    make install
    ;;
  --plugins-only) test -x "$QWEN_BUILD_ROOT/install/bin/spike" ;;
  *) printf 'Usage: build_in_container.sh [--plugins-only]\n' >&2; exit 2 ;;
esac
g++ -std=c++17 -O2 -fno-rtti -fPIC -shared -I/riscv-llvm/include \
  "$QWEN_BUILD_SCRIPTS/BF16ToVCIX.cpp" -o "$QWEN_BUILD_ROOT/libPyTorchSimBF16.so"
g++ -std=c++17 -O2 -fno-rtti -fPIC -shared -I/riscv-llvm/include \
  "$QWEN_BUILD_SCRIPTS/BF16Memory.cpp" -o "$QWEN_BUILD_ROOT/libPyTorchSimBF16Memory.so"
/riscv-llvm/bin/mlir-opt --load-pass-plugin="$QWEN_BUILD_ROOT/libPyTorchSimBF16.so" \
  --help | rg -- '--pytorchsim-bf16-to-vcix'
sha256sum "$QWEN_BUILD_ROOT/install/bin/spike" \
  "$QWEN_BUILD_ROOT/libPyTorchSimBF16.so" \
  "$QWEN_BUILD_ROOT/libPyTorchSimBF16Memory.so" \
  "$QWEN_BUILD_SCRIPTS/BF16ToVCIX.cpp" \
  "$QWEN_BUILD_SCRIPTS/BF16Memory.cpp" \
  "$QWEN_BUILD_SCRIPTS/spike-bf16.patch" > "$QWEN_BUILD_ROOT/build-sha256.txt"
