#!/usr/bin/env bash
set -euo pipefail
LLVM_BUILD_ROOT=/workspace/llvm-build
cmake -S /workspace/llvm-source/llvm -B "$LLVM_BUILD_ROOT/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$LLVM_BUILD_ROOT/install" \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=RISCV \
  -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_ENABLE_RTTI=OFF \
  -DBUILD_SHARED_LIBS=OFF -DLLVM_USE_LINKER=lld \
  -DLLVM_INCLUDE_TESTS=ON -DMLIR_INCLUDE_TESTS=ON \
  -DLLVM_PARALLEL_LINK_JOBS=2 -DPython3_EXECUTABLE=/usr/bin/python3
cmake --build "$LLVM_BUILD_ROOT/build" --parallel "$1"
cmake --build "$LLVM_BUILD_ROOT/build" --parallel "$1" --target FileCheck
"$LLVM_BUILD_ROOT/build/bin/llvm-lit" -v \
  /workspace/llvm-source/mlir/test/Analysis/tile-operation-graph-mixed-width.mlir
cmake --install "$LLVM_BUILD_ROOT/build"
"$LLVM_BUILD_ROOT/install/bin/mlir-opt" --help | rg -- 'mixed-width-matmul'
"$LLVM_BUILD_ROOT/install/bin/llvm-config" --version --has-rtti --assertion-mode --targets-built
