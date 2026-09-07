#!/usr/bin/env bash
set -euo pipefail
QWEN_GEM5_BUILD_ROOT=/workspace/gem5-toolchain
QWEN_GEM5_SCRIPTS=/workspace/PyTorchSim/Simulator/experiments/qwen2_5_7b/tools/gem5
QWEN_GEM5_REV=0511678eb5334c1102725cd928f2d3de8720d1bc
QWEN_GEM5_JOBS=${1:-16}
if ! [[ "$QWEN_GEM5_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Build job count must be a positive integer\n' >&2; exit 2
fi
cd "$QWEN_GEM5_BUILD_ROOT"
printf '%s  %s\n' \
  6716624b8b5150442b5de81a991aec175123ffb52d0a59a8f76a178ce80675b9 gem5-release.tar.gz \
  a4c3b434330e2d7d975002fd6783284ba348bf394db94c8f83fdc5bf69cdb8d7 build-deps/SCons-4.8.1-py3-none-any.whl \
  | sha256sum -c -
if [ ! -d "gem5-$QWEN_GEM5_REV" ]; then
  tar -xzf gem5-release.tar.gz
fi
if [ ! -f matrix-vl.applied.sha256 ]; then
  patch --batch --forward --fuzz=0 --directory "gem5-$QWEN_GEM5_REV" -p1 \
    --input "$QWEN_GEM5_SCRIPTS/matrix-vl.patch"
  sha256sum "$QWEN_GEM5_SCRIPTS/matrix-vl.patch" > matrix-vl.applied.sha256
else
  sha256sum -c matrix-vl.applied.sha256
fi
/usr/bin/python3 -m zipfile -e build-deps/SCons-4.8.1-py3-none-any.whl python-deps
export PYTHONPATH="$QWEN_GEM5_BUILD_ROOT/python-deps"
cd "gem5-$QWEN_GEM5_REV"
/usr/bin/python3 -m SCons build/RISCV/gem5.opt -j"$QWEN_GEM5_JOBS" \
  PYTHON_CONFIG=/usr/bin/python3-config --linker=lld --without-tcmalloc
mkdir -p "$QWEN_GEM5_BUILD_ROOT/bin"
cp build/RISCV/gem5.opt "$QWEN_GEM5_BUILD_ROOT/bin/gem5.opt"
/usr/bin/python3 "$QWEN_GEM5_SCRIPTS/record_build.py" "$QWEN_GEM5_BUILD_ROOT" "$QWEN_GEM5_JOBS"
