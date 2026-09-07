#!/usr/bin/env bash
# Download immutable inputs and build without changing the container image.
set -euo pipefail
: "${TMPDIR:?Set TMPDIR to a scratch directory with sufficient space}"
QWEN_GEM5_SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
QWEN_GEM5_REV=0511678eb5334c1102725cd928f2d3de8720d1bc
QWEN_GEM5_BUILD_ROOT=$(mktemp -d "$TMPDIR/qwen-gem5-toolchain.XXXXXX")
printf 'gem5 toolchain directory: %s\n' "$QWEN_GEM5_BUILD_ROOT"
curl -fsSL --max-time 180 "https://codeload.github.com/PSAL-POSTECH/gem5/tar.gz/$QWEN_GEM5_REV" \
  -o "$QWEN_GEM5_BUILD_ROOT/gem5-release.tar.gz"
python3 -m pip download --no-cache-dir --no-deps --only-binary=:all: \
  --index-url https://pypi.org/simple scons==4.8.1 \
  --dest "$QWEN_GEM5_BUILD_ROOT/build-deps"
QWEN_GEM5_ROOT= QWEN_GEM5_BUILD_ROOT="$QWEN_GEM5_BUILD_ROOT" \
QWEN_TIMEOUT_SECONDS="${QWEN_GEM5_BUILD_TIMEOUT_SECONDS:-3600}" \
  bash "$QWEN_GEM5_SCRIPTS/../../run.sh" toolchain \
  bash Simulator/experiments/qwen2_5_7b/tools/gem5/build_in_container.sh \
  "${QWEN_GEM5_BUILD_JOBS:-16}"
printf '\nUse the rebuilt gem5 for new, isolated runs:\nexport QWEN_GEM5_ROOT=%q\n' "$QWEN_GEM5_BUILD_ROOT"
