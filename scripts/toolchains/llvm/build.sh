#!/usr/bin/env bash
# Build the normal LLVM/MLIR tools. Source/install are permanent; scratch is not.
set -euo pipefail
: "${TMPDIR:?Set TMPDIR to a scratch directory with sufficient space}"
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  printf 'Usage: build.sh LLVM_SOURCE [EXISTING_SCRATCH_BUILD_ROOT]\n' >&2; exit 2
fi
LLVM_SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LLVM_SOURCE=$(realpath -e -- "$1")
LLVM_BUILD_ROOT=${2:-$(mktemp -d "$TMPDIR/pytorchsim-llvm-build.XXXXXX")}
LLVM_BUILD_ROOT=$(realpath -e -- "$LLVM_BUILD_ROOT")
case "$LLVM_BUILD_ROOT" in
  "$(realpath -e -- "$TMPDIR")"/*) ;;
  *) printf 'Build root must be under TMPDIR\n' >&2; exit 2 ;;
esac
LLVM_JOBS=${LLVM_BUILD_JOBS:-48}
if ! [[ "$LLVM_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'LLVM_BUILD_JOBS must be a positive integer\n' >&2; exit 2
fi
LLVM_IMAGE=${PYTORCHSIM_IMAGE:-$TMPDIR/torchsim-ci-v1.1.0.sif}
test -f "$LLVM_SOURCE/llvm/CMakeLists.txt"
test -f "$LLVM_IMAGE"
mkdir -p "$LLVM_BUILD_ROOT"/{tmp,cache,apptainer-config}
git -C "$LLVM_SOURCE" rev-parse HEAD > "$LLVM_BUILD_ROOT/source-revision.txt"
git -C "$LLVM_SOURCE" diff --binary > "$LLVM_BUILD_ROOT/source.diff"
sha256sum "$LLVM_SOURCE/mlir/test/lib/Analysis/TestTileOperationGraph.cpp" \
  "$LLVM_SOURCE/mlir/tools/mlir-opt/mlir-opt.cpp" \
  > "$LLVM_BUILD_ROOT/tog-source-before.sha256"
export APPTAINER_TMPDIR="$LLVM_BUILD_ROOT/tmp"
export APPTAINER_CACHEDIR="$TMPDIR/apptainer-cache"
export APPTAINER_CONFIGDIR="$LLVM_BUILD_ROOT/apptainer-config"
printf 'LLVM build directory: %s\n' "$LLVM_BUILD_ROOT"
printf 'Build log: %s/build.log\n' "$LLVM_BUILD_ROOT"
timeout --signal=TERM --kill-after=30s "${LLVM_BUILD_TIMEOUT_SECONDS:-7200}s" \
  apptainer exec --cleanenv --no-home \
  --bind "$LLVM_SOURCE:/workspace/llvm-source:ro" \
  --bind "$LLVM_BUILD_ROOT:/workspace/llvm-build" \
  --bind "$LLVM_BUILD_ROOT/tmp:/tmp" \
  --bind "$LLVM_SCRIPTS:/workspace/pytorchsim-llvm-scripts:ro" \
  --env TMPDIR=/workspace/llvm-build/tmp \
  --env XDG_CACHE_HOME=/workspace/llvm-build/cache \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --pwd /workspace/llvm-build "$LLVM_IMAGE" \
  bash /workspace/pytorchsim-llvm-scripts/build_in_container.sh "$LLVM_JOBS" \
  > "$LLVM_BUILD_ROOT/build.log" 2>&1
sha256sum --check "$LLVM_BUILD_ROOT/tog-source-before.sha256"
python3 -B "$LLVM_SCRIPTS/record_build.py" record "$LLVM_SOURCE" "$LLVM_BUILD_ROOT"
printf 'Built and checked. Staged installation: %s/install\n' "$LLVM_BUILD_ROOT"
