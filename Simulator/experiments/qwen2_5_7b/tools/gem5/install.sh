#!/usr/bin/env bash
# Install a fork-backed source checkout + executable, without build objects.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  printf 'Usage: install.sh SCRATCH_BUILD_ROOT NEW_INSTALL_PREFIX FORK_URL\n' >&2; exit 2
fi
QWEN_GEM5_SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
QWEN_GEM5_FROM=$(realpath -e -- "$1")
QWEN_GEM5_DEST=$(realpath -m -- "$2")
QWEN_GEM5_FORK=$3
QWEN_GEM5_REV=0511678eb5334c1102725cd928f2d3de8720d1bc
if [ -e "$QWEN_GEM5_DEST" ]; then
  printf 'Installation destination already exists; refusing to overwrite: %s\n' "$QWEN_GEM5_DEST" >&2; exit 2
fi
case "$QWEN_GEM5_FORK" in
  https://github.com/PSAL-POSTECH/gem5*)
    printf 'Use your fork URL, not the upstream repository\n' >&2; exit 2 ;;
  https://github.com/*/gem5.git) ;;
  *) printf 'Expected https://github.com/OWNER/gem5.git fork URL\n' >&2; exit 2 ;;
esac
python3 -B "$QWEN_GEM5_SCRIPTS/record_build.py" --verify "$QWEN_GEM5_FROM" "$QWEN_GEM5_SCRIPTS/matrix-vl.patch"
mkdir -p "$(dirname -- "$QWEN_GEM5_DEST")"
mkdir "$QWEN_GEM5_DEST"
git init --quiet "$QWEN_GEM5_DEST"
git -C "$QWEN_GEM5_DEST" remote add origin "$QWEN_GEM5_FORK"
git -C "$QWEN_GEM5_DEST" remote add upstream https://github.com/PSAL-POSTECH/gem5.git
git -C "$QWEN_GEM5_DEST" fetch --depth=1 origin "$QWEN_GEM5_REV"
git -C "$QWEN_GEM5_DEST" switch --create shwet/qwen-performance-tpuv3 FETCH_HEAD
git -C "$QWEN_GEM5_DEST" apply --check "$QWEN_GEM5_SCRIPTS/matrix-vl.patch"
git -C "$QWEN_GEM5_DEST" apply "$QWEN_GEM5_SCRIPTS/matrix-vl.patch"
mkdir -p "$QWEN_GEM5_DEST/bin" "$QWEN_GEM5_DEST/pytorchsim-patches"
cp "$QWEN_GEM5_FROM/bin/gem5.opt" "$QWEN_GEM5_DEST/bin/gem5.opt"
cp "$QWEN_GEM5_FROM/build.json" "$QWEN_GEM5_DEST/build.json"
cp "$QWEN_GEM5_SCRIPTS/matrix-vl.patch" "$QWEN_GEM5_DEST/pytorchsim-patches/matrix-vl.patch"
cp "$QWEN_GEM5_SCRIPTS/install.gitignore" "$QWEN_GEM5_DEST/.git/info/exclude"
python3 -B "$QWEN_GEM5_SCRIPTS/record_build.py" --record-install "$QWEN_GEM5_DEST"
printf 'Installed source and gem5 binary. Select with:\nexport QWEN_GEM5_ROOT=%q\n' "$QWEN_GEM5_DEST"
