#!/usr/bin/env bash
# Install to a NEW prefix; never overwrite an existing compiler installation.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  printf 'Usage: install.sh SCRATCH_BUILD_ROOT LLVM_SOURCE NEW_INSTALL_PREFIX\n' >&2; exit 2
fi
LLVM_SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LLVM_BUILD_ROOT=$(realpath -e -- "$1")
LLVM_SOURCE=$(realpath -e -- "$2")
LLVM_INSTALL=$(realpath -m -- "$3")
if [ -e "$LLVM_INSTALL" ]; then
  printf 'Refusing to overwrite existing installation: %s\n' "$LLVM_INSTALL" >&2; exit 2
fi
python3 -B "$LLVM_SCRIPTS/record_build.py" verify "$LLVM_BUILD_ROOT/install" --source "$LLVM_SOURCE"
cp -a -- "$LLVM_BUILD_ROOT/install" "$LLVM_INSTALL"
python3 -B "$LLVM_SCRIPTS/record_build.py" verify "$LLVM_INSTALL" --source "$LLVM_SOURCE"
printf 'Installed LLVM: %s\n' "$LLVM_INSTALL"
