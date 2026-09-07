#!/usr/bin/env bash
# Bounded pilot runs using the existing image; every transient file lives in TMPDIR.
set -euo pipefail
ulimit -c 0
: "${TMPDIR:?Set TMPDIR to a scratch directory with sufficient space}"
QWEN_EXPERIMENT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
QWEN_CHECKOUT=$(cd -- "$QWEN_EXPERIMENT/../../.." && pwd)
cd -- "$QWEN_CHECKOUT"
QWEN_MODE=${1:-audit}
if [ "$#" -gt 0 ]; then shift; fi
case "$QWEN_MODE" in
  -h|--help)
    printf 'Usage: bash Simulator/experiments/qwen2_5_7b/run.sh MODE [ARGS...]\n'
    printf 'Modes: audit, cpu, functional, timing; toolchain COMMAND [ARGS...]\n'
    printf 'Workload options: --component decoder|attention|rmsnorm|q_proj|mlp|attention_parts (default: decoder)\n'
    printf 'Use audit --help for all workload arguments. See the adjacent README.md.\n'
    exit 0 ;;
  audit) exec python3 -B -m Simulator.experiments.qwen2_5_7b --mode audit "$@" ;;
  cpu|functional|timing|toolchain) ;;
  *) printf 'Unknown mode: %s\n' "$QWEN_MODE" >&2; exit 2 ;;
esac
QWEN_IMAGE=${QWEN_IMAGE:-$TMPDIR/torchsim-ci-v1.1.0.sif}
if [ ! -f "$QWEN_IMAGE" ]; then
  printf 'Image not found: %s (set QWEN_IMAGE)\n' "$QWEN_IMAGE" >&2
  exit 2
fi
QWEN_RUN=$(mktemp -d "$TMPDIR/qwen25-$QWEN_MODE.XXXXXX")
mkdir -p "$QWEN_RUN"/{tmp,cache,generated,logs,gemm_candidates,apptainer-config}
export APPTAINER_CACHEDIR="$TMPDIR/apptainer-cache"
export APPTAINER_TMPDIR="$QWEN_RUN/tmp"
export APPTAINER_CONFIGDIR="$QWEN_RUN/apptainer-config"
printf 'Run directory: %s\n' "$QWEN_RUN"
QWEN_REPO=/workspace/PyTorchSim
QWEN_COMMAND=(python -B -u -m Simulator.experiments.qwen2_5_7b
  --mode "$QWEN_MODE" --output-dir "$QWEN_RUN" "$@")
if [ "$QWEN_MODE" = toolchain ]; then
  if [ "$#" -eq 0 ]; then
    printf 'toolchain mode requires a command\n' >&2
    exit 2
  fi
  QWEN_COMMAND=("$@")
fi
QWEN_STATUS=0
QWEN_EXTRA_ARGS=()
if [ -n "${QWEN_GEM5_ROOT:-}" ] && [ -n "${QWEN_GEM5_BUILD_ROOT:-}" ]; then
  printf 'Select either QWEN_GEM5_ROOT or QWEN_GEM5_BUILD_ROOT, not both\n' >&2; exit 2
fi
if [ -n "${QWEN_GEM5_ROOT:-}" ]; then
  # A permanent installation need not live in TMPDIR. Only build scratch does.
  QWEN_GEM5_ROOT=$(realpath -e -- "$QWEN_GEM5_ROOT")
  if [ ! -x "$QWEN_GEM5_ROOT/bin/gem5.opt" ] || [ ! -f "$QWEN_GEM5_ROOT/build.json" ]; then
    printf 'QWEN_GEM5_ROOT requires bin/gem5.opt and build.json; build gem5 first\n' >&2; exit 2
  fi
  python3 -B "$QWEN_EXPERIMENT/tools/gem5/record_build.py" --verify "$QWEN_GEM5_ROOT"
  QWEN_EXTRA_ARGS+=(--bind "$QWEN_GEM5_ROOT/bin/gem5.opt:/workspace/gem5-toolchain/bin/gem5.opt:ro")
  QWEN_EXTRA_ARGS+=(--env GEM5_PATH=/workspace/gem5-toolchain/bin/gem5.opt)
  cp "$QWEN_GEM5_ROOT/build.json" "$QWEN_RUN/gem5-build.json"
  if [ -f "$QWEN_GEM5_ROOT/installation.json" ]; then
    cp "$QWEN_GEM5_ROOT/installation.json" "$QWEN_RUN/gem5-installation.json"
  fi
fi
if [ -n "${QWEN_GEM5_BUILD_ROOT:-}" ]; then
  if [ "$QWEN_MODE" != toolchain ]; then
    printf 'QWEN_GEM5_BUILD_ROOT is only for toolchain build commands\n' >&2; exit 2
  fi
  QWEN_GEM5_BUILD_ROOT=$(realpath -- "$QWEN_GEM5_BUILD_ROOT")
  case "$QWEN_GEM5_BUILD_ROOT" in
    "$(realpath -- "$TMPDIR")"/*) ;;
    *) printf 'QWEN_GEM5_BUILD_ROOT must be under TMPDIR\n' >&2; exit 2 ;;
  esac
  QWEN_EXTRA_ARGS+=(--bind "$QWEN_GEM5_BUILD_ROOT:/workspace/gem5-toolchain")
fi
if [ -n "${QWEN_TOGSIM_BUILD:-}" ]; then
  QWEN_TOGSIM_BUILD=$(realpath -- "$QWEN_TOGSIM_BUILD")
  case "$QWEN_TOGSIM_BUILD" in
    "$(realpath -- "$TMPDIR")"/*) ;;
    *) printf 'QWEN_TOGSIM_BUILD must be under TMPDIR\n' >&2; exit 2 ;;
  esac
  QWEN_EXTRA_ARGS+=(--bind "$QWEN_TOGSIM_BUILD:$QWEN_REPO/TOGSim/build")
  QWEN_EXTRA_ARGS+=(--bind "$QWEN_CHECKOUT/TOGSim/src:$QWEN_REPO/TOGSim/src:ro")
  QWEN_EXTRA_ARGS+=(--bind "$QWEN_CHECKOUT/TOGSim/include:$QWEN_REPO/TOGSim/include:ro")
  if [ -f "$QWEN_TOGSIM_BUILD/completion-fix-sha256.json" ]; then
    cp "$QWEN_TOGSIM_BUILD/completion-fix-sha256.json" "$QWEN_RUN/togsim-build.json"
  fi
fi
QWEN_EXTRA_ARGS+=(--bind "$QWEN_CHECKOUT/TOGSim/tests:$QWEN_REPO/TOGSim/tests:ro")
if [ -n "${QWEN_INPUT_RUN:-}" ]; then
  QWEN_INPUT_RUN=$(realpath -- "$QWEN_INPUT_RUN")
  case "$QWEN_INPUT_RUN" in
    "$(realpath -- "$TMPDIR")"/*) ;;
    *) printf 'QWEN_INPUT_RUN must be under TMPDIR\n' >&2; exit 2 ;;
  esac
  QWEN_EXTRA_ARGS+=(--bind "$QWEN_INPUT_RUN:$QWEN_INPUT_RUN:ro")
  QWEN_EXTRA_ARGS+=(--env "QWEN_INPUT_RUN=$QWEN_INPUT_RUN")
fi
if [ -n "${QWEN_TOOLCHAIN_ROOT:-}" ]; then
  QWEN_TOOLCHAIN_ROOT=$(realpath -- "$QWEN_TOOLCHAIN_ROOT")
  case "$QWEN_TOOLCHAIN_ROOT" in
    "$(realpath -- "$TMPDIR")"/*) ;;
    *) printf 'QWEN_TOOLCHAIN_ROOT must be under TMPDIR\n' >&2; exit 2 ;;
  esac
  QWEN_EXTRA_ARGS+=(--bind "$QWEN_TOOLCHAIN_ROOT:/workspace/bf16-toolchain")
fi
if [ -n "${TORCHSIM_SPIKE:-}" ]; then
  QWEN_EXTRA_ARGS+=(--env "TORCHSIM_SPIKE=$TORCHSIM_SPIKE")
fi
if [ -n "${TORCHSIM_BF16_PLUGIN:-}" ]; then
  QWEN_EXTRA_ARGS+=(--env "TORCHSIM_BF16_PLUGIN=$TORCHSIM_BF16_PLUGIN")
fi
timeout --signal=TERM --kill-after=10s "${QWEN_TIMEOUT_SECONDS:-300}s" \
  apptainer exec --cleanenv --no-home \
  "${QWEN_EXTRA_ARGS[@]}" \
  --bind "$QWEN_RUN:$QWEN_RUN" \
  --bind "$QWEN_RUN/tmp:/tmp" \
  --bind "$QWEN_CHECKOUT/PyTorchSimFrontend:$QWEN_REPO/PyTorchSimFrontend:ro" \
  --bind "$QWEN_CHECKOUT/Simulator:$QWEN_REPO/Simulator:ro" \
  --bind "$QWEN_RUN/gemm_candidates:$QWEN_REPO/validation/gemm_candidates" \
  --env TMPDIR="$QWEN_RUN" \
  --env XDG_CACHE_HOME="$QWEN_RUN/cache" \
  --env XDG_CONFIG_HOME="$QWEN_RUN/cache/config" \
  --env MPLCONFIGDIR="$QWEN_RUN/cache/matplotlib" \
  --env HF_HOME="$QWEN_RUN/cache/huggingface" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 \
  --env TORCHSIM_DIR="$QWEN_REPO" \
  --env TORCHSIM_DUMP_PATH="$QWEN_RUN/generated" \
  --env TORCHSIM_LOG_PATH="$QWEN_RUN/logs" \
  --env TORCHINDUCTOR_CACHE_DIR="$QWEN_RUN/generated/.torchinductor" \
  --env TOGSIM_DEBUG_LEVEL=trace \
  --pwd "$QWEN_REPO" "$QWEN_IMAGE" \
  "${QWEN_COMMAND[@]}" \
  > "$QWEN_RUN/console.log" 2>&1 || QWEN_STATUS=$?
if [ "$QWEN_STATUS" -eq 0 ] && [ "$QWEN_MODE" = timing ] && { [ -f "$QWEN_RUN/attention_phases.json" ] || [ -f "$QWEN_RUN/decoder_phases.json" ]; }; then
  # Numerical success alone must not label an invalid timing schedule a pass.
  # The audit is stdlib-only and runs after the simulator's log is closed.
  python3 -B -m Simulator.experiments.qwen2_5_7b.analysis.dependency_audit "$QWEN_RUN" "$QWEN_RUN/analysis" \
    > "$QWEN_RUN/dependency-audit.log" 2>&1 || QWEN_STATUS=$?
  printf 'Dependency audit: %s/dependency-audit.log\n' "$QWEN_RUN"
fi
printf 'Exit status: %s; details: %s/console.log\n' "$QWEN_STATUS" "$QWEN_RUN"
exit "$QWEN_STATUS"
