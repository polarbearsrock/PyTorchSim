# gem5 matrix-width fix and permanent installation

## Current installation

The corrected simulator is installed at
`/data2/s2chitni/psal-postech/gem5`. This is a real, shallow Git checkout of
[polarbearsrock/gem5](https://github.com/polarbearsrock/gem5), a fork of
[PSAL-POSTECH/gem5](https://github.com/PSAL-POSTECH/gem5).

- `origin`: `https://github.com/polarbearsrock/gem5.git`
- `upstream`: `https://github.com/PSAL-POSTECH/gem5.git`
- Local branch: `shwet/qwen-performance-tpuv3`
- Base: release `v1.0.1`, commit `0511678eb5334c1102725cd928f2d3de8720d1bc`
- Backport: only the `src/cpu/minor/execute.cc` vector-length accounting from
  upstream commit `3ac8959462ad42e5096598594d165d5887ce5b39`
- Executable: `bin/gem5.opt`
- Build record: `build.json`; fork/checkout verification: `installation.json`

The source fix is committed and pushed on the fork's branch as
[`555dbe3`](https://github.com/polarbearsrock/gem5/commit/555dbe315d92fcead1aef095b72f6bad19822282).
Runtime binaries and installation metadata are excluded by `.git/info/exclude`.
The container's original gem5 was not overwritten. All compiler intermediates,
downloads, caches, and test outputs remain under `$TMPDIR`; only the permanent
checkout, executable, and installation records live at the requested prefix.

Select this installation for subsequent study runs:

```bash
export QWEN_GEM5_ROOT=/data2/s2chitni/psal-postech/gem5
```

The launcher verifies the executable against `build.json`, mounts only the
executable read-only, and explicitly sets the container's `GEM5_PATH` to
`/workspace/gem5-toolchain/bin/gem5.opt`. It copies the build and installation
records into each new run. The runner also hashes the actual selected gem5.
Without this export the image's original executable is still the default.
The installed binary was built for execution inside this image; native execution
on a host with different Python/shared-library versions is not promised.

The Spike/BF16 plugin and TOGSim selections are independent and remain necessary;
see [BF16 setup](toolchain.md) and [the attention walkthrough](attention.md).

## Root cause and scope of the fix

The release binary counted every input push, weight push, and output pop as
eight elements. Our BF16 inputs carry sixteen elements per full 256-bit vector,
while FP32 results carry eight. For an M=128 block, the eight input pushes
created only 64 modeled readiness tokens. Eight result pops drained them, and
the ninth pop waited forever.

The backport uses the instruction's actual decoded VL for both output readiness
and committed push/pop accounting. It does not change Transformers, tiling,
numerical tolerances, opcodes, or VPU timing parameters. The source patch is
[matrix-vl.patch](../tools/gem5/matrix-vl.patch). No other changes from the
newer upstream fork were pulled into this release-based build.

The build uses the existing container's GCC 11.4 and Python 3.10, pinned SCons
4.8.1 installed only in scratch, the RISCV target, 16 jobs, LLD, and
`--without-tcmalloc`. Source archive, SCons wheel, patch, patched source,
build-script and executable checksums are recorded.

## Verification

The [regression harness](../tests/integration/gem5_matrix.py) checks actual
input/pop counts as well as normal termination. Cases cover zero VL, short
VL=2/7/8, BF16-to-FP32 width changes, M=16/128, clamped AVL, fractional LMUL,
LMUL=2/4 feeds, and FP32 controls. These are timing-queue tests, not numerical
GEMM validation. Tick-limit exits are failures, never latency measurements.

- Original binary: 3/11 controls passed; 8 failed, including four stalls.
- Rebuilt binary: all 11 controls passed.
- All five unchanged S=128 timing binaries that had stalled now terminate.
- The installed permanent copy passed all 11 controls again.
- The unchanged output-projection binary also completed using the normal
  production gem5 script, without the bounded diagnostic hook. Its statistics
  match the bounded replay; the six compute-region counts are
  `286, 15, 40, 416, 136, 6584` cycles. The final `6794` checkpoint is excluded
  by the normal cycle-list reader. These are kernel-sample region latencies,
  not a full-layer latency or TPU hardware measurement.

Evidence:

- [Original binary's failing regression report](/data2/s2chitni/.tmp/qwen25-toolchain.Kl7GcB/gem5-matrix/result.json)
- [Corrected controls and five saved-binary replays](/data2/s2chitni/.tmp/qwen25-toolchain.SWzOtd/gem5-matrix/result.json)
- [Permanent installation's passing regression report](/data2/s2chitni/.tmp/qwen25-toolchain.TC1Czd/gem5-matrix/result.json)
- [Production-script replay statistics](/data2/s2chitni/.tmp/qwen25-toolchain.Hydkrh/native-m5out/stats.txt)
- [Permanent build record](/data2/s2chitni/psal-postech/gem5/build.json)
- [Permanent fork/checkout record](/data2/s2chitni/psal-postech/gem5/installation.json)

The executable SHA256 is
`0f58402a8c8b7b56e48faa6285051495fc9cf8b624132c0eb29adc05f6834c1b`.
The original image binary is
`f641193c9c3255d0cb7c7df092ba43bdef700816ff61854a25332f39a847eca9`.

Run the bounded regressions from the repository root:

```bash
export QWEN_GEM5_ROOT=/data2/s2chitni/psal-postech/gem5
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain \
  python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.gem5_matrix
```

For a saved kernel, bind its original run with `QWEN_INPUT_RUN` and add
`--replay-binary "$QWEN_INPUT_RUN/generated/KERNEL_ID/cycle_bin"`. All output
goes into the new launcher's scratch directory, not into the saved run.

## Reproduce a build and install a fresh checkout

```bash
bash Simulator/experiments/qwen2_5_7b/tools/gem5/build.sh
```

The script downloads and checksum-verifies the pinned source and SCons wheel,
applies the focused patch, and builds under a newly printed TMPDIR directory.
`QWEN_GEM5_BUILD_JOBS` defaults to 16; `QWEN_GEM5_BUILD_TIMEOUT_SECONDS` defaults
to 3600. `QWEN_GEM5_BUILD_ROOT` is an internal, toolchain-only writable mount;
it must remain under TMPDIR and must not be combined with `QWEN_GEM5_ROOT`.

After the build, install from that printed build directory and a fork URL:

```bash
bash Simulator/experiments/qwen2_5_7b/tools/gem5/install.sh \
  "$TMPDIR/qwen-gem5-toolchain.PRINTED_SUFFIX" \
  /data2/s2chitni/psal-postech/gem5-next \
  https://github.com/polarbearsrock/gem5.git
```

The installer requires a new destination and never overwrites an existing
checkout. It fetches the pinned commit from the fork, adds the upstream remote,
applies the patch, copies the executable, and verifies that the installed source
matches the source that produced the binary. Source edits require rebuilding;
editing the checkout alone does not update the executable.

## Remaining experiment work

The stall is fixed and tested. The 2026-09-07 B=1, S=128 run was subsequently
invalidated by a separate [LLVM timing-region bug](tog.md). The
[2026-09-08 corrected rerun](utilization-s128.md) now completes with both fixes
and passing attribution, dependency, and trace-counter checks. Full-layer CPU
numerical tolerance differences remain unresolved. Regenerate the S=8 timing
baseline too: completion with the old eight-element assumption did not prove
that its short/tail vector timing was correct. Preserve the old results as
historical evidence; use fresh latency caches for the corrected simulator.
This gem5 fix does not resolve the separate full-layer numerical differences.
