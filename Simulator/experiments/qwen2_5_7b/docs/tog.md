# Mixed-width matrix timing fix — integrated in LLVM

The fix now lives in the **normal compiler source**, in the permanent fork
checkout at `/data2/s2chitni/psal-postech/llvm-project`, on
`shwet/qwen-performance-tpuv3`. It is compiled into `mlir-opt` under the
original `test-tile-operation-graph` pass name. The separate TOG plugin and
its experiment-specific build helpers have been retired.

See the [main LLVM build/install documentation](../../../../scripts/toolchains/llvm/README.md)
for the fork, source files, local commit, permanent installation, default
selection, reproducible build commands, and verification evidence.

## Why this fix is necessary

The LLVM TOG pass decides which operations belong to `MatmulCompute` and
`VectorCompute` timing regions. The bundled v1.0.8 pass assumed that matrix
input-push and result-read instruction counts were equal. With 256-bit
transfers, BF16 inputs carry 16 elements while FP32 results carry eight.

At M=128, eight input pushes therefore accompany sixteen result reads. The
old pass closed the matrix region after the first eight reads and accumulator
writes, assigning the remaining identical operations to VPU. M=8 happened to
have one push and one read, so that case did not expose the faulty assumption.

The corrected pass counts the actual emitted result-read sequence in each
matrix segment. It retains all reads and corresponding accumulator writes in
the matrix region, while leaving the following vector epilogue on VPU. It
handles short and padded transfers without assuming a fixed 2x ratio.

This is independent of the [gem5 actual-VL readiness fix](gem5.md). It makes
the simulator's operation grouping consistent; it does not establish physical
TPU instruction placement or make queue occupancy equal to useful-FLOP utilization.

## Verify through the standard compiler

The workspace's `configs/toolchains.local.json` selects the permanent LLVM
installation automatically. Keep the existing BF16/Spike, gem5 and TOGSim
selections. No TOG plugin export is needed.

```bash
QWEN_INPUT_RUN=/data2/s2chitni/.tmp/qwen25-timing.gmvfrY \
QWEN_TIMEOUT_SECONDS=900 \
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain bash -ec '
  python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.frontend_regression
  python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.tog_regions \
    --saved-run "$QWEN_INPUT_RUN"
'

QWEN_TIMEOUT_SECONDS=900 \
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain \
  python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.matrix_timing
```

The native compiler regression, all 59 saved unique kernel IR files, and fresh
BF16/FP16/FP32 end-to-end GEMMs pass. The original compiler remains available
inside the unchanged image solely for the regression's negative/control checks.

The earlier plugin established the fix before the permanent compiler was
built. Its helpers were archived at
`/data2/s2chitni/.tmp/pytorchsim-llvm-build.MePB6b/retired-tog-plugin`;
they are no longer part of the source tree's normal workflow.

The [fresh full-layer S=128 rerun](utilization-s128.md) now passes attribution
checks for all 64 timing sources and 93 kernel invocations. The old trace
remains invalidated. Separate full-layer numerical differences are unchanged
and are not resolved by this compiler attribution fix.
