# Decode value-head expansion: compiler fix (2026-09-06)

The large decode error came from PyTorchSim's **frontend tile-constraint axis
lookup**, not the upstream Transformers model or BF16-to-FP32 rounding. The
correction is in `TileAdjustMixin.apply_constraints` in
[mlir_common.py](../../../../PyTorchSimFrontend/mlir/mlir_common.py).
No Transformers code, model math, tolerance, Spike binary, or gem5 binary was
changed for this fix.

## Required operation

Qwen has 28 query heads and 4 KV heads. Grouped-query attention therefore repeats
each value head seven times. For the first decode step:

```text
BF16 value cache: [1, 4, 9, 128]
repeat_kv(values, 7).float()
FP32 values:     [1, 28, 9, 128]
```

Heads 0–6 must copy KV head 0, heads 7–13 copy KV head 1, and so on. The compiler
fuses this replication into the cast before the probability-times-value matrix
multiply. Both replication and BF16-to-FP32 widening are exact operations.

## Where the wrong indexing arose

Flattening the token/head-dimension pair gives 1,152 values per head. The compiled
input address is `index1 + 1152 * FloorDiv(index0, 7)`: `index0` selects the output
head and `index1` selects the value within that head.

`BaseMLIRKernel.extract_dividers` correctly records the divisor constraint under
the **symbol** `index0`. However, expression traversal encounters `index1`
first, so the resulting dictionary is ordered `index1`, then `index0`.
The old `apply_constraints` discarded the keys:

```python
for idx, (axis_constraints, axis_size) in enumerate(zip(constraints.values(), ranges)):
```

It consequently forced axis 1 to size 7 instead of axis 0. A direct regression
with the real address expression demonstrates `[256, 7]` where `[7, 256]` is
required. The fix looks up each axis by its `indexN` symbol; dictionary order
and absent unconstrained axes no longer change its meaning.

The downstream effect is visible in the generated MLIR:

| First decode value-expansion kernel | Before | After |
|---|---|---|
| Output tile: heads × flattened values | `28 × 7` | `7 × 256` |
| Input scratchpad: KV heads × repetitions × values | `4 × 7 × 7` | `1 × 7 × 256` |
| Lane-distributed dimension | Head dimension | Flattened value dimension |
| Input/output used lanes, stride 2 | 2 / 14 | 128 / 128 |

In the broken kernel, splitting the head index into KV heads and repetitions
left the input DMA distributing the *four KV heads* across lanes, while the
output DMA distributed the *28 output heads*. The vector loop therefore read
and wrote incompatible per-lane layouts. The head tile happened to be divisible
by seven, so the existing divisibility/recompile check did not reject it.
See `select_vlane_axis` in `mlir_common.py` and the FloorDiv handling in
[mlir_codegen_backend.py](../../../../PyTorchSimFrontend/mlir/mlir_codegen_backend.py).

With the correct constraint, the head tile contains one group of seven repeats.
The data dimension is lane-distributed consistently on both sides. The final
partial data tile is also covered by the numerical regression.

## Regression evidence

- Before the fix, an isolated upstream `repeat_kv(..., 7).float()` at shape
  `[1,4,9,128]` failed: 29,836 / 32,256 elements differed, maximum error 4.71875.
  [Report](/data2/s2chitni/.tmp/qwen25-toolchain.h0Qd5L/value_expansion.json).
- All three new tile-constraint tests failed before the correction: reversed
  insertion order, sparse axis keys, and the real decode address expression.
  [Log](/data2/s2chitni/.tmp/qwen25-toolchain.UWYSQx/console.log).
- After the correction, all 15 frontend tests pass, including those three.
  [Log](/data2/s2chitni/.tmp/qwen25-toolchain.QxvbCx/console.log).
- All nine compiled expansion cases pass **bit-for-bit**: contiguous cache
  lengths 8/9/10/17; transposed projection layout; batch 2; repetition factors
  1/2/7; and BF16-only replication without a cast.
  [Report](/data2/s2chitni/.tmp/qwen25-toolchain.dgjrwT/value_expansion.json).
- The exact saved-buffer checker rejects the original full-layer run's K50 and
  K82: 29,952 / 32,256 and 33,280 / 35,840 mismatched FP32 values respectively,
  maximum error 6.65625. These are **bit-exact** counts, not the older
  tolerance-based counts in the historical utilization report.
  [Report](/data2/s2chitni/.tmp/qwen25-toolchain.w7YrqP/value_expansion_saved.json).
- In the fresh full-layer rerun, those two actual expansion buffers now match
  exactly: **0 / 32,256 and 0 / 35,840** mismatched FP32 elements, maximum error
  zero on both. [Report](/data2/s2chitni/.tmp/qwen25-toolchain.Gk7i5R/value_expansion_saved.json).

The older isolated GQA test used transposed projection-layout BF16 inputs and
returned BF16 outputs. It did not cover the contiguous decode cache fused with
an FP32 cast. The new regressions explicitly cover both layouts and outputs.

## Reproduce

After selecting the [BF16 toolchain](toolchain.md), run from the repository root:

```bash
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.frontend_regression
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.value_expansion
```

Use `--case decode_1` or `--case decode_2` to select one exact compiled regression.
To check the actual saved buffers from a complete decoder timing run without
re-executing it:

```bash
QWEN_INPUT_RUN="$TMPDIR/qwen25-timing.<run>" \
  bash Simulator/experiments/qwen2_5_7b/run.sh toolchain bash -ec \
  'python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.value_expansion --saved-run "$QWEN_INPUT_RUN"'
```

The saved-buffer checker requires one recognized expansion kernel per decode
phase, verifies wrapper/launch source correspondence, and checks raw FP32 bits
against upstream `repeat_kv` on that kernel's actual saved BF16 input. It fails
on unknown coverage instead of silently skipping a changed graph.

This fixes the identified data corruption; it does not remove the separate
eager-versus-compiled BF16 precision differences. Full-layer comparisons retain
the original `rtol=0.02, atol=0.02` gate. See [utilization results](utilization.md)
for the corrected full-layer rerun and its numerical status.
