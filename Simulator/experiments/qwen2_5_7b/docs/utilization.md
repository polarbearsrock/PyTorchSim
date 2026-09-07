# Exploratory decoder utilization — 2026-09-06

Timing-toolchain update: both runs preserved below used the original gem5.
Its [matrix vector-length accounting bug is now fixed](gem5.md). Regenerate
these timing results with the corrected simulator before comparing against
new S=128 measurements; small-run completion did not validate short/tail timing.

The measurements below preserve the **original, pre-fix** experiment. Its
decode value-expansion bug is now [fixed in the frontend tiler](value-expansion.md).
Do not reuse the original decode trace as a corrected model baseline.
See the [corrected rerun](#corrected-rerun) for the new measurements and plots.

The complete one-layer prefill/decode experiment finished and produced an
audited timing trace. **Numerical validation failed.** Prefill reproduces the
previous BF16 rounding difference; decode has a separate, large value-head
expansion/conversion error. These are measurements of the current compiled
PyTorchSim program, not an accepted Qwen or native TPUv3 performance baseline.

## Scope and configuration

- Unmodified Transformers 4.43.4 `Qwen2DecoderLayer`, Qwen2.5-7B-Instruct dimensions.
- One layer; synthetic BF16 weights, seed 0; no pretrained checkpoint.
- Batch 1; 8-token prefill, then two cached 1-token calls (cache lengths 9 and 10).
- Hidden width 3584, MLP width 18944, 28 query heads, 4 KV heads, head width 128.
- One modeled TPUv3 core at 940 MHz, two MXUs and one 128-lane VPU.
- Heuristic mapping, unchanged BF16 toolchain and corrected TOGSim completion build.
- Functional Spike execution enabled alongside gem5/TOGSim timing. Host input,
  mask preparation, numerical checks, and compilation are outside device timing.
- No embedding, final model norm, LM head, token generation, or full-model execution.

## Results

Occupancy is the union of half-open `[issue_cycle, finish_cycle)` intervals:
a nonempty modeled compute queue, including queued/pipelined work. It is **not**
the fraction of peak FLOPs achieved or physical MAC activity.

| Phase | Kernels | Core cycles | Simulated ms | VPU queue | MXU0 queue | MXU1 queue |
|---|---:|---:|---:|---:|---:|---:|
| Prefill, 8 tokens | 29 | 2,507,511 | 2.668 | 10.46% | 9.67% | 9.82% |
| Decode 1, cache 9 | 32 | 2,457,746 | 2.615 | 9.61% | 9.57% | 9.80% |
| Decode 2, cache 10 | 32 | 2,461,738 | 2.619 | 9.63% | 9.59% | 9.79% |

Including one post-completion cycle, total execution is **7,426,996 cycles / 7.901 ms**.
Wall-clock execution took 749.54 seconds (about 12.5 minutes), not 7.901 ms.

- All three compute queues are empty for 87.75% of the total interval. This does
  not mean the memory system is idle, and it does not establish a cause for every gap.
- All three queues are occupied simultaneously for 7.49% of the total interval.
- Conditional on at least one occupied compute queue, VPU and at least one MXU
  overlap for 61.20%; the two unit types do not overlap for the remaining 38.80%.
- The three large MLP matrix-kernel lifetimes account for 90.12% of prefill and
  about 91.1–91.3% of each decode phase. These lifetimes include their memory
  activity and waits; they are not matrix arithmetic cycles.

The native counters use different accounting. Across the whole run they report
779,603 VPU active cycles, 77,357 MXU0 active cycles, and 76,772 MXU1 active cycles
(10.50%, 1.04%, and 1.03%). The analyzer reproduces all three exactly, including
bubble subtraction, VPU retirement increments, and periodic counter resets.

## Checks and decode issue in the original run

- All 93 wrapper calls match the 93 submitted kernels in order and their saved
  MLIR byte-for-byte. No direct ATen/extern-kernel calls occur in the generated
  wrappers. This is source/launch coverage, not proof of all backend semantics.
- The dependency audit passes all 92 inter-kernel boundaries, 219,890 constructed
  instructions, 4,321 DMA transfers, and 1,364 store-response completions.
- Every reconstructed compute finish matches the observed event. All 34 matrix
  graph nodes checked have positive latency, permitting verified round-robin MXU assignment.
- Shapes, finite values, CPU cached/full-prefix checks, and exact preservation
  of the simulator's old KV prefix pass.
- Prefill hidden error: max 0.0703125; 742 / 28,672 elements fail the unchanged tolerance.
- Decode hidden errors: max 7.58203125 and 7.859375; 3,510 and 3,508 / 3,584
  elements fail. These are not established harmless rounding differences.

The saved decode value-expansion kernels (K50 and K82) do not agree with
`Transformers repeat_kv(saved_input, 7).float()`. The input/output shapes are
`[1,4,9,128] -> [1,28,9,128]` and `[1,4,10,128] -> [1,28,10,128]` respectively.
Both have maximum error 6.65625; 29,567 / 32,256 and 32,870 / 35,840 elements
fail the same tolerance. This check uses the current run's exact saved inputs.

A separate diagnostic on the first attempt verified the original inputs,
parameters, CPU expected output, and saved simulator output bit-for-bit. Local
matrix operations, softmax, Q RoPE, scaled K expansion, and context cast pass;
the value expansion is the identified failing attention step. The subsequent
[diagnosis and correction](value-expansion.md) identifies the wrong tile-axis
lookup. The original artifacts remain unchanged, and their numerical gate
has not been relabeled passed.

The initial attempt also exposed a host-checker mistake: the old-cache reference
was left on the simulated device. The driver now snapshots it on CPU before
cache update, and comparison moves both operands to CPU. The new tests cover
this path and retain strict cache checks.

## Artifacts and reproduction

- [Source run and recorded settings](/data2/s2chitni/.tmp/qwen25-timing.9mgJdo/result.json)
- [Analysis summary](/data2/s2chitni/.tmp/qwen25-toolchain.Y5Cd3F/analysis/summary.json)
- [Averaged occupancy plot](/data2/s2chitni/.tmp/qwen25-toolchain.Y5Cd3F/analysis/queue_occupancy.png)
- [Exact prefill intervals plot](/data2/s2chitni/.tmp/qwen25-toolchain.Y5Cd3F/analysis/unit_activity_prefill.png)
- [Exact decode 1 intervals plot](/data2/s2chitni/.tmp/qwen25-toolchain.Y5Cd3F/analysis/unit_activity_decode_1.png)
- [Exact decode 2 intervals plot](/data2/s2chitni/.tmp/qwen25-toolchain.Y5Cd3F/analysis/unit_activity_decode_2.png)
- [Individual compute intervals CSV](/data2/s2chitni/.tmp/qwen25-toolchain.Y5Cd3F/analysis/compute_intervals.csv)
- [Source-linked kernel inventory](/data2/s2chitni/.tmp/qwen25-toolchain.Y5Cd3F/analysis/kernel_inventory.json)
- [Per-kernel occupancy CSV](/data2/s2chitni/.tmp/qwen25-toolchain.Y5Cd3F/analysis/kernel_queue_occupancy.csv)
- [Current-run value-expansion check](/data2/s2chitni/.tmp/qwen25-toolchain.x0FkWg/value_expansion_check.json)
- [First-attempt local numerical diagnosis](/data2/s2chitni/.tmp/qwen25-toolchain.PASMeB/decode_local_math.json)

All raw/intermediate outputs remain under TMPDIR. The first attempt at
`/data2/s2chitni/.tmp/qwen25-timing.OJBZts` is preserved as a failed run, not rewritten.

Select the [BF16 toolchain](toolchain.md) and [corrected TOGSim build](attention.md#build-the-completion-fix), then:

```bash
QWEN_TIMEOUT_SECONDS=1200 bash Simulator/experiments/qwen2_5_7b/run.sh timing \
  --component decoder --seq-len 8 --decode-steps 2 \
  --validate-timing --allow-numerical-mismatch
```

The `--allow-numerical-mismatch` option records differences; it does not certify
that any difference is benign. Without it the original strict gate still applies.
See the [exploratory analysis commands](transformers.md#exploratory-utilization-with-numerical-differences).
For source-linked kernel statistics, run `analysis.kernel_inventory RUN ANALYSIS_DIR
--allow-numerical-mismatch` after `analysis.trace_summary`, using the same analysis directory.

Verification: 33 host unit tests, 9 CPU numerical-policy/upstream integration
tests, and 10/10 scheduler regression cases passed for the original experiment.
The subsequent compiler fix and rerun are recorded separately below.

## Corrected rerun

The fresh run `qwen25-timing.StiNdu` uses the same model, seed, inputs, dimensions,
BF16 settings, simulator binaries, heuristic mapping policy, and validation
tolerances. Only the frontend tile-constraint lookup changed. All 93 wrapper
calls retain their operand metadata and compiler origins. Generated MLIR changes
at the six K/V expansion calls (K12, K18, K42, K50, K74, K82); the other 87
kernel sources are unchanged.

### Numerical verification

The actual saved K50 and K82 outputs now match upstream
`repeat_kv(saved_input, 7).float()` **bit-for-bit**, with zero mismatches out of
32,256 and 35,840 FP32 elements. The per-head checks pass for all 28 heads.
The same exact checker rejects the original run, where only heads 0 and 1
matched. This directly verifies the corrected data transformation, independently
of the larger layer's arithmetic tolerances.

| Full-layer hidden output | Original max absolute error | Corrected max absolute error | Corrected elements outside tolerance |
|---|---:|---:|---:|
| Prefill | 0.0703125 | 0.0703125 | 742 / 28,672 |
| Decode 1 | 7.58203125 | 0.078125 | 134 / 3,584 |
| Decode 2 | 7.859375 | 0.09375 | 111 / 3,584 |

KV comparisons, finite/shape/dtype checks, CPU cached/full-prefix checks, and
exact preservation of the simulator's old cache prefix pass. The separate
full-layer BF16 precision discrepancy remains: the run still has status
**`completed_with_numerical_mismatch`**, not `passed`. Fixing the expansion does
not certify that every remaining difference is benign or change the acceptance
contract. There is no pretrained or full-model result here.

### Corrected timing and occupancy

The queue-occupancy definition remains the one above, not useful FLOPs:

| Phase | Core cycles | Simulated ms | VPU queue | MXU0 queue | MXU1 queue |
|---|---:|---:|---:|---:|---:|
| Prefill, 8 tokens | 2,496,336 | 2.656 | 10.14% | 9.74% | 9.86% |
| Decode 1, cache 9 | 2,443,205 | 2.599 | 9.35% | 9.69% | 9.84% |
| Decode 2, cache 10 | 2,441,540 | 2.597 | 9.36% | 9.61% | 9.89% |

Total, including the one-cycle tail: **7,381,082 cycles / 7.852 ms**. Host
wall-clock execution took 747.13 seconds. All three compute queues are empty
for 88.02% of this interval, without implying that the memory system is idle.
VPU plus at least one MXU overlap for 62.94% of compute-nonempty time; the two
unit types do not overlap for 37.06%. All three queues overlap for 7.54% of total time.

All 92 inter-kernel boundaries pass the dependency audit, covering 218,864
instructions, 3,637 DMA transfers, and 1,022 store-response completions. All 93
wrapper calls match launched MLIR byte-for-byte. All 34 matrix graph nodes have
positive latency and reconstructed compute finishes agree with the trace.
Native active-counter replay reconciles exactly: VPU 753,835 cycles (10.21%),
MXU0 77,435 (1.05%), MXU1 76,493 (1.04%). These are different counters from queue
occupancy, not a hardware-calibrated arithmetic-efficiency measurement.

### Corrected artifacts and tests

- [Run settings and numerical status](/data2/s2chitni/.tmp/qwen25-timing.StiNdu/result.json)
- [Exact expansion check on the actual decode buffers](/data2/s2chitni/.tmp/qwen25-toolchain.Gk7i5R/value_expansion_saved.json)
- [Analysis summary](/data2/s2chitni/.tmp/qwen25-toolchain.Gk7i5R/analysis/summary.json)
- [Averaged MXU/VPU occupancy plot](/data2/s2chitni/.tmp/qwen25-toolchain.Gk7i5R/analysis/queue_occupancy.png)
- [Exact prefill intervals](/data2/s2chitni/.tmp/qwen25-toolchain.Gk7i5R/analysis/unit_activity_prefill.png)
- [Exact decode 1 intervals](/data2/s2chitni/.tmp/qwen25-toolchain.Gk7i5R/analysis/unit_activity_decode_1.png)
- [Exact decode 2 intervals](/data2/s2chitni/.tmp/qwen25-toolchain.Gk7i5R/analysis/unit_activity_decode_2.png)
- [Individual compute intervals CSV](/data2/s2chitni/.tmp/qwen25-toolchain.Gk7i5R/analysis/compute_intervals.csv)
- [Source-linked kernel inventory](/data2/s2chitni/.tmp/qwen25-toolchain.Gk7i5R/analysis/kernel_inventory.json)
- [Per-kernel occupancy CSV](/data2/s2chitni/.tmp/qwen25-toolchain.Gk7i5R/analysis/kernel_queue_occupancy.csv)

Post-fix verification: 33 host unit tests; 15 frontend tests including the three
new pre-fix-failing axis regressions; 9 CPU validation/upstream tests; 9 exact
compiled expansion cases; and the existing 22 BF16 arithmetic/matrix/BMM/lookup
cases all pass. Evidence for the CPU and adjacent compiled suites is in
`qwen25-toolchain.3EJCgc/console.log` and `qwen25-toolchain.PdBOJt/console.log`
under TMPDIR. The final expansion suite is in
`qwen25-toolchain.dgjrwT/value_expansion.json`.

The remaining numerical acceptance decision and investigation of the long MLP
compute-empty intervals are separate from the now-fixed expansion bug.
