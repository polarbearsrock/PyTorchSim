# B=1, S=128 — corrected unit-attribution rerun

## Current result — 2026-09-08

The fresh run with the permanent native LLVM fix completed all 93 kernels.
The known matrix-result-tail attribution bug is absent from the generated
regions and issued trace. **Timing-accounting checks pass; full-layer numerical
validation still fails.** This remains an exploratory simulator result, not an
accepted numerical baseline or a physical TPU measurement.

Scope is unchanged: one full-width, unmodified Transformers 4.43.4
`Qwen2DecoderLayer`, synthetic BF16 weights, seed 0, batch 1, 128-token prefill,
then two one-token decode steps (cache lengths 129/130). The single modeled
TPUv3 core has two MXUs and a VPU, at 940 MHz. No model, mapping policy, or
numerical tolerance was changed.

These are the same **compute-queue occupancy** measurements used previously:
the union of half-open `[issue, finish)` intervals divided by the complete
phase duration. They include queued/pipelined work and MXU preloads, not just
useful arithmetic. Empty compute queues alone do not establish the bottleneck.

| Phase | Cycles | Time at 940 MHz | VPU occupied | MXU0 occupied | MXU1 occupied | All three empty |
|---|---:|---:|---:|---:|---:|---:|
| Prefill, S=128 | 3,531,146 | 3.757 ms | 23.26% | 29.28% | 29.56% | 52.08% |
| Decode 1, cache=129 | 2,498,930 | 2.658 ms | 9.88% | 9.98% | 10.03% | 87.51% |
| Decode 2, cache=130 | 2,498,479 | 2.658 ms | 9.88% | 9.99% | 10.04% | 87.51% |

Total is **8,528,556 cycles / 9.073 ms**, including one final drain cycle.
Host wall time was 1,050.763 seconds (17.5 minutes), including compilation and
functional execution; it is not simulated inference latency.

Prefill VPU occupancy falls from the invalidated 74.11% to 23.26%.
VPU and at least one MXU overlap for only **194,429 cycles: 5.51% of the
whole prefill phase, or 11.49% of cycles with any compute queue occupied**.
The remaining prefill VPU work includes the fused sigmoid/multiply/cast kernel
(kernel 26, 279,424 occupied cycles) and vector portions of the matrix kernels.
The new figures come from a fresh schedule, not subtraction from the old graph.

- [Whole-run occupancy, 5,000-cycle windows](/data2/s2chitni/.tmp/qwen25-toolchain.3DOKRU/analysis/queue_occupancy.png)
- [Prefill exact intervals](/data2/s2chitni/.tmp/qwen25-toolchain.3DOKRU/analysis/unit_activity_prefill.png)
- [Decode 1 exact intervals](/data2/s2chitni/.tmp/qwen25-toolchain.3DOKRU/analysis/unit_activity_decode_1.png)
- [Decode 2 exact intervals](/data2/s2chitni/.tmp/qwen25-toolchain.3DOKRU/analysis/unit_activity_decode_2.png)
- [Summary and simultaneous queue states](/data2/s2chitni/.tmp/qwen25-toolchain.3DOKRU/analysis/summary.json)
- [Per-kernel inventory](/data2/s2chitni/.tmp/qwen25-toolchain.3DOKRU/analysis/kernel_inventory.json)

### What was checked

- All 48 host unit tests passed.
- The [attribution audit](/data2/s2chitni/.tmp/qwen25-toolchain.3DOKRU/analysis/attribution_audit.json)
  checks all 64 unique timing sources and all 93 kernel invocations, reconciling
  135,736 issued compute jobs with graph unit labels, gem5 latencies, and loop
  repetition counts. All matrix result reads are inside matrix regions, with
  complete accumulator-write coverage; lowered LLVM IR preserves that coverage.
  Large BF16 matrix regions contain eight input pushes, sixteen FP32 result
  reads, and sixteen accumulator writes.
- All 64 functional object files are byte-identical to the historical run.
  Only the five unique timing objects affected by the bug changed. The 14,224
  removed compute instructions correspond to the spurious VPU tail regions,
  not removed functional arithmetic.
- The [dependency audit](/data2/s2chitni/.tmp/qwen25-toolchain.3DOKRU/analysis/dependency_audit.json)
  passes for 93 kernels, 92 boundaries, 243,632 instructions, 21,992 DMA
  transfers, and 6,960 store-response events. Every predicted compute finish
  and the native counters reconcile exactly; all 32 matrix graph nodes have
  positive latencies.
- Both [saved decode value expansions](/data2/s2chitni/.tmp/qwen25-toolchain.3DOKRU/value_expansion_saved.json)
  pass bit-exact checks (462,336 and 465,920 FP32 elements). Old-cache prefix
  preservation and K/V tolerances pass. Hidden-output numerical statistics
  are unchanged from the historical run: 14,331 / 458,752, 87 / 3,584, and
  90 / 3,584 elements remain outside `rtol=0.02, atol=0.02`.

Run: `/data2/s2chitni/.tmp/qwen25-timing.EjU9VX`.
The [result manifest](/data2/s2chitni/.tmp/qwen25-timing.EjU9VX/result.json)
records `completed_with_numerical_mismatch` and the actual selected binary
hashes. LLVM revision is `e528d7b07639780de4685b5037a150daa0257a12`, selected
automatically from the permanent installation; gem5 remains the fixed
`555dbe315d92fcead1aef095b72f6bad19822282` installation.

The launch command and toolchain exports recorded below also reproduce this
configuration now that local LLVM selection includes the native fix. For
analysis, substitute the new run path above for the historical `QWEN_INPUT_RUN`.
The additional read-only attribution audit script is saved at
`/data2/s2chitni/.tmp/qwen25-utilization-audit.xSenhK/audit_attribution.py`.

## Historical run — invalid unit attribution

**Do not use the graphs, utilization percentages, or total timing below as a
corrected baseline.** A subsequent compiler audit found that the LLVM TOG pass
assumed equal matrix input-push and result-read counts. For BF16 inputs and
FP32 results at M=128, it closed the matrix region after eight of sixteen
result reads, assigning the remaining reads and accumulator updates to VPU.
This affects scheduling and total runtime, not just graph colors.

The [TOG pass fix and regressions](tog.md) address this separately from gem5.
The corrected full-layer rerun is recorded above. The old trace's 1,938,832 misattributed
prefill VPU cycles are 70.2% of its reported VPU occupancy; **subtracting them
does not produce corrected utilization**, because the schedule must change too.
Original dependency/counter checks below remain useful consistency checks,
but did not validate the compiler's unit labels. Numerical failures are separate.

The historical **2026-09-07** run completed all 93 kernels: one full-width upstream
Qwen2 decoder layer, batch 1, 128-token prefill, and two one-token decode steps
(cache lengths 129/130). It used the [fixed gem5 installation](gem5.md), with
fresh generated kernels and latency statistics. No model, tiling policy, or
numerical tolerance was changed for this retry.

**Dependency/counter consistency checks passed, but unit attribution is invalid;
full-layer numerical validation also still fails.** The model uses synthetic
BF16 weights, not the Qwen checkpoint. These are not physical TPU measurements.

## Invalidated historical graphs and timing

- [Whole-run queue-occupancy graph](/data2/s2chitni/.tmp/qwen25-toolchain.wLRtmr/analysis/queue_occupancy.png)
- [Prefill exact activity intervals](/data2/s2chitni/.tmp/qwen25-toolchain.wLRtmr/analysis/unit_activity_prefill.png)
- [Decode 1 exact activity intervals](/data2/s2chitni/.tmp/qwen25-toolchain.wLRtmr/analysis/unit_activity_decode_1.png)
- [Decode 2 exact activity intervals](/data2/s2chitni/.tmp/qwen25-toolchain.wLRtmr/analysis/unit_activity_decode_2.png)
- [Machine-readable summary](/data2/s2chitni/.tmp/qwen25-toolchain.wLRtmr/analysis/summary.json)
- [Kernel inventory and per-kernel occupancy](/data2/s2chitni/.tmp/qwen25-toolchain.wLRtmr/analysis/kernel_inventory.json)

Percentages below are the union of `[issue, finish)` compute-queue residence
intervals, divided by the entire phase duration. They include queued/pipelined
work; they are **not useful-FLOP or physical MAC utilization**. The overview
averages 5,000-cycle windows; the phase plots retain exact interval endpoints.

| Phase | Cycles | Time at 940 MHz | VPU queue occupied | MXU0 queue occupied | MXU1 queue occupied | All three queues empty |
|---|---:|---:|---:|---:|---:|---:|
| Prefill, S=128 | 3,725,914 | 3.964 ms | 74.11% | 25.92% | 28.07% | 25.26% |
| Decode 1, cache=129 | 2,500,581 | 2.660 ms | 9.87% | 9.98% | 10.03% | 87.52% |
| Decode 2, cache=130 | 2,498,070 | 2.658 ms | 9.88% | 9.99% | 10.04% | 87.50% |

The full trace is **8,724,566 cycles / 9.281 ms**, including one final drain
cycle. Host wall time was 1,017.912 seconds (about 17 minutes), including model
setup, compilation, gem5 latency generation, Spike functional execution, and
TOGSim scheduling. It is not simulated inference time.

The apparent high prefill VPU occupancy is contaminated by the attribution bug;
do not infer the layer's bottleneck from it. An empty compute queue alone also
does not prove how much idle time is avoidable.

## Verification and numerical limitations

- All 38 host unit tests passed on this retry.
- All [11 gem5 matrix-width regressions](/data2/s2chitni/.tmp/qwen25-toolchain.exOohQ/gem5-matrix/result.json) passed using the permanent installation.
- The [dependency audit](/data2/s2chitni/.tmp/qwen25-timing.gmvfrY/dependency-audit.log) checked all 93 kernels, 92 inter-kernel boundaries, 257,856 instructions, 21,992 DMA transfers, and 6,960 store-response events.
- The analyzer verified 32 positive-latency matrix graph nodes and reconciled every compute finish cycle and the native counters exactly. Native active-cycle counters (VPU 3,318,928; MXU0 883,469; MXU1 982,500) use different bubble/retirement accounting from the plotted queue occupancy.
- All 93 wrapper calls matched launched MLIR byte-for-byte, with phase coverage checked. Kernel labels remain compiler origins, not guessed operator names.
- The actual [saved decode value expansions](/data2/s2chitni/.tmp/qwen25-toolchain.wLRtmr/value_expansion_saved.json) matched upstream `repeat_kv(saved_input, 7).float()` bit-exactly: 462,336 elements at kernel 50 and 465,920 at kernel 82, with zero mismatches.
- CPU cached-versus-full-prefix checks and both bit-exact old-cache preservation checks passed. All K/V elements passed the unchanged CPU tolerances.

Hidden outputs still fail the existing `rtol=0.02, atol=0.02` checks:

| Phase | Maximum absolute hidden-output difference | Elements outside tolerance |
|---|---:|---:|
| Prefill | 0.125 | 14,331 / 458,752 |
| Decode 1 | 0.09375 | 87 / 3,584 |
| Decode 2 | 0.0625 | 90 / 3,584 |

The [run result](/data2/s2chitni/.tmp/qwen25-timing.gmvfrY/result.json) therefore
records `completed_with_numerical_mismatch`, not `passed`. These residual
differences still require investigation; the timing checks do not establish
their cause or waive them.

## Historical command and corrected-rerun prerequisite

The exports below document the old run. The corrected frontend requires the
[source-built LLVM compiler with the native TOG fix](tog.md), now selected by
this workspace's local configuration. Use a new cache. New timings are a new experiment,
not a re-analysis that can correct this saved trace.

Simulator source: `99aa0edcf4c0e40a30e8e3fd95bc40f26224cdaf`.
Gem5 fork source: `555dbe315d92fcead1aef095b72f6bad19822282`.
The actual gem5 binary SHA-256 recorded by the run is
`0f58402a8c8b7b56e48faa6285051495fc9cf8b624132c0eb29adc05f6834c1b`.
The result also records the Spike and BF16 plugin hashes.

```bash
export QWEN_GEM5_ROOT=/data2/s2chitni/psal-postech/gem5
export QWEN_TOGSIM_BUILD=/data2/s2chitni/.tmp/qwen25-toolchain.xyHHnw/togsim-build
export QWEN_TOOLCHAIN_ROOT=/data2/s2chitni/.tmp/qwen-bf16-toolchain.7o13Ab
export TORCHSIM_SPIKE=/workspace/bf16-toolchain/install/bin/spike
export TORCHSIM_BF16_PLUGIN=/workspace/bf16-toolchain/libPyTorchSimBF16.so
QWEN_TIMEOUT_SECONDS=7200 bash Simulator/experiments/qwen2_5_7b/run.sh timing \
  --component decoder --batch 1 --seq-len 128 --decode-steps 2 \
  --validate-timing --allow-numerical-mismatch
```

The two `/workspace/bf16-toolchain` paths are container paths. Omitting
`QWEN_GEM5_ROOT` selects the old image gem5, not the fixed installation.
To reproduce the analysis without rerunning the model:

```bash
QWEN_INPUT_RUN=/data2/s2chitni/.tmp/qwen25-timing.gmvfrY \
QWEN_TIMEOUT_SECONDS=900 \
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain bash -ec '
  python -B -m Simulator.experiments.qwen2_5_7b.analysis.trace_summary \
    "$QWEN_INPUT_RUN" --output-dir "$TMPDIR/analysis" \
    --allow-numerical-mismatch --plot
  python -B -m Simulator.experiments.qwen2_5_7b.analysis.kernel_inventory \
    "$QWEN_INPUT_RUN" "$TMPDIR/analysis" --allow-numerical-mismatch
  python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.value_expansion \
    --saved-run "$QWEN_INPUT_RUN"
'
```

All generated data and graphs remain under TMPDIR. The source run is mounted
read-only during analysis. The historical failed attempt below is preserved.

## Historical failed attempt — 2026-09-06

**The earlier attempt did not complete or produce S=128 utilization results.**
The unchanged toolchain stalled during gem5 latency generation for the larger
BF16 matrix tiles. The run was stopped after bounded replays confirmed that
instructions were no longer retiring; all its processes have exited.

Update: the [gem5 matrix-width accounting bug is now fixed](gem5.md). All five
unchanged timing binaries that stalled in this attempt complete with the rebuilt
simulator. The successful full-workload retry is recorded above.

## Requested scope

- Same unmodified Transformers 4.43.4 `Qwen2DecoderLayer` as the S=8 experiment.
- Qwen2.5-7B-Instruct dimensions, one full-width layer, synthetic BF16 weights,
  seed 0, no checkpoint, embedding, final model norm, or LM head.
- One TPUv3-modeled core, 940 MHz, two MXUs, 128-lane VPU; heuristic mapping.
- Batch 1, prefill 128, then two one-token decode steps (cache lengths 129/130).
- Same corrected frontend and existing BF16/TOGSim binaries as the corrected
  [S=8 run](utilization.md#corrected-rerun). No model, compiler, mapping policy,
  or numerical tolerance was changed for this attempt.

The attempted command, after selecting the existing toolchain/build exports:

```bash
QWEN_TIMEOUT_SECONDS=7200 bash Simulator/experiments/qwen2_5_7b/run.sh timing \
  --component decoder --batch 1 --seq-len 128 --decode-steps 2 \
  --validate-timing --allow-numerical-mismatch
```

## What stopped execution

The 29-call prefill wrapper was generated, but only the first two normalization
kernels were enqueued. Five gem5 latency-generation processes for matrix kernels
continued consuming CPU without completing new statistics checkpoints.
Prefill numerical validation and both decode steps were never completed.

An isolated, read-only replay examined the already-generated output-projection
kernel: `M=128, N=3584, K=3584`, with tile
`(TILE_M,TILE_N,TILE_K)=(128,1792,1792)`.

| Bounded gem5 replay | Final tick | Retired instruction count | Cycles since latest statistics reset |
|---|---:|---:|---:|
| First | 2,200,000,000 | 2,065,973 | 133,316 |
| Extended | 5,000,000,000 | 2,065,973 | 2,933,316 |

There is **no instruction-retirement progress over 2.8 million additional
gem5 cycles**. The last retired instruction is at `0x12ccc`; the next
instruction, at `0x12cd0`, is `sf.vc.v.i 0x2, 0x0, v9, 0x0`, a matrix-result
read emitted by the BF16/FP32 matrix lowering. This localized the stall; the
underlying compiler/simulator contract was subsequently diagnosed and corrected
as described in the [gem5 fix report](gem5.md).
It is not a completed CPU-versus-Spike numerical mismatch.

The bounded diagnostic runs intentionally exit at their tick limits, not with
successful kernel completion. Their measured tick limits are not kernel
latencies and must not be used to fill in the missing timing data.

## Preserved evidence

- [Explicit interruption report](/data2/s2chitni/.tmp/qwen25-timing.nTVAMD/run_outcome.json)
- [Full attempt log](/data2/s2chitni/.tmp/qwen25-timing.nTVAMD/console.log)
- [Workload contract](/data2/s2chitni/.tmp/qwen25-timing.nTVAMD/workload.json)
- [First bounded replay log](/data2/s2chitni/.tmp/qwen25-toolchain.fgjbZ3/console.log)
- [First replay statistics](/data2/s2chitni/.tmp/qwen25-toolchain.fgjbZ3/probe/stats.txt)
- [Extended replay statistics](/data2/s2chitni/.tmp/qwen25-toolchain.1lZ0q4/probe/stats.txt)
- [Short pipeline-state trace](/data2/s2chitni/.tmp/qwen25-toolchain.1lZ0q4/probe/minor.log)
- [Instruction disassembly](/data2/s2chitni/.tmp/qwen25-toolchain.4U5WZo/console.log)

All raw/intermediate files remain under TMPDIR. The launcher exited 143 after
the verified experiment process group was terminated; no runner `result.json`
or complete TOGSim trace was produced. The separate `run_outcome.json` records
this explicitly without inventing a passed or completed experiment.

The corrected-gem5 S=8 baseline still needs regeneration. Forcing smaller tiles would be a
different mapping experiment and should be labeled separately, not silently
substituted for this run.
