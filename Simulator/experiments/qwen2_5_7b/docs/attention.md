# Real-width attention bring-up

This is one Qwen2.5-7B attention module with **synthetic weights**, not a whole
decoder layer or full-model inference. It uses the pinned Transformers 4.43.4
`Qwen2Attention` eager implementation: hidden width 3,584, 28 query heads,
four KV heads, and head dimension 128. We retain BF16 weights/activations and
the reference's FP32 softmax computation.

## Run in this order

Select the [experimental BF16 toolchain](toolchain.md) and the
[patched TOGSim build](#build-the-completion-fix) first, then:

```bash
# Independent CPU check of cached attention against full-prefix recomputation.
bash Simulator/experiments/qwen2_5_7b/run.sh cpu --component attention --seq-len 8 --decode-steps 2

# Isolate lookup, rotation, RoPE arithmetic, GQA, both BMMs, softmax, and cache append.
bash Simulator/experiments/qwen2_5_7b/run.sh functional --component attention_parts --seq-len 8

# Real-width assembled attention; carry the simulator-produced cache forward.
bash Simulator/experiments/qwen2_5_7b/run.sh functional --component attention --seq-len 8 --decode-steps 2

# One continuous timing session, with numerical checks and actual lookup indices.
bash Simulator/experiments/qwen2_5_7b/run.sh timing --component attention --seq-len 8 --decode-steps 2 --validate-timing
```

`workloads/attention.py` exposes the reference's mutable `DynamicCache` as explicit
input/output tensors. It does not replace attention math or move rotary lookup
onto the CPU. The cache remains four KV heads; seven-way head replication is
used only by attention's matrix products. Input/mask/position preparation and
CPU comparisons are host work, outside the simulated device timing boundary.

`--validate-timing` uses heuristic mapping and enables both Spike functional
execution and gem5/TOGSim timing. Spike provides the actual values and indirect
DMA index files; gem5 provides instruction timing; TOGSim schedules the kernels
in one session. Host validation time is not reported as simulated device time.
This checks the selected mapping, not an exhaustive search of all mappings.
After an assembled attention timing run, `run.sh` also performs the separate
inter-kernel dependency audit and returns failure if it does not pass. Its
report is `analysis/dependency_audit.json`; `result.json` records the functional
and simulator-execution result, not that post-run audit.

Pure timing mode cannot safely handle these data-dependent lookups: TOGSim
otherwise warns about missing index files and falls back to direct addresses.
The frontend now rejects that case, including indirect-address autotuning.

## What is checked

- Output, attention probabilities, and K/V contents against CPU, using the
  existing BF16 tolerances (`rtol=0.02`, `atol=0.02`), without relaxing them.
- Cached CPU execution against independently recomputed full prefixes.
- Probability row sums, exactly zero future-token weights, four-head cache
  shapes, and bit-exact preservation of the old cache prefix on each decode.
- Pure lookup, rotation, GQA replication, and cache-copy checks are exact.

Each run records `result.json`, `attention_phases.json`, generated MLIR and
RISC-V, and per-kernel raw input/output files. Timing phases also record kernel
IDs for matching the launch trace. These are numerical and simulator-integration
checks, **not validation of timing accuracy against physical TPUv3 hardware**.

## Functional result (2026-09-04)

Eight-token prefill plus two cached single-token decode steps passes:

| Phase | Output maximum absolute error | Probability maximum absolute error |
|---|---:|---:|
| Prefill | 0.00390625 | 0.00390625 |
| Decode 1 | 0.001953125 | 0.0009765625 |
| Decode 2 | 0.001953125 | 0.0009765625 |

The maximum K-cache error is 0.0078125; V-cache error is at most 0.00000190735.
Evidence: `/data2/s2chitni/.tmp/qwen25-functional.t2Q6n4/result.json`.

## Timing and unit activity

The corrected combined functional/timing run passed with **559,658 simulated cycles**
at 940 MHz (approximately 0.595 ms). It launched 64 compiled kernels in one
session: 20 for prefill and 22 for each decode step. This is the simulator's
heuristic-mapping result for one attention module, not full-model or native
TPUv3 latency. Evidence: `/data2/s2chitni/.tmp/qwen25-timing.gROSiL/result.json`.
The previous 543,101-cycle run is **not a valid performance baseline**:
its consumer kernels could read buffers before their producers wrote them.
For a controlled timing-only comparison, replaying the original saved graphs,
addresses, and launch trace gives 543,101 cycles with the original binary and
564,411 with the patched binary (+3.92%). Both use exactly the same input
artifacts. The fresh functional/timing rerun has new allocations and reports
559,658 cycles; its +3.05% difference is not an isolated measurement of the fix.
Controlled logs: `/data2/s2chitni/.tmp/qwen25-toolchain.T27Una/console.log`
(original) and `/data2/s2chitni/.tmp/qwen25-toolchain.dJ4zIn/console.log`
(patched). The patched replay also passes the full inter-kernel audit.

| Unit | Nonempty compute queue, cycles | Queue occupancy | Native active-cycle counter |
|---|---:|---:|---:|
| VPU | 99,375 | 17.76% | 105,538 |
| MXU 0 | 102,855 | 18.38% | 9,637 |
| MXU 1 | 117,170 | 20.94% | 9,981 |

| Phase | Cycles | VPU queue occupancy | MXU 0 | MXU 1 |
|---|---:|---:|---:|---:|
| Prefill | 192,730 | 19.26% | 17.38% | 21.03% |
| Decode 1 | 183,557 | 16.95% | 19.73% | 20.75% |
| Decode 2 | 183,370 | 16.98% | 18.07% | 21.02% |

These measures are intentionally distinct. Queue occupancy is the union of
`[issue_cycle, finish_cycle)` intervals, including queued/pipelined work. Native
counters subtract the simulator's modeled bubble term; the VPU also counts an
extra cycle at each retirement. The analyzer reproduces **all three native
counters exactly**, including periodic resets and saturating subtraction.
Neither measure is a direct count of useful arithmetic or active physical MACs.

The trace does not explicitly identify each MXU. The analyzer uses the verified
core's round-robin rule, checks that no zero-cycle matrix nodes were silently
skipped, and verifies every predicted finish against the observed event.
`compute_intervals.csv` preserves the individual cycle intervals;
`occupancy_windows.csv` and the plot average occupancy over 5,000-cycle windows.

**The new phase markers require tile retirement and DMA-response drain.**
The corrected trace passes all 63 inter-kernel boundaries, checks all 29,652
constructed instructions and 1,313 issued DMA transfers, and contains 452
explicit store-response completions. The last kernel completes at 559,657;
the simulation ends one cycle later. The audit checks dispatch and first
instruction issue against the previous kernel's completion, and every kernel
completion against its last instruction/response event. It does not establish
intra-kernel memory-hazard correctness or timing accuracy on real hardware.

The originally failing dependency is now ordered correctly: K0's Q-output
store issues at 68,042, finishes injection at 68,154, and receives all responses
at 68,231. K0 completes at 68,232; K1's Q read issues at 68,234. The simulator
no longer admits the consumer before that output is available in its memory model.

To analyze the corrected recorded run:

```bash
QWEN_INPUT_RUN="$TMPDIR/qwen25-timing.gROSiL" \
  bash Simulator/experiments/qwen2_5_7b/run.sh toolchain bash -ec \
  'python -B -m Simulator.experiments.qwen2_5_7b.analysis.trace_summary "$QWEN_INPUT_RUN" --output-dir "$TMPDIR/trace_analysis" --plot
   python -B -m Simulator.experiments.qwen2_5_7b.analysis.operator_timeline "$QWEN_INPUT_RUN" "$TMPDIR/trace_analysis"'
```

`QWEN_INPUT_RUN` binds the saved run read-only. All outputs go into the new
analysis invocation's scratch directory. `trace_summary.py` refuses to produce
utilization results if the dependency audit fails. `operator_timeline.py`
checks its semantic labels against wrapper origins and byte-matched saved MLIR.

## Build the completion fix

The image's original binary remains unchanged. Copy its dependency objects to
scratch, then compile all TOGSim translation units from this checkout and link
against those dependencies. No dependency download is needed:

```bash
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain bash -ec \
  'cp -a TOGSim/build "$TMPDIR/togsim-build"'
# Use the exact Run directory printed by that command:
export QWEN_TOGSIM_BUILD="$TMPDIR/qwen25-toolchain.XXXXXX/togsim-build"
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain \
  python -B -m Simulator.experiments.qwen2_5_7b.tools.build_togsim
```

Keep `QWEN_TOGSIM_BUILD` selected when running timing experiments. It binds only
scratch build artifacts and the checkout's source/header directories. A build
SHA-256 manifest records the source and executable. New launcher invocations
copy that manifest to their run directory when this build is selected.

Local patched build: `/data2/s2chitni/.tmp/qwen25-toolchain.xyHHnw/togsim-build`.
The original binary fails 9/10 new C++ regression cases; the patched build
passes 10/10, including pending stores, async loads, multi-slot completion,
same-kernel async overlap, immediate responses, and zero-length DMA. Run the
tests again with the `tools.build_togsim --test-only` module entry point. Python analysis tests run with
`python3 -B -m unittest discover -s Simulator/experiments/qwen2_5_7b/tests/unit -p 'test_*.py'`.

## Bugs exposed by composing the operators

1. **BF16 BMM accumulation:** add an FP32 scratchpad accumulator, preserving
   partial sums across K tiles and rounding once before the output epilogue.
2. **Negation:** preserve the floating-point operand type in generated MLIR.
3. **Indirect lookup:** move `arith-expand` after DMA lowering and TOG extraction.
   Its folding had removed the special affine marker carrying row indices.
4. **Narrow matrix tiles:** use physical scratchpad vector loads for matrix
   weights; logical last-dimension transfer bounds wrongly masked K elements
   when the output tile had only eight columns.
5. **Short K tails:** pad BF16 values as integer-vector bits. A BF16 shuffle
   became scalar loads/broadcasts, incorrectly repeating lane 0 across lanes.
6. **TOGSim kernel completion:** track dispatched-but-unretired tiles and
   outstanding DMA independently of the ready/blocked queues; refresh the
   scheduler on retirement and polling. Retain DMA tile ownership through
   responses and log store acknowledgements separately from injection.

The BMM suite forces three K tiles for its accumulation tests and includes
eight-/nine-column and short-K cases. Run the `tests.integration.batched_matmul`
and `tests.integration.indirect_lookup` module entry points through launcher
`toolchain` mode (see [the commands](toolchain.md#validate-in-order)). BMM's ordinary tile selector does not yet
honor the external GEMM mapping file; the forced mapping is local to the test.

Next gates are broader dependency coverage, a complete decoder layer, two consecutive layers, and then
memory/runtime scaling toward 28 layers plus embeddings and the output head.
Neither full-model fit nor end-to-end model utilization is established here.
