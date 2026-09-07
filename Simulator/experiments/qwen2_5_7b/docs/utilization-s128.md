# B=1, S=128 utilization attempt — 2026-09-06

**The experiment did not complete. No S=128 utilization results are available.**
The unchanged toolchain stalled during gem5 latency generation for the larger
BF16 matrix tiles. The run was stopped after bounded replays confirmed that
instructions were no longer retiring; all its processes have exited.

Update: the [gem5 matrix-width accounting bug is now fixed](gem5.md). All five
unchanged timing binaries that stalled in this attempt complete with the rebuilt
simulator. The complete utilization experiment still needs a fresh rerun.

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

Next: repeat the same B=1, S=128 experiment with the corrected gem5 and regenerate
the S=8 baseline too. Forcing smaller tiles would be a
different mapping experiment and should be labeled separately, not silently
substituted for this run.
