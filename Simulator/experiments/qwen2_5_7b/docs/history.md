# Qwen study history and experiment contract

This preserves the bring-up findings and prior run evidence. For current entry points
and the source layout, start with [the study README](../README.md).

# Qwen2.5-7B single-core TPUv3 study

This directory starts the **compatibility and correctness** stage of the study.
It does not yet implement complete-model NPU inference, and it does not contain
a full-model utilization result.

The initial BF16 toolchain blockers now have tested fixes. Real-width RMSNorm,
query projection, MLP, and **attention with cached decode** pass numerical
checks. Attention also completes continuous gem5/TOGSim timing using the
completion fix, with audited inter-kernel boundaries and a counter-reconciled
VPU/MXU queue-occupancy trace. The original 543,101-cycle run has an invalid
inter-kernel schedule and is superseded by the 559,658-cycle rerun. See the
[attention walkthrough](attention.md) and [toolchain guide](toolchain.md).

## Experiment contract

- Model: `Qwen/Qwen2.5-7B-Instruct`, revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Text-only, batch 1, BF16 weights/activations; retain FP32 reductions where
  required by the reference implementation.
- Single simulated core, two MXUs, one VPU, 940 MHz, with the existing TPUv3
  timing configuration. Budget against 16 GiB of HBM for capacity checks.
- Separate prefill from cached, single-token decode. Initial runs use synthetic
  weights and fixed token sequences, not checkpoint-driven text generation.
- Measure device execution only. Compilation, checkpoint loading, tokenization,
  and host-side sampling are outside the initial timing boundary.
- Preserve generated DMA, barriers, dependencies, and fusion. The eventual
  complete-model trace must use a continuous TOGSim session.
- FP32 probe runs are diagnostic controls, **not a change to the BF16 baseline**.

The [pinned upstream configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/config.json)
is embedded in `model.json`. The parameter and weight-byte counts are checked
against the
[checkpoint index](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/model.safetensors.index.json).
No checkpoint tensors are downloaded by these scripts.

## Memory audit

The 28-layer model has hidden size 3,584, intermediate size 18,944, 28 query
heads, four KV heads, head dimension 128, and vocabulary size 152,064.

The analytical inventory includes Q/K/V biases, both untied embedding/output
matrices, all feed-forward weights, and normalization weights. It reproduces
the checkpoint's **7,615,616,512 parameters** exactly.

| Item | Bytes |
|---|---:|
| BF16 weights | 15,231,233,024 |
| BF16 KV cache per token per sequence | 57,344 |
| KV cache at batch 1, 2,048 total cached tokens | 117,440,512 |
| Remaining from 16 GiB, before other allocations | 1,831,195,648 |

This is a lower-bound memory budget, **not a verified fit**. Activations,
compiler workspace, allocation lifetimes, padding, and runtime allocations
still need measurement. The scripts do not alter or claim to enforce the
simulator's DRAM capacity.

## Reproduce the first-stage checks

For BF16 NPU runs, first build/select the patched toolchain using the
[BF16 bring-up instructions](toolchain.md). CPU/audit and FP32 control
runs do not require that extension.

From the repository root:

```bash
python3 -B -m unittest discover -s Simulator/experiments/qwen2_5_7b/tests/unit -p 'test_*.py'
bash Simulator/experiments/qwen2_5_7b/run.sh audit --context-tokens 2048
bash Simulator/experiments/qwen2_5_7b/run.sh cpu --seq-len 8 --decode-steps 2
bash Simulator/experiments/qwen2_5_7b/run.sh functional --component rmsnorm --seq-len 16
bash Simulator/experiments/qwen2_5_7b/run.sh functional --component rmsnorm --seq-len 16 --dtype float32
bash Simulator/experiments/qwen2_5_7b/run.sh timing --component q_proj --seq-len 1
bash Simulator/experiments/qwen2_5_7b/run.sh timing --component q_proj --seq-len 1 --dtype float32
```

`audit` needs only Python's standard library. The other modes use Apptainer and
the existing `$TMPDIR/torchsim-ci-v1.1.0.sif`; override its location using
`QWEN_IMAGE`. Each run prints its unique directory under `$TMPDIR`, containing
`console.log`, `result.json`, generated code, and, when applicable, traces.
Failed compiler probes return a nonzero exit status. A process killed by the
outer timeout may have only `console.log`, without `result.json`.

The launcher defaults to a 300-second wall-time limit, configurable using
`QWEN_TIMEOUT_SECONDS`. Both framework caches and the container's `/tmp` are
redirected into the run directory; no host `/tmp` or home-directory scratch is
used. Hub access is disabled inside the container.

The experiment, `PyTorchSimFrontend`, and `Simulator` directories are mounted
read-only from this checkout, so these probes use our frontend and functional
I/O fixes. Compiled dependencies still come from the image unless the opt-in
BF16 toolchain is selected. The observed image versions are PyTorch
`2.8.0+cu126` and Transformers `4.43.4`.

## First-stage findings (2026-09-04)

These are the **initial, pre-fix** results. The subsequent BF16 bring-up is
documented in the [BF16 toolchain guide](toolchain.md).

| Check | Result |
|---|---|
| Four standard-library audit tests | Passed |
| Full Qwen2 model instantiated on PyTorch's metadata-only device | Parameter count matches exactly; no full-size weights allocated |
| Reduced CPU model: eight-token prefill plus two cached decode forwards | Passed; cached logits match recomputed-prefix logits for both steps |
| Real-width BF16 RMSNorm, `[1,16,3584]` | Failed in frontend type conversion: `Unsupported conversion: bf16 -> f32` |
| Same RMSNorm in FP32, through Spike | Passed; maximum absolute error `4.76837158203125e-7` |
| Real-width BF16 query projection, `[1,3584] @ [3584,3584]` plus bias | Failed during LLVM/RISC-V code generation, before usable gem5 timing |
| Same query projection in FP32, timing only | Compiled and completed gem5/TOGSim; numerical output not validated |

The CPU correctness model has two layers, hidden size 896, intermediate size
1,024, seven query heads and one KV head. It preserves head dimension 128 and
the 7:1 GQA ratio, but **is not a performance proxy for the full 7B model**.

The RMSNorm failure is traceable to `ExtensionOverrides.to_dtype` in
`PyTorchSimFrontend/mlir/mlir_ops.py`: floating-point types are classified by
their first character (`f`), which does not recognize `bf16`. The official
Qwen2 RMSNorm performs BF16-to-FP32 conversion, reduction/normalization, and a
conversion back to the input type.

The separate BF16 projection reaches the RISC-V LLVM compiler and fails with
`LLVM ERROR: Do not know how to split this operator's operand!`. Its generated
IR contains bfloat vectors and custom vector-coprocessor intrinsics. Frontend
type classification alone was insufficient; the subsequent changes address
matrix lowering, raw memory transport, and simulator compatibility separately.

Local evidence from the first runs:

- CPU: `/data2/s2chitni/.tmp/qwen25-cpu.f19EmP/result.json`
- BF16 RMSNorm: `/data2/s2chitni/.tmp/qwen25-functional.LjJJ3J/console.log`
- FP32 RMSNorm: `/data2/s2chitni/.tmp/qwen25-functional.GbEoik/result.json`
- BF16 projection: `/data2/s2chitni/.tmp/qwen25-timing.4sti42/console.log`
- FP32 projection control: `/data2/s2chitni/.tmp/qwen25-timing.IqnJwa/result.json`

These scratch paths are local evidence, not portable inputs; the commands above
produce fresh runs.

## Remaining gates

The early kernel-completion bug is fixed and all 63 kernel boundaries in the
attention rerun pass the new DMA-response-aware audit. Continue auditing
intra-kernel memory hazards and broader configurations; this is not validation
against physical TPUv3. See [ATTENTION.md](attention.md).

1. Extend coverage beyond the tested short attention sequence and heuristic
   mapping. Unsupported reduction-fusion paths remain disabled; indirect-address
   autotuning is rejected until it can supply real index traces.
2. Compose the tested attention and MLP/norm pieces with residuals into a layer.
3. Validate one complete layer and then two consecutive layers. Preserve cache
   contents/positions and compare numerical outputs against the CPU reference.
4. Measure runtime and trace-size growth before attempting all 28 layers,
   embeddings, final norm, and the vocabulary projection. Specify whether
   prefill computes only last-position logits; Transformers 4.43's ordinary
   forward computes logits for every input position.
5. Capture a continuous model-wide trace with phase, token, layer, and compiled
   operation labels. Plot MXU 0/1 occupancy, VPU occupancy, DMA issuance and
   outstanding traffic. Keep occupancy distinct from useful-compute accounting;
   attribute idle gaps only when dependency/resource evidence supports it.
6. Reconcile reconstructed intervals with native counters, measure peak memory,
   then sweep prompt/context lengths at batch 1. Produce full-run overview plots
   and cycle-level zooms for selected layers and tokens.

The initial BF16 casts/GEMM/BMM blockers have fixes and regression coverage.
Attention with eight-token prefill and two cached decode steps is checked;
complete layers and model-wide validation remain. No full-model performance
or roofline claim is made.
