# Standard Transformers baseline

The default workload now calls the installed **Transformers 4.43.4
`Qwen2DecoderLayer` directly**. There is no local decoder subclass, copied
`forward`, monkey patch, or hand-assembled norm/attention/MLP sequence. The
existing attention wrapper and isolated operator tests remain diagnostics;
they are not the new complete-layer baseline.

## Ownership and timing boundary

| Responsibility | Implementation |
|---|---|
| Both RMSNorms, Q/K/V/O projections, RoPE, GQA, attention, SwiGLU, both residuals | Installed `transformers.models.qwen2.modeling_qwen2` |
| KV cache creation/update/append | Installed `transformers.cache_utils.DynamicCache`, using its public interface |
| Causal-mask construction | Installed `Qwen2Model._update_causal_mask`; host-side setup outside layer timing |
| Synthetic weight initialization | Installed Qwen2 initializer; not checkpoint weights |
| Inputs, CPU comparisons, phase names, simulator submission | [workloads/decoder.py](../workloads/decoder.py) |
| Compiler and TPUv3 timing model | Existing PyTorchSim/BF16 toolchain; not native TPU XLA |

A metadata-only `Qwen2Model` supplies its initializer and mask helper without
allocating all 28 layers or the embedding table. A separate, real
`Qwen2DecoderLayer(config, layer_idx=0)` is initialized and executed. No
metadata-only weights enter the execution. The config retains all original
Qwen dimensions; the result explicitly records that only one layer is measured.

`DynamicCache` persists between calls. The driver neither reads nor writes its
private cache lists or seen-token counter. Upstream attention owns cache mutation.
`output_attentions=False` matches normal inference and avoids exposing debug
probability outputs solely to preserve a particular compiler schedule.

The compiler may still fuse or reorder the upstream operations. Selecting
Transformers eager attention does **not** disable PyTorch compiler attention
rewrites. `fullgraph=True, dynamic=False` rejects Dynamo graph breaks and
specializes each cache shape; it is not, by itself, proof against every backend
fallback. Generated kernels and their device coverage also need auditing before
accepting performance results.

## Reproducibility and validation

[provenance.py](../provenance.py) requires `transformers==4.43.4`, checks the six
relevant package source files against their installed distribution's SHA256
records, and records their paths and hashes. This detects local file edits or
version drift, not malicious changes to the distribution record itself. No
package upgrade, model download, or remote model code is needed.

Runs record the actual PyTorch version, frontend and toolchain hashes, seed,
full configuration, synthetic-weight status, measured parameter count, and
compilation options. `transformers_source.json` and `workload.json` are written
before compiling, so a failure still has provenance. Successful phases go into
`decoder_phases.json`. `result.json` distinguishes passed and failed execution.

Numerical checks compare the compiled layer and KV cache with the same upstream
layer on CPU, compare cached CPU execution with a fresh full-prefix evaluation,
and require old cache entries to remain bit-exact. The existing BF16 tolerances
are unchanged: `rtol=0.02, atol=0.02`, applied elementwise. Timing mode requires
`--validate-timing`, then a complete dependency audit. A failed numerical check
must not become an accepted timing baseline. The explicit exploratory mode
below can collect timing without declaring the numerical check passed.

## Run in order

From the repository root, with `TMPDIR` set:

```bash
python3 -B -m unittest discover -s Simulator/experiments/qwen2_5_7b/tests/unit -v
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.transformers_decoder

# Full-width CPU reference, using the actual upstream decoder class.
bash Simulator/experiments/qwen2_5_7b/run.sh cpu --component decoder --seq-len 8 --decode-steps 2

# First select the BF16 toolchain using the toolchain guide.
bash Simulator/experiments/qwen2_5_7b/run.sh functional --component decoder --seq-len 8 --decode-steps 2
```

Only after functional correctness is established, select the corrected TOGSim
build and run:

```bash
QWEN_TIMEOUT_SECONDS=900 bash Simulator/experiments/qwen2_5_7b/run.sh timing --component decoder --seq-len 8 --decode-steps 2 --validate-timing
```

See [BF16 setup](toolchain.md) and the corrected TOGSim build instructions in
[the attention walkthrough](attention.md). The launcher defaults to a
300-second wall-time cap; the full layer may require longer than attention alone.
`analysis.trace_summary` supports decoder phase files for unit occupancy. The
historical `analysis.operator_timeline` intentionally rejects decoder runs:
its fixed attention labels cannot be reused for the new graph.
`analysis.kernel_inventory` instead matches each decoder wrapper call to the
launched MLIR and records compiler origins, dimensions, and per-kernel occupancy
without guessing semantic labels from kernel positions.

## Exploratory utilization with numerical differences

To study the generated schedule while retaining the known eager/compiled BF16
differences, explicitly opt into reporting finite tolerance mismatches:

```bash
# Select the BF16 toolchain and corrected TOGSim build as above.
QWEN_TIMEOUT_SECONDS=1200 bash Simulator/experiments/qwen2_5_7b/run.sh timing \
  --component decoder --seq-len 8 --decode-steps 2 \
  --validate-timing --allow-numerical-mismatch
```

This does not change Transformers, compiler transformations, or tolerances.
Spike still executes every submitted kernel and supplies real indirect indices.
Each failed phase preserves its comparison and output tensors under
`numerical/<phase>/`; `decoder_phases.json` records failed numerical status.
Shape, dtype, finite-value, CPU full-prefix, and exact old-cache-prefix checks
remain mandatory. Compiler/runtime errors are not numerical warnings.

A completed run with mismatches has status
`completed_with_numerical_mismatch`, never `passed`. Exit status zero means
requested execution completed (and, through the launcher, the dependency audit
passed), not numerical validation success. The timing dependency audit remains
mandatory. These are exploratory PyTorchSim queue-occupancy measurements, not
a validated native TPUv3/XLA or full-model inference baseline.

Analyze such a run with the matching explicit option:

```bash
QWEN_INPUT_RUN="$TMPDIR/qwen25-timing.<run>" \
  bash Simulator/experiments/qwen2_5_7b/run.sh toolchain bash -ec \
  'python -B -m Simulator.experiments.qwen2_5_7b.analysis.trace_summary \
    "$QWEN_INPUT_RUN" --output-dir "$TMPDIR/analysis" \
    --allow-numerical-mismatch --plot'
```

Analysis outputs go inside the *new container run's* `TMPDIR`.
Raw `[issue, finish)` intervals and simultaneous queue-state
cycle counts are preserved alongside the 5,000-cycle averaged plot.
The option does not classify differences as harmless. See the
[utilization report](utilization.md) for recorded numerical status. The original
run exposed a separate decode value-head expansion error; the
[frontend axis-lookup fix](value-expansion.md) now passes exact compiled
regressions, while smaller full-layer eager/compiled differences remain.

## Historical validation status (2026-09-05, before the expansion fix)

- 27 standard-library tests pass, including default workload selection,
  source-integrity rejection, the mandatory timing-validation flag, and legacy
  phase-file compatibility.
- Full-width CPU: batch 1, hidden 3584, MLP width 18944, 28 query heads,
  4 KV heads, head dimension 128, eight-token prefill plus two cached decode
  steps. Passed, including full-prefix/cache checks. The layer has 233,057,792
  parameters; all weights are synthetic BF16.
  Evidence: `/data2/s2chitni/.tmp/qwen25-cpu.Vx2knG/result.json`.
- Three small CPU integration tests pass. These verify exact upstream class
  identity, bit-exact agreement with the layer called *inside* `Qwen2Model`
  in BF16 and FP32, and cache mutation under full-graph Dynamo capture with the
  eager test backend. Small test dimensions are not performance proxies.
  Evidence: `/data2/s2chitni/.tmp/qwen25-toolchain.dBqVSH/console.log`.
- The full-width PyTorchSim functional run compiled the unmodified layer and
  executed prefill through Spike, but **failed numerical validation** after
  about 91 seconds. Hidden-state maximum absolute error was 0.0703125;
  742 of 28,672 elements failed the existing comparison. KV checks passed.
  Decode did not run. No decoder timing baseline has been accepted.
  Evidence: `/data2/s2chitni/.tmp/qwen25-functional.Z5UUAS/result.json` and
  `comparison_failure.json` in that directory.

Generated code in that failed run contains attention rewrites with FP32 score
computation. This is a diagnostic lead, **not an isolated explanation of the
failure**. A post-run CPU FP32 comparison also differs from BF16 eager execution;
it does not establish which compiler transformation accounts for the observed
error. No tolerance was relaxed, and no model or compiler fix was introduced to
hide the failure.

## Remaining stages

1. Review the remaining eager/compiled BF16 precision differences and the
   numerical acceptance contract, keeping the package model unchanged. The
   separate value-head data-corruption bug is fixed and has exact regressions.
2. Complete one-layer validated timing, phase coverage, and generated-operator
   attribution. Recheck precision and unit assignment after compiler rewrites.
3. Use upstream `Qwen2Model`/`Qwen2ForCausalLM` for the two-layer and full-model
   stages. Let the parent model own masks, positions, layer traversal, final
   normalization, embedding lookup and logits in the appropriate timing scope.
4. Separately decide when to load actual checkpoint weights and run real token
   generation. The current synthetic layer is not pretrained end-to-end inference.
