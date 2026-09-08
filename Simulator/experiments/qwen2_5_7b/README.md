# Qwen2.5-7B / single-core TPUv3 study

Start here for running and extending the study. The default workload now calls
the installed Transformers **`Qwen2DecoderLayer` directly**, with no custom model
forward or private cache manipulation. See [the Transformers baseline](docs/transformers.md).
Full-width CPU validation passes; the initial compiled layer fails the existing
numerical check, so decoder timing is **not yet validated**. The prior attention
diagnostic remains the latest validated timing workload. Full-model timing is
not implemented yet. All current model weights are synthetic BF16.

The [exploratory full-layer utilization experiment](docs/utilization.md) now
has a complete 93-kernel prefill/decode trace with passing timing-dependency
checks. Numerical validation still fails: decode exposed a separate value-head
expansion error, which is now [fixed in the frontend tiler](docs/value-expansion.md).
All nine exact expansion regressions pass. Smaller full-layer CPU tolerance
differences remain, so this is not yet an accepted model-performance baseline.

The [corrected B=1, S=128 rerun](docs/utilization-s128.md) completes with the
[fixed fork-backed gem5 installation](docs/gem5.md) and the permanent LLVM
[native TOG fix](docs/tog.md). All 64 timing sources and 93 kernel invocations
pass the attribution audit; prefill VPU queue occupancy is now 23.26%, not the
invalidated 74.11%. Dependency and counter checks pass too. The older trace is
preserved as historical evidence. Full-layer CPU tolerance differences remain
unresolved, so these corrected timing-accounting results are still exploratory;
a fresh S=8 rerun with this compiler remains pending.

## Source map

```text
qwen2_5_7b/
  run.sh                  Host/container entry point
  runner.py               CLI, dispatch and result manifests
  config.py               Model loading and backend setup
  provenance.py           Pinned Transformers version and source integrity
  validation.py           Shared numerical checks
  submission.py           Complete phase metadata without device barriers
  configs/                Pinned model dimensions and tile mappings
  workloads/              Direct upstream decoder; attention/component diagnostics
  analysis/               Memory inventory, dependency audit and activity traces
  tests/unit/             Fast tests requiring only the Python standard library
  tests/integration/      Frontend and compiled numerical regression suites
  tools/                  TOGSim, BF16 and gem5 build/install helpers
  docs/                   Walkthroughs, toolchain instructions and historical results
```

The main model-facing files are:

- [configs/model.json](configs/model.json): pinned Qwen dimensions and revision.
- [workloads/decoder.py](workloads/decoder.py): direct `Qwen2DecoderLayer` calls,
  upstream mask construction and public `DynamicCache` interface; no local model class.
- [workloads/attention.py](workloads/attention.py): `CachedAttention` wraps
  Transformers `Qwen2Attention`; `run_attention` runs prefill/cached decode.
- [workloads/components.py](workloads/components.py): isolated upstream RMSNorm,
  query projection and SwiGLU workloads.
- [workloads/cpu_reference.py](workloads/cpu_reference.py): reduced two-layer CPU
  correctness reference, explicitly not a full-width performance result.

Full architecture code is supplied by the container's pinned Transformers 4.43.4
package (`transformers.models.qwen2.modeling_qwen2`), not vendored into this study.
The new decoder checks its installed-package source hashes before executing.

## Run

Run these commands from the repository root. `TMPDIR` must point to scratch
storage. The launcher also works by absolute path from another working directory.

```bash
# Host-only checks: no container, PyTorch or checkpoint needed.
bash Simulator/experiments/qwen2_5_7b/run.sh audit --context-tokens 2048
python3 -B -m unittest discover -s Simulator/experiments/qwen2_5_7b/tests/unit -v

# New standard decoder workload: CPU first, then numerical simulator validation.
bash Simulator/experiments/qwen2_5_7b/run.sh cpu --component decoder --seq-len 8 --decode-steps 2
# Select the BF16 toolchain before running functional mode. Currently fails the numerical gate.
bash Simulator/experiments/qwen2_5_7b/run.sh functional --component decoder --seq-len 8 --decode-steps 2

# Historical attention-only diagnostic.
bash Simulator/experiments/qwen2_5_7b/run.sh cpu --component attention --seq-len 8 --decode-steps 2

# Select the BF16 toolchain and corrected TOGSim build first; see docs below.
bash Simulator/experiments/qwen2_5_7b/run.sh timing --component attention --seq-len 8 --decode-steps 2 --validate-timing
```

`--context-tokens` affects the separate memory inventory, not the timed attention
sequence. `--seq-len` and `--decode-steps` set that workload. The default component
is now `decoder`, including in CPU mode. Legacy CPU requests for isolated
`rmsnorm`, `q_proj`, or `mlp` still select the reduced-model CPU reference;
use the explicit decoder or attention component for real-width layer work.

The package entry point is `python -B -m Simulator.experiments.qwen2_5_7b`.
Container-only checks and analysis use `run.sh toolchain python -B -m MODULE ...`:

```bash
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.frontend_regression

QWEN_INPUT_RUN="$TMPDIR/qwen25-timing.gROSiL" \
  bash Simulator/experiments/qwen2_5_7b/run.sh toolchain bash -ec \
  'python -B -m Simulator.experiments.qwen2_5_7b.analysis.trace_summary "$QWEN_INPUT_RUN" --output-dir "$TMPDIR/trace_analysis" --plot
   python -B -m Simulator.experiments.qwen2_5_7b.analysis.operator_timeline "$QWEN_INPUT_RUN" "$TMPDIR/trace_analysis"'
```

The `attention_parts` component selects the operator integration checks in
`tests/integration/attention_components.py`; they are not mixed into the model
workload. Integration modules require the container; host test discovery should
target `tests/unit/` only. Analysis retains the prior recorded-run JSON/CSV schema.
Phase IDs are captured after a host-stream marker so the final asynchronous
kernel submission cannot be omitted; this does not add a simulated device barrier.

## Documentation

- [Attention execution and trace walkthrough](docs/attention.md)
- [Standard Transformers baseline and current validation status](docs/transformers.md)
- [Exploratory decoder utilization results and rerun](docs/utilization.md)
- [B=1, S=128 utilization graphs and historical stall evidence](docs/utilization-s128.md)
- [gem5 matrix-width fix, permanent fork installation and regression tests](docs/gem5.md)
- [LLVM mixed-width timing-region fix and compiler regression tests](docs/tog.md)
- [Decode value-expansion root cause, compiler fix and exact regressions](docs/value-expansion.md)
- [BF16 setup, compiled regression suites and known limitations](docs/toolchain.md)
- [Experiment contract, bring-up history and prior evidence](docs/history.md)
- [File migration guide](docs/layout.md)

The old `experiments/qwen2_5_7b/run.sh` forwards to this launcher for compatibility.
Direct Python script paths have moved: use the module entry points above. Existing
scratch runs and their provenance records are preserved, not rewritten.
