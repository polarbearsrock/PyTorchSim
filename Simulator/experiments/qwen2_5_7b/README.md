# Qwen2.5-7B / single-core TPUv3 study

Start here for running and extending the study. We use upstream Transformers
model components, not a handwritten Qwen implementation. The current continuous
timing workload is **one real-width attention module**, with synthetic BF16
weights, eight-token prefill and two cached decode steps. A complete decoder
layer and full-model timing are not implemented yet.

## Source map

```text
qwen2_5_7b/
  run.sh                  Host/container entry point
  runner.py               CLI, dispatch and result manifests
  config.py               Model loading and backend setup
  validation.py           Shared numerical checks
  submission.py           Complete phase metadata without device barriers
  configs/                Pinned model dimensions and tile mappings
  workloads/              Upstream attention, component and CPU workloads
  analysis/               Memory inventory, dependency audit and activity traces
  tests/unit/             Fast tests requiring only the Python standard library
  tests/integration/      Frontend and compiled numerical regression suites
  tools/                  Offline TOGSim build helper and BF16 toolchain sources
  docs/                   Walkthroughs, toolchain instructions and historical results
```

The main model-facing files are:

- [configs/model.json](configs/model.json): pinned Qwen dimensions and revision.
- [workloads/attention.py](workloads/attention.py): `CachedAttention` wraps
  Transformers `Qwen2Attention`; `run_attention` runs prefill/cached decode.
- [workloads/components.py](workloads/components.py): isolated upstream RMSNorm,
  query projection and SwiGLU workloads.
- [workloads/cpu_reference.py](workloads/cpu_reference.py): reduced two-layer CPU
  correctness reference, explicitly not a full-width performance result.

The next workload should wrap upstream `Qwen2DecoderLayer`, preserving its norms,
attention, MLP and residuals. Full architecture code is supplied by the container's
Transformers 4.43.4 package (`transformers.models.qwen2.modeling_qwen2`), not vendored
into this study.

## Run

Run these commands from the repository root. `TMPDIR` must point to scratch
storage. The launcher also works by absolute path from another working directory.

```bash
# Host-only checks: no container, PyTorch or checkpoint needed.
bash Simulator/experiments/qwen2_5_7b/run.sh audit --context-tokens 2048
python3 -B -m unittest discover -s Simulator/experiments/qwen2_5_7b/tests/unit -v

# CPU attention reference at the same dimensions as the timing experiment.
bash Simulator/experiments/qwen2_5_7b/run.sh cpu --component attention --seq-len 8 --decode-steps 2

# Select the BF16 toolchain and corrected TOGSim build first; see docs below.
bash Simulator/experiments/qwen2_5_7b/run.sh timing --component attention --seq-len 8 --decode-steps 2 --validate-timing
```

`--context-tokens` affects the separate memory inventory, not the timed attention
sequence. `--seq-len` and `--decode-steps` set that workload. `cpu` without an
attention component retains the existing reduced-model CPU-reference behavior.

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
- [BF16 setup, compiled regression suites and known limitations](docs/toolchain.md)
- [Experiment contract, bring-up history and prior evidence](docs/history.md)
- [File migration guide](docs/layout.md)

The old `experiments/qwen2_5_7b/run.sh` forwards to this launcher for compatibility.
Direct Python script paths have moved: use the module entry points above. Existing
scratch runs and their provenance records are preserved, not rewritten.
