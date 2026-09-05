# Simulator experiments

Reproducible studies live here, separate from the simulator implementation.
Each study owns its launcher, workloads, configuration, analysis, tests and
documentation. Generated code, logs, plots and builds belong under `TMPDIR`,
never in this source tree.

- [Qwen2.5-7B on the TPUv3 model](qwen2_5_7b/README.md): full-width attention,
  cached prefill/decode, BF16 compatibility checks and unit-activity analysis.

The older generic examples and published artifact scripts remain in the
repository's top-level `experiments/` directory. They were not moved as part of
the Qwen study cleanup.
