# Source-layout migration

The Qwen study moved from `experiments/qwen2_5_7b/` into
`Simulator/experiments/qwen2_5_7b/`. Only a README and forwarding launcher remain
at the old location. The simulator implementation and unrelated legacy
experiments are unchanged by this cleanup.

| Previously in the study root | Now |
|---|---|
| `baseline.py` | `runner.py`, `config.py`, `analysis/memory_audit.py`, `workloads/cpu_reference.py`, `workloads/components.py` |
| `attention_probe.py` | `workloads/attention.py`, `workloads/common.py`, `validation.py`, `tests/integration/attention_components.py` |
| `model.json`, `mappings/` | `configs/model.json`, `configs/mappings/` |
| `dependency_audit.py`, `trace_summary.py`, `operator_timeline.py` | `analysis/` (same basenames) |
| `test_audit.py`, other `test_*.py` | `tests/unit/` (`test_memory_audit.py` replaces `test_audit.py`) |
| `bf16_regression.py` | `tests/integration/frontend_regression.py` |
| `bf16_smoke.py` | `tests/integration/bf16_arithmetic.py` |
| `matrix_smoke.py` | `tests/integration/matrix_kernels.py` |
| `bmm_smoke.py` | `tests/integration/batched_matmul.py` |
| `lookup_smoke.py` | `tests/integration/indirect_lookup.py` |
| `build_togsim.py` | `tools/build_togsim.py` |
| `toolchain/` sources/build scripts | `tools/bf16/` |
| `ATTENTION.md`, toolchain README, historical root README | `docs/attention.md`, `docs/toolchain.md`, `docs/history.md` |

Use Python module entry points from the repository root, rather than executing
package-internal files directly. For example:

```bash
python3 -B -m Simulator.experiments.qwen2_5_7b --mode audit
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain python -B -m Simulator.experiments.qwen2_5_7b.tools.build_togsim --test-only
```

Launcher mode names, workload arguments, environment overrides, BF16 precision,
timing boundaries, and recorded-run schemas are preserved. Legacy result fields
such as `probe` and `probe_script_sha256` remain for compatibility; the latter
now fingerprints `runner.py`. New manifests also fingerprint the split workload
and helper modules. Historical manifests still refer to their original paths.

This is primarily a layout/import refactor, not a decoder-layer implementation
or a change to simulator scheduling. Verification also exposed an existing
phase-metadata race: the asynchronous host stream could still be submitting the
last kernel when its IDs were read. `submission.py` fences host submissions with
a marker before taking the snapshot, without adding a simulated `DEVICE_SYNC`.
The regression is checked both with a fake submission queue and the real
OpenReg stream (`tests.integration.submission`). Model math is unchanged.

Before this cleanup, the original study sources were
archived at `/data2/s2chitni/.tmp/qwen-layout-refactor.R9DHlt/before-cleanup.tar.gz`.

## Verification of the cleanup

- 20 standard-library unit tests pass, including entry-point compatibility,
  documentation links and the asynchronous phase-snapshot regression.
- The 12 relocated frontend/BF16 regression tests pass in the container:
  `/data2/s2chitni/.tmp/qwen25-toolchain.jY0Fi6/console.log`.
- The host-only phase fence passes against the actual OpenReg stream:
  `/data2/s2chitni/.tmp/qwen25-toolchain.6r3vws/console.log`.
- The full-width CPU attention reference passes prefill and both cached decode
  steps: `/data2/s2chitni/.tmp/qwen25-cpu.CMEZst/result.json`.
- The moved analysis module entry points regenerate all activity plots from the
  original corrected run:
  `/data2/s2chitni/.tmp/qwen25-toolchain.llkVPq/trace_analysis/`.
- Source comparison checks preserve the moved workload math, numerical checks,
  analysis logic, configuration and compiler sources. The intentional exception
  is the phase-metadata fence described above. The 28-item comparison report is
  `/data2/s2chitni/.tmp/qwen-layout-refactor.R9DHlt/refactor_audit.json`.

The fresh combined functional/timing check
`/data2/s2chitni/.tmp/qwen25-timing.UoJJuw` passed numerical validation for all
three phases, then exceeded its 300-second wall-time limit while TOGSim was
running. It is **not a completed combined run** and has no final `result.json`.
Its phase snapshots exposed the missing final IDs 19, 41 and 63 before the
metadata-fence fix; those historical snapshots were left unchanged.

All 64 kernels were present in the parser/launch journal. A separate replay of
those generated artifacts completed in 564,438 simulated cycles, preserved the
original partial run's 23,062 compute events exactly, and passed the complete
dependency audit. Evidence:
`/data2/s2chitni/.tmp/qwen25-toolchain.WabWae/recovered_timing_audit.json`.
The fence itself is tested separately; the replay does not execute model tensors.
This fresh-allocation timing is a verification result, not an isolated
performance comparison with the historical 559,658-cycle baseline.
