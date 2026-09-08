# Source-built LLVM for PyTorchSim

The mixed-width matrix timing fix is built into the **normal**
`test-tile-operation-graph` pass. There is no TOG pass plugin or renamed pass.
The compiler's VCIX dialect is also registered directly in `mlir-opt`, so
generated VCIX IR can be parsed without a dialect-registration workaround.

## This workspace

- Fork: [polarbearsrock/psal-postech-llvm-project](https://github.com/polarbearsrock/psal-postech-llvm-project), forked from [PSAL-POSTECH/llvm-project](https://github.com/PSAL-POSTECH/llvm-project). The account's unrelated `llvm-project` fork was not changed.
- Source: `/data2/s2chitni/psal-postech/llvm-project`.
- Branch: `shwet/qwen-performance-tpuv3`; fix commit `e528d7b07`.
- Base: v1.0.8, commit `696476d3d064dd4a75f34ffddda42bfa8143a398`, matching the original simulator release.
- Installation: `/data2/s2chitni/psal-postech/llvm-project/install`.
- Build scratch: `/data2/s2chitni/.tmp/pytorchsim-llvm-build.MePB6b`.

The actual source changes are in the LLVM fork:

- [Matrix timing-region logic](/data2/s2chitni/psal-postech/llvm-project/mlir/test/lib/Analysis/TestTileOperationGraph.cpp).
- [Native VCIX registration](/data2/s2chitni/psal-postech/llvm-project/mlir/tools/mlir-opt/mlir-opt.cpp).
- [Compiler regression](/data2/s2chitni/psal-postech/llvm-project/mlir/test/Analysis/tile-operation-graph-mixed-width.mlir) and its [fixtures/checker](/data2/s2chitni/psal-postech/llvm-project/mlir/test/Analysis/Inputs/tog-mixed-width.py).

The pass counts emitted matrix result reads rather than assuming their count
matches the input pushes. All result reads and corresponding accumulator writes
stay in `MatmulCompute`; the following vector epilogue stays in `VectorCompute`.
The regression covers mixed/equal widths, short/padded vectors, adjacent matrix
segments, both sampling modes, graph/marker agreement, and orphan-read rejection.

## Default selection

This checkout's ignored, machine-local `configs/toolchains.local.json` contains:

```json
{
  "llvm_install": "/data2/s2chitni/psal-postech/llvm-project/install"
}
```

The Qwen launcher reads this configuration, checks installed binary hashes,
mounts the installation read-only at the same path in the container, and sets
`TORCHSIM_LLVM_PATH` to its `bin` directory. **No LLVM export is required for
ordinary runs from this checkout.** The unchanged container compiler remains
at `/riscv-llvm` for negative-control tests; it is not selected for these runs.

To explicitly select another built installation, set `TORCHSIM_LLVM_ROOT`.
For direct frontend use, `TORCHSIM_LLVM_PATH` is still supported; an explicit
root takes priority. An explicitly empty root disables the local default for
diagnostics. BF16 timing independently rejects a compiler that lacks the
built-in mixed-width capability, rather than silently using the old pass.

The existing BF16 arithmetic/matrix-lowering plugins and Spike selection are
separate from this TOG fix and remain as before. Their build script now uses
headers from the selected LLVM installation. The end-to-end checks below passed
with the existing BF16 plugin binaries against the new compiler.

## Rebuild and install

Use an already-patched source checkout. All objects, staging files, caches, and
logs stay under TMPDIR. The build uses the existing PyTorchSim container,
Release mode, the RISCV target, static LLVM libraries, and assertions/RTTI off,
matching the relevant settings of the original compiler.

```bash
bash scripts/toolchains/llvm/build.sh /data2/s2chitni/psal-postech/llvm-project
```

The default is 48 compile jobs and two concurrent links; set `LLVM_BUILD_JOBS`
to adjust it. Set `PYTORCHSIM_IMAGE` to use another compatible container image.
The native compiler regression must pass before a staged installation is
recorded. To resume a build, pass its printed scratch directory as argument two.

Install into a **new** prefix; the installer refuses to overwrite an existing
compiler. Update the local configuration to select that new prefix afterward.

```bash
bash scripts/toolchains/llvm/install.sh \
  "$TMPDIR/pytorchsim-llvm-build.PRINTED_SUFFIX" \
  /data2/s2chitni/psal-postech/llvm-project \
  /data2/s2chitni/psal-postech/llvm-project/install-next
```

`share/pytorchsim/llvm-build.json` records source revision/branch, changed-source
fingerprints, build settings, and hashes of the actual compiler tools. Source
edits require a rebuild; editing the checkout does not update its installed tools.
Verify both source and binaries with:

```bash
python3 -B scripts/toolchains/llvm/record_build.py verify \
  /data2/s2chitni/psal-postech/llvm-project/install \
  --source /data2/s2chitni/psal-postech/llvm-project
```

The verified runtime is the existing Ubuntu simulator container; host-library
ABI compatibility is not assumed. This local installation does not publish a
new compiler release or change the shared Docker release pins.

## Verification evidence

- 48 host unit tests passed, including local-selection, capability, source/binary-fingerprint, and no-overwrite controls.
- Native LLVM regression passed using the built-in pass, with no TOG/dialect plugin.
- [16 frontend tests and 79 compiler checks](/data2/s2chitni/.tmp/qwen25-toolchain.Gq7CIp/console.log) passed, including all 59 saved unique Qwen kernel IR files and the old compiler's expected regression failure.
- [Four end-to-end GEMMs](/data2/s2chitni/.tmp/qwen25-toolchain.aDdsPT/matrix-timing-result.json) passed through the installed compiler, fixed gem5, Spike, and TOGSim: BF16 M=8/128 plus FP16/FP32 M=128, all N=128, K=384, with exact cancellation across three K tiles.
- These tests check both graph and LLVM timing-marker coverage, positive gem5 region latencies, simulator completion, and bit-exact outputs.
- [Existing BF16 arithmetic and 12 functional GEMM cases](/data2/s2chitni/.tmp/qwen25-toolchain.OQPq2b/console.log) passed with the installed compiler, including FP16/FP32 controls.

The [fresh Qwen S=128 utilization run](../../../Simulator/experiments/qwen2_5_7b/docs/utilization-s128.md)
completed with this compiler. Attribution checks pass for all 64 timing sources
and 93 kernel invocations; the previous trace remains invalidated. Full-layer
numerical validation still fails, so the corrected timing results remain
exploratory rather than an accepted model-performance baseline.
