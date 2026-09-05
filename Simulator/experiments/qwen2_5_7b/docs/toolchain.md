# BF16 toolchain bring-up

This is an **experimental, opt-in compiler/simulator extension**, not a TPU XLA
backend or a claim of complete Qwen support. It keeps weights and ordinary
activations in BF16 memory, uses FP32 matrix partial sums, and makes the tested
Qwen components executable in PyTorchSim.

## Validated results (2026-09-04)

| Check | Result |
|---|---|
| Frontend/raw-I/O/pipeline unit tests | 12 passed |
| Compiled BF16 conversion/arithmetic cases | 7 passed, including negation, all 65,536 input bit patterns, and rounding edge cases |
| Small matrix suites | Four BF16, four legacy FP16, and four FP32 cases passed |
| Batched matrix suites | Eight BF16 cases; five legacy FP16 and five FP32 controls passed |
| Exact dynamic lookup | Offset, nonmonotonic/repeated positions, and decode passed in BF16 and FP32 |
| Qwen RMSNorm, `[1,16,3584]` | Passed; maximum absolute error 0 |
| Qwen biased query projection, 16 tokens | Passed; maximum absolute error 0.00390625 |
| Qwen MLP/SwiGLU, one token, real widths | Passed; maximum absolute error 0.0009765625 |
| Query-projection timing, one token | gem5 and TOGSim completed; latest run reported 59,295 simulated cycles |
| Numerical replay of the selected timing tile | Passed with zero error; input MLIR is byte-identical to the timing run's input MLIR |
| Real-width attention, eight-token prefill plus two cached decode steps | Numerical checks and inter-kernel dependency audit passed with patched TOGSim; 559,658 cycles. The original 543,101-cycle schedule is invalid. |

The [attention walkthrough](attention.md) explains the additional fixes,
numerical checks, and reconstructed VPU/MXU timeline. Projection timing below
is earlier evidence; the compiler has since gained the attention fixes.

The measured projection mapping is `(TILE_M,TILE_N,TILE_K)=(8,896,3584)`,
with matching subtile sizes; `M=1` is padded within the tile. It is recorded in
`../configs/mappings/q_proj_decode_bf16.json`. The two completed autotuned timing runs
reported 59,294 and 59,295 cycles; bit-exact timing repeatability has not been
established. These are compatibility checks for one projection, not model
latency or a native TPU utilization measurement.

Local evidence (all scratch artifacts):

- Regression suite: `/data2/s2chitni/.tmp/qwen25-toolchain.DHTkmz/console.log`
- RMSNorm: `/data2/s2chitni/.tmp/qwen25-functional.vh1Nsy/result.json`
- Projection, 16 tokens: `/data2/s2chitni/.tmp/qwen25-functional.MJBgvC/result.json`
- MLP: `/data2/s2chitni/.tmp/qwen25-functional.rktxMl/result.json`
- Projection timing: `/data2/s2chitni/.tmp/qwen25-timing.4pq1Ok/result.json`
- Timing event log: `/data2/s2chitni/.tmp/qwen25-timing.4pq1Ok/console.log`
- Selected-tile numerical replay: `/data2/s2chitni/.tmp/qwen25-functional.Uhup9o/result.json`
- Latest unit/GEMM/BMM regression: `/data2/s2chitni/.tmp/qwen25-toolchain.GhAEnp/console.log`
- Lookup/cast and legacy dtype controls: `/data2/s2chitni/.tmp/qwen25-toolchain.QSOJKV/console.log`
- Attention functional: `/data2/s2chitni/.tmp/qwen25-functional.t2Q6n4/result.json`
- Attention functional + timing: `/data2/s2chitni/.tmp/qwen25-timing.GfWORi/result.json`

## What was fixed

| Layer | Failure | Resolution |
|---|---|---|
| PyTorchSim frontend | Types beginning with `b` were not recognized as floating point | Explicit BF16-aware type classification, arithmetic, constants, and casts |
| Functional output | BF16 raw words were decoded as IEEE FP16 | Read as `uint16`, then reinterpret as `torch.bfloat16`; no numeric FP16 conversion |
| Vector code generation | BF16 arithmetic/casts required unsupported instructions or runtime helpers | MLIR arithmetic emulation in FP32 plus bitwise BF16 conversion expansion |
| Spike vector execution | Integer widening wrote the wrong vector lane | Build a pinned revision containing the upstream `VI_VV_EXT` lane fix |
| Matrix lowering | VCIX could not accept BF16 vectors, and output partial sums used the output dtype | BF16-only MLIR plugin: transport input bits as `i16`, pop FP32 results into an FP32 accumulator tile |
| Spike matrix execution | Every 16-bit input was interpreted as FP16 | Explicit BF16 selector in the custom matrix-feed instruction; preserve the old FP16 selector |
| Masked memory | Masked BF16 loads became calls to missing `__truncsfbf2` | LLVM plugin transports BF16 loads/stores, including masked operations, as integer bits |
| gem5 decoder | Integer widening at LMUL=8 panicked | Cap fixed-length BF16 vector groups at LMUL=4 in **both** functional and timing compilation |

The Spike base is
[`fb28e7aa5ea26e637bf104541d65a04f8fd0847e`](https://github.com/PSAL-POSTECH/riscv-isa-sim/commit/fb28e7aa5ea26e637bf104541d65a04f8fd0847e).
It contains the
[vector-lane fix](https://github.com/PSAL-POSTECH/riscv-isa-sim/commit/0422d112d83949f0aa429110979e62e2f1bfe42d)
and still supports this image's legacy DMA instruction interface. Later
revisions replace that interface; updating to the branch head is not safe here.

`BF16ToVCIX.cpp` derives its tiling/DMA handling from the upstream LLVM fork at
`970a927190e8402348cadf9585bf134c8a8c09c2`. Both plugins build against the actual
LLVM 20 headers installed in the existing container, avoiding a full LLVM rebuild.
Neither the container image nor its bundled LLVM/gem5 executables is overwritten.

## Build and select the toolchain

From the repository root:

```bash
bash Simulator/experiments/qwen2_5_7b/tools/bf16/build.sh
```

The script downloads the pinned Spike source, applies `spike-bf16.patch`, builds
with eight jobs, installs in a new directory under `$TMPDIR`, and builds the two
compiler plugins. It prints these exports with the generated directory filled in:

```bash
export QWEN_TOOLCHAIN_ROOT="$TMPDIR/qwen-bf16-toolchain.<printed-suffix>"
export TORCHSIM_SPIKE=/workspace/bf16-toolchain/install/bin/spike
export TORCHSIM_BF16_PLUGIN=/workspace/bf16-toolchain/libPyTorchSimBF16.so
```

The latter two paths are **container paths**. The launcher mounts
`QWEN_TOOLCHAIN_ROOT` there. Keep both `libPyTorchSimBF16.so` and
`libPyTorchSimBF16Memory.so` together. Use a fresh run/cache when changing either
plugin. The launcher already gives every invocation an isolated cache.

For this workspace, a verified build is currently available at
`/data2/s2chitni/.tmp/qwen-bf16-toolchain.7o13Ab`. Scratch directories are not
portable dependencies; the build script is the reproducible entry point.

To rebuild only the plugins in an existing selected toolchain, then refresh
the recorded checksums:

```bash
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain bash Simulator/experiments/qwen2_5_7b/tools/bf16/build_in_container.sh --plugins-only
```

## Validate in order

With those exports set:

```bash
# Frontend, raw I/O, compiler stage order, and BF16-only option isolation.
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.frontend_regression

# Compiled conversions/arithmetic, rounding ties, and all 65,536 BF16 bit patterns.
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.bf16_arithmetic

# GEMM, bias, decode shape, and an adversarial three-K-tile accumulation case.
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.matrix_kernels

# Batched products, narrow/short tiles, and exact dynamic table lookup.
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.batched_matmul
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.indirect_lookup

# Existing dtype controls.
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.matrix_kernels --dtype float32
bash Simulator/experiments/qwen2_5_7b/run.sh toolchain python -B -m Simulator.experiments.qwen2_5_7b.tests.integration.matrix_kernels --dtype float16

# Real Qwen widths, synthetic weights: numerical validation first.
bash Simulator/experiments/qwen2_5_7b/run.sh functional --component rmsnorm --seq-len 16
bash Simulator/experiments/qwen2_5_7b/run.sh functional --component q_proj --seq-len 16
bash Simulator/experiments/qwen2_5_7b/run.sh functional --component mlp --seq-len 1

# Separate timing validation; timing-only tensor values are not valid outputs.
bash Simulator/experiments/qwen2_5_7b/run.sh timing --component q_proj --seq-len 1

# Replay the selected decode timing tile through the numerical checker.
bash Simulator/experiments/qwen2_5_7b/run.sh functional --component q_proj --seq-len 1 --mapping-file Simulator/experiments/qwen2_5_7b/configs/mappings/q_proj_decode_bf16.json
```

Each invocation reports a directory containing logs and generated IR. Component
runs also write `result.json`, including selected binary/plugin hashes and
frontend source hashes. The build records `build-sha256.txt`.

The cancellation test computes `256 + 1 - 256` across three forced K tiles.
Its result must be exactly `1`; rounding the running sum to BF16 can lose the
middle contribution and produce `0`. The FP16 control deliberately checks the
legacy FP16 partial-sum behavior, which is not the new BF16/FP32 contract.

## How a BF16 matrix kernel executes

1. DMA copies BF16 inputs into BF16 scratchpad buffers: two bytes per element.
2. Matrix-feed vectors are bitcast to `i16`. Custom `iVpush`/`wVpush` use their
   immediate `rs1=1` to identify BF16. `rs1=0` retains the old FP16 behavior.
3. Patched Spike interprets BF16 bits by shifting them into FP32's upper 16 bits.
   Its existing systolic model performs the matrix arithmetic in host `float`.
4. Matrix pops use FP32 vectors. Partial results are added to an **FP32
   scratchpad accumulator**, retained across every K tile. Tile selection budgets
   this extra storage; it does not create a full FP32 weight copy in HBM.
5. Bias is loaded synchronously after the K loop, widened, and added in FP32.
   The final matrix result is rounded to BF16 once, then passed to the normal
   epilogue/output path. Other BF16 operators keep their own dtype boundaries.

The compiler order is preparation/global indexing, BF16 matrix lowering,
BF16 arithmetic emulation, VCIX/TOG extraction and DMA lowering, BF16 cast
expansion, remaining LLVM lowering/translation, BF16 memory legalization,
and RISC-V machine-code generation. Cast expansion must follow indirect DMA
lowering so its affine folding cannot erase lookup-index markers. The standalone
MLIR plugin uses a textual pass pipeline. Intermediate files use generic MLIR
syntax because this fork's custom DMA assembly does not always round-trip.

gem5 still supplies timing for the generated instruction stream. The existing
matrix decoder identifies the feed/pop operations independently of the new
BF16 immediate selector; its matrix model is not the numeric reference.
Spike is the numeric checker. TOGSim schedules tile operations and memory traffic.

## Scope and interpretation

- BF16 GEMM/BMM reduction fusion is disabled until its transposed accumulator
  path is implemented. Ordinary BMM now uses the FP32 accumulator path.
- Real-width attention/RoPE/GQA/softmax and cache updates pass the tested short
  sequence. Complete layers and full-model inference remain unvalidated.
- The LMUL cap, software BF16 conversions, FP32 scratchpad partial sums, and
  synchronous bias load are visible code-generation choices. They can affect
  the VPU/MXU timeline; this is not proof of native TPUv3/XLA instruction timing.
- A functional success does not validate every autotuned timing tile. Selected
  mappings need cross-checking before drawing model-wide conclusions. The
  recorded query-projection decode mapping has a dedicated replay command above.
- Native utilization counters are not cycle-by-cycle occupancy. Reconcile
  issue/finish intervals with counters before interpreting unit overlap or gaps.
- Neither complete-model fit in 16 GiB nor end-to-end BF16 inference is established.
