import os
import re
import shlex
import subprocess
import torch

from PyTorchSimFrontend import extension_config
from torch._inductor.codecache import get_hash, write
from torch._inductor.async_compile import AsyncCompile
from AsmParser.tog_generator import tog_generator
from PyTorchSimFrontend.mlir.mlir_caller_codegen import MLIRKernelCallerCodeGen
from Simulator.simulator import FunctionalSimulator, CycleSimulator, TOGSimulator

# Configure logger for extension_codecache module (WARNING level by default)
logger = extension_config.setup_logger()

LOCK_TIMEOUT = 600

def hash_prefix(hash_value):
    return hash_value[1:12]

def get_write_path(src_code):
    return os.path.join(extension_config.CONFIG_TORCHSIM_DUMP_PATH, hash_prefix(get_hash(src_code.strip())))


def get_lock_path(write_path):
    """Return lock file path for the given write_path (per-source_code lock)."""
    return os.path.join(write_path, ".compile.lock")

def dump_metadata(args, arg_attributes, path):
    meta_path = os.path.join(path, "meta.txt")
    if os.path.isfile(meta_path):
        return

    with open(meta_path, "a") as file:
        for (arg_name, arg_attribute), arg in zip(arg_attributes, args):
            file.write(f'{arg_name}=({arg_attribute[0]}, {arg.dtype}, {arg.shape})\n')
    return

def bf16_legalization_passes(enabled):
    # Keep BF16 storage, but perform unsupported arithmetic in FP32.
    return "-arith-emulate-unsupported-floats='source-types=bf16 target-type=f32'" if enabled else ""


def bf16_expansion_pass(enabled):
    # Expand BF16 casts only AFTER indirect DMA lowering (and TOG extraction).
    # arith-expand also folds affine.apply. Our indirect-addressing marker uses
    # an otherwise-unused affine symbol; folding it early silently drops the
    # lookup indices. The DMA pass must consume that marker first.
    return "-arith-expand='include-bf16=true'" if enabled else ""


def bf16_matrix_pass(enabled, vectorlane_size, vlen):
    plugin = os.environ.get("TORCHSIM_BF16_PLUGIN")
    if not enabled or not plugin:
        return ""
    return (f"--load-pass-plugin={shlex.quote(plugin)} "
            f"-pytorchsim-bf16-to-vcix='systolic-array-size={vectorlane_size} vlen={vlen}'")


def bf16_llc_options(enabled):
    # The bundled gem5 decoder rejects integer-extension instructions at LMUL=8.
    # Apply the same conservative register-group cap to functional and timing
    # compilation; do not silently benchmark a different instruction stream.
    return "-riscv-v-fixed-length-vector-lmul-max=4" if enabled else ""


def split_bf16_plugin_stage(commands):
    """Plugin passes are registered too late for MLIR's individual CLI flags.

    Run the plugin with a textual pipeline between the existing preparation
    and lowering stages. Keep each process separate so failures propagate.
    """
    tokens = shlex.split(commands[0])
    positions = [i for i, token in enumerate(tokens) if token.startswith("--load-pass-plugin=")]
    if not positions:
        return commands
    index = positions[0]
    output_index = tokens.index("-o")
    source, output = tokens[output_index - 1], tokens[output_index + 1]
    before, after = output + ".bf16_pre.mlir", output + ".bf16_post.mlir"
    name, options = tokens[index + 1].lstrip("-").split("=", 1)
    # Expanding casts inside a named linalg.matmul region would violate that
    # operation's verifier. Lower matmul first, then legalize the remaining
    # BF16 vector arithmetic and casts.
    legalization = [token for token in tokens[:index] if token.startswith("-arith-")]
    # This fork's custom DMA assembly printer/parser cannot round-trip every
    # attribute combination. Generic operation syntax preserves them exactly.
    preparation = [token for token in tokens[:index] if token not in legalization] + ["--mlir-print-op-generic", source, "-o", before]
    plugin = [tokens[0], tokens[index],
              f"--pass-pipeline=builtin.module({name}{{{options}}})",
              "--mlir-print-op-generic", before, "-o", after]
    dialect_plugin = tokens[index].replace("--load-pass-plugin=", "--load-dialect-plugin=", 1)
    lowering = [tokens[0], dialect_plugin] + legalization + tokens[index + 2:]
    lowering[lowering.index("-o") - 1] = after
    return [shlex.join(stage) for stage in (preparation, plugin, lowering)] + commands[1:]


def add_bf16_memory_stage(commands, enabled):
    plugin = os.environ.get("TORCHSIM_BF16_PLUGIN")
    if not enabled or not plugin:
        return commands
    memory_plugin = os.path.join(os.path.dirname(plugin), "libPyTorchSimBF16Memory.so")
    result = []
    translated = legalized = None
    for command in commands:
        arguments = shlex.split(command)
        if translated:
            arguments = [legalized if argument == translated else argument for argument in arguments]
        result.append(shlex.join(arguments))
        if os.path.basename(arguments[0]) == "mlir-translate":
            translated = arguments[arguments.index("-o") + 1]
            legalized = translated + ".bf16_memory.ll"
            result.append(shlex.join([
                os.path.join(extension_config.CONFIG_TORCHSIM_LLVM_PATH, "opt"),
                f"--load-pass-plugin={memory_plugin}",
                "-passes=pytorchsim-bf16-memory,instcombine", "-S",
                translated, "-o", legalized,
            ]))
    return result


def mlir_compile_command(filename, vectorlane_size, vlen=256, uses_bf16=False):
    commands = [re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-opt \
            -test-loop-padding \
            -dma-fine-grained='systolic-array-size={vectorlane_size}' \
            -global-idx='vlen={vlen}' \
            {bf16_legalization_passes(uses_bf16)} \
            {bf16_matrix_pass(uses_bf16, vectorlane_size, vlen)} \
            -test-pytorchsim-to-vcix='systolic-array-size={vectorlane_size} vlen={vlen}' \
            -test-memref-to-gemmini="vectorlane={vectorlane_size}" \
            {bf16_expansion_pass(uses_bf16)} \
            -convert-linalg-to-loops \
            -convert-vector-to-scf='full-unroll' \
            -lower-affine \
            -finalize-memref-to-llvm \
            -lower-vector-multi-reduction \
            -convert-vector-to-llvm \
            -convert-arith-to-llvm \
            -convert-math-to-llvm \
            -convert-scf-to-cf \
            -convert-cf-to-llvm \
            -convert-func-to-llvm \
            -convert-index-to-llvm \
            -reconcile-unrealized-casts \
            {'--mlir-print-ir-after-all' if extension_config.CONFIG_TORCHSIM_DUMP_MLIR_IR else ''} \
            {filename}.mlir -o {filename}_llvm.mlir
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-translate -mlir-to-llvmir {filename}_llvm.mlir -o {filename}.ll
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/llc \
                {bf16_llc_options(uses_bf16)} \
                -relocation-model=pic -march=riscv64 -O3 --stack-size-section \
                -mattr=+m,+f,+d,+a,+c,+v,+zvfh,+xsfvcp,zvl{vlen}b \
                -filetype=obj \
                {'--print-after-all' if extension_config.CONFIG_TORCHSIM_DUMP_LLVM_IR else ''} \
                -O2 {filename}.ll -o {filename}.o
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/llc \
                {bf16_llc_options(uses_bf16)} \
                -relocation-model=pic -march=riscv64 -O3 --stack-size-section \
                -mattr=+m,+f,+d,+a,+c,+v,+zvfh,+xsfvcp,zvl{vlen}b \
                -O2 {filename}.ll -o {filename}.s
        """,
    ).strip()]
    return add_bf16_memory_stage(split_bf16_plugin_stage(commands), uses_bf16)

def mlir_gem5_compile_command(filename, sample_filename, tog_file, vectorlane_size, vlen=256, uses_bf16=False):
    commands = [re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-opt \
            -test-loop-padding='timing_mode=1' \
            -dma-fine-grained='systolic-array-size={vectorlane_size}' \
            -global-idx='vlen={vlen}' \
            {bf16_legalization_passes(uses_bf16)} \
            {bf16_matrix_pass(uses_bf16, vectorlane_size, vlen)} \
            -test-pytorchsim-to-vcix='systolic-array-size={vectorlane_size} vlen={vlen}' \
            -test-tile-operation-graph='vectorlane={vectorlane_size} sample-mode={extension_config.CONFIG_TLS_MODE}' \
            -test-memref-to-gemmini="vectorlane={vectorlane_size} timing=1" \
            {bf16_expansion_pass(uses_bf16)} \
            -convert-linalg-to-loops \
            -convert-vector-to-scf='full-unroll' \
            -lower-affine \
            -finalize-memref-to-llvm \
            -lower-vector-multi-reduction \
            -convert-vector-to-llvm \
            -convert-arith-to-llvm \
            -convert-math-to-llvm \
            -convert-scf-to-cf \
            -convert-cf-to-llvm \
            -convert-func-to-llvm \
            -convert-index-to-llvm \
            -reconcile-unrealized-casts \
            {'--mlir-print-ir-after-all' if extension_config.CONFIG_TORCHSIM_DUMP_MLIR_IR else ''} \
            {filename}.mlir -o {sample_filename}_llvm.mlir
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/mlir-translate -mlir-to-llvmir {sample_filename}_llvm.mlir -o {sample_filename}.ll
        """,
    ).strip(),
            re.sub(r"[ \n]+", " ",
        f"""
            {extension_config.CONFIG_TORCHSIM_LLVM_PATH}/llc \
                {bf16_llc_options(uses_bf16)} \
                -relocation-model=pic -march=riscv64 -O3 --stack-size-section \
                -mattr=+m,+f,+d,+a,+c,+v,+zvfh,+xsfvcp,zvl{vlen}b \
                -filetype=obj \
                {'--print-after-all' if extension_config.CONFIG_TORCHSIM_DUMP_LLVM_IR else ''} \
                -O2 {sample_filename}.ll -o {sample_filename}.o
        """,
    ).strip()]
    return add_bf16_memory_stage(split_bf16_plugin_stage(commands), uses_bf16)

class SpadOverflowError(Exception):
    def __init__(self, message="SPAD overflow occurred."):
        super().__init__(message)

class TileSizeError(Exception):
    def __init__(self, message="SPAD overflow occurred."):
        super().__init__(message)

class MLIRCodeCache:
    cache = dict()
    clear = staticmethod(cache.clear)   # Todo: Cache

    @staticmethod
    def _load_library(path):
        pass

    @classmethod
    def load(cls, source_code,
             validation_wrapper_name="validation_wrapper",
             validation_binary_name="validation_bin",
             cycle_wrapper_name="cycle_wrapper",
             cycle_binary_name="cycle_bin",
             arg_attributes=[], vectorlane_size=16,
             spad_info=None, origins=None, silent_mode=False, **kwargs):
        vlen = kwargs['vlen']
        vlenb = vlen // 8
        write_path = get_write_path(source_code)
        key, input_path = write(source_code, "mlir", specified_dir=write_path)
        new_input_path = os.path.splitext(input_path)[0]
        raw_tog_path = new_input_path + "_tog.py"
        tog_path = os.path.join(write_path, "tile_graph.onnx")
        sample_mlir_path = new_input_path + "_sample"
        validation_binary_path = os.path.join(write_path, validation_binary_name)
        # Shaped types spell this as "...xbf16", so a leading word boundary
        # would miss kernels that have no scalar BF16 constants.
        uses_bf16 = "bf16" in source_code
        if re.search(r"linalg\.matmul\s+ins\([^)]*bf16", source_code):
            if not os.environ.get("TORCHSIM_BF16_PLUGIN"):
                raise RuntimeError(
                    "BF16 matrix operations require TORCHSIM_BF16_PLUGIN and the "
                    "patched Spike for functional execution. See "
                    "Simulator/experiments/qwen2_5_7b/docs/toolchain.md."
                )
        gem5_cmds = mlir_gem5_compile_command(new_input_path, sample_mlir_path, raw_tog_path, vectorlane_size, vlen=vlen, uses_bf16=uses_bf16)

        from filelock import FileLock
        os.makedirs(write_path, exist_ok=True)
        lock = FileLock(get_lock_path(write_path), timeout=LOCK_TIMEOUT)

        if spad_info is not None:
            link_option = f"-Wl,--section-start=.spad=0x{spad_info['spad_vaddr']:x}"
        else:
            link_option = ""
        # Generate LLVM kernel calller and binary for validation
        if extension_config.pytorchsim_functional_mode:
            # Use custom malloc to avoid size error
            new_link_option = link_option + " -Wl,--wrap=malloc -Wl,--wrap=free"
            cmds = mlir_compile_command(new_input_path, vectorlane_size, vlen=vlen, uses_bf16=uses_bf16)
            with lock:
                try:
                    for command in cmds:
                        subprocess.check_call(shlex.split(command))
                except subprocess.CalledProcessError as e:
                    logger.error(f"Command failed with exit code {e.returncode}")
                    logger.error(f"Error output: {e.output.decode() if isinstance(e.output, bytes) else e.output}")
                    assert(0)

                val_llvm_caller = MLIRKernelCallerCodeGen(extension_config.pytorchsim_functional_mode, arg_attributes)
                val_llvm_caller.generate_wrapper_file(write_path, validation_wrapper_name)
                val_llvm_caller.compile_wih_kernel(write_path, key, validation_wrapper_name,
                                                   validation_binary_name, new_link_option)

                stack_size = val_llvm_caller.parse_stack_sizes(f"{write_path}/{key}.s", vlenb=vlenb)
                spad_size =  val_llvm_caller.get_spad_size(validation_binary_path)
                spad_usage = stack_size + spad_size # Spad usage per lane
                if extension_config.CONFIG_SPAD_INFO["spad_size"] < spad_usage:
                    logger.debug(
                        f"Scratchpad size exceeded: required {spad_usage} bytes, "
                        f"but only {extension_config.CONFIG_SPAD_INFO['spad_size']} bytes available."
                    )
                    raise SpadOverflowError()

        # Skip if TOG file already exists
        if os.path.isfile(tog_path):
            return key

        # Launch tile graph generator
        lock = FileLock(get_lock_path(write_path), timeout=LOCK_TIMEOUT)
        with lock:
            try:
                for command in gem5_cmds:
                    arguments = shlex.split(command)
                    if any(arg.startswith("-test-tile-operation-graph=") for arg in arguments):
                        result = subprocess.check_output(arguments)
                        with open(raw_tog_path, "wb") as file:
                            file.write(result)
                    else:
                        subprocess.check_call(arguments)
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with exit code {e.returncode}")
                logger.error(f"Error output: {e.output.decode() if isinstance(e.output, bytes) else e.output}")
                assert(0)

            if not extension_config.pytorchsim_timing_mode:
                return key

            # Generate MLIR kernel calller and binary for cycle calculation
            cycle_llvm_caller = MLIRKernelCallerCodeGen(False, arg_attributes, cycle_sim=True)
            cycle_llvm_caller.generate_wrapper_file(write_path, cycle_wrapper_name)
            cycle_llvm_caller.compile_wih_kernel(write_path, key + "_sample", cycle_wrapper_name, cycle_binary_name, link_option)

            # Run cyclesim
            cyclesim = CycleSimulator()
            cycle_list = cyclesim.compile_and_simulate(os.path.join(write_path, cycle_binary_name), vectorlane_size, silent_mode=silent_mode)

            # Create TOG
            w_offset, x_offset = vectorlane_size, vectorlane_size
            if kwargs['loop_size'] is not None and kwargs['loop_size'][-3] < vectorlane_size:
                x_offset = kwargs['loop_size'][-3]
            if kwargs['loop_size'] is not None and kwargs['loop_size'][-1] < vectorlane_size:
                w_offset = kwargs['loop_size'][-1]
            w_offset = 0 # max(w_offset - x_offset, 0)
            tile_graph_generator = tog_generator(origins)
            tile_graph_generator.load_file(raw_tog_path)
            tile_graph_generator.generate_tile_graph(
                tog_path,
                cycle_list=cycle_list,
                x_offset=x_offset, # FIXME.
                w_offset=w_offset, # FIXME.
                vector_lane=vectorlane_size
            )
        return key

def check_indirect_timing_supported(source_code, functional, timing, autotune):
    # TOGSim reads per-invocation index files emitted by Spike. Without them it
    # only warns and falls back to direct addresses, yielding misleading timing.
    if "indirect_access" in source_code and timing and (not functional or autotune):
        raise RuntimeError(
            "Indirect-address timing requires a functional Spike run to produce "
            "the actual index trace; timing-only/autotune execution is unsafe. "
            "Enable functional + timing modes and use heuristic mapping. The "
            "Qwen attention probe provides --validate-timing for this workflow."
        )


class CustomAsyncCompile(AsyncCompile):
    def __init__(self):
        self.validation_wrapper_name = "validation_wrapper"
        self.validation_binary_name = "validation_binary"
        self.cycle_wrapper_name = "cycle_wrapper"
        self.cycle_binary_name = "cycle_binary"

    def mlir(self, source_code, arg_attributes=[], vectorlane_size=16, tile_size=[], spad_info=None, origins=None, silent_mode=False, **kwargs):
        autotune = kwargs.get('autotune', False)
        check_indirect_timing_supported(source_code, extension_config.pytorchsim_functional_mode,
                                        extension_config.pytorchsim_timing_mode, autotune)
        def task():
            key = MLIRCodeCache.load(source_code,
                                          valdiation_wrapper_name=self.validation_binary_name,
                                          validation_binary_name=self.validation_binary_name,
                                          arg_attributes=arg_attributes, vectorlane_size=vectorlane_size,
                                          tile_size=tile_size, spad_info=spad_info, origins=origins,
                                          silent_mode=autotune, **kwargs)
            return key
        future = self.submit(task)

        def run_kernel_simulation(*args, autotune_subprocess_timeout_sec=None, **kwargs):
            # Wait for compilation
            key = future.result()
            from filelock import FileLock
            result_path = os.path.join(extension_config.CONFIG_TORCHSIM_DUMP_PATH, hash_prefix(key))
            lock = FileLock(get_lock_path(result_path), timeout=LOCK_TIMEOUT)
            with lock:
                # Run simulator pass
                # Dump arguments and meta data
                dump_metadata(args, arg_attributes, result_path)
                runtime_path = FunctionalSimulator.get_runtime_dump_path(result_path)
                if extension_config.pytorchsim_functional_mode and not autotune:
                    funcsim = FunctionalSimulator(result_path, key)
                    funcsim.run_spike(args, arg_attributes,
                                    runtime_path, self.validation_binary_name,
                                    vectorlane_size=vectorlane_size, spad_info=spad_info,
                                    silent_mode=autotune)

                if not extension_config.pytorchsim_timing_mode:
                    return [float("inf")]

                # Prepare arguments for launch kernel
                onnx_path = os.path.join(result_path, "tile_graph.onnx")
                attribute_dir = os.path.join(runtime_path, "attribute")
                kernel_attribute_path = TOGSimulator.write_kernel_attribute_file(attribute_dir, args)

                TOGSim = torch.npu.get_tog_simulator()
                if not autotune and TOGSim is not None:
                    torch.npu.launch_kernel(onnx_path, kernel_attribute_path)
                    result = None # No result for non-autotune mode
                else:
                    result_path = TOGSimulator.run_standalone(
                        onnx_path,
                        kernel_attribute_path,
                        autotune_mode=autotune,
                        timeout_sec=autotune_subprocess_timeout_sec,
                    )
                    result = TOGSimulator.get_result_from_file(result_path)
                return result
        return run_kernel_simulation
