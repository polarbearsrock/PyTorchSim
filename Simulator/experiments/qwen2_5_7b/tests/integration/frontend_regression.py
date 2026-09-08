"""Fast BF16 frontend/readback tests, run inside the simulator environment."""

import os
import shlex
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, os.environ["TORCHSIM_DIR"])

import numpy as np
import sympy
import torch
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch._inductor.virtualized import V
from torch.utils._sympy.value_ranges import ValueRanges

from PyTorchSimFrontend.mlir.mlir_ops import ExtensionOverrides as Ops
from PyTorchSimFrontend.mlir.mlir_common import BaseMLIRKernel, MLIR_INF, MLIRMultiDimTile
from PyTorchSimFrontend.extension_codecache import mlir_compile_command, mlir_gem5_compile_command, check_indirect_timing_supported
from Simulator.simulator import FunctionalSimulator


class Symbol(str):
    def __new__(cls, name):
        result = super().__new__(cls, name)
        result.bounds = ValueRanges.unknown()
        return result


class FrontendTests(unittest.TestCase):
    def test_indirect_timing_requires_real_index_trace(self):
        for functional, timing, autotune in ((False, True, False), (True, True, True)):
            with self.assertRaisesRegex(RuntimeError, "actual index trace"):
                check_indirect_timing_supported("{indirect_access}", functional, timing, autotune)
        for functional, timing, autotune in ((True, False, False), (True, True, False)):
            check_indirect_timing_supported("{indirect_access}", functional, timing, autotune)
        check_indirect_timing_supported("ordinary kernel", False, True, True)

    def test_float_conversions(self):
        for lanes in (1, 8):
            for source, dest, opcode in (("bf16", "f32", "extf"),
                                         ("f32", "bf16", "truncf"),
                                         ("bf16", "f64", "extf")):
                with self.subTest(lanes=lanes, source=source, dest=dest):
                    with V.set_kernel_handler(SimpleNamespace(var_info={"x": [lanes, source]})):
                        code, info = Ops.to_dtype("x", dest)
                    self.assertIn("arith." + opcode, code)
                    self.assertEqual(info, [lanes, dest])

    def test_same_width_float_conversion_is_not_a_bitcast(self):
        for source, dest in (("f16", "bf16"), ("bf16", "f16")):
            kernel = SimpleNamespace(var_info={"x": [8, source], "wide": [8, "f32"]})
            with V.set_kernel_handler(kernel), patch(
                "PyTorchSimFrontend.mlir.mlir_ops.ops.to_dtype", return_value="wide"
            ) as widen:
                code, info = Ops.to_dtype("x", dest)
            widen.assert_called_once_with("x", "f32")
            self.assertIn("arith.truncf %wide", code)
            self.assertEqual(info, [8, dest])

    def test_bf16_arithmetic_opcodes(self):
        x, y = Symbol("x"), Symbol("y")
        for operation in ("add", "sub", "mul"):
            with V.set_kernel_handler(SimpleNamespace(var_info={x: [8, "bf16"], y: [8, "bf16"]})):
                code, info = getattr(Ops, operation)(x, y)
            self.assertIn("arith." + operation + "f ", code)
            self.assertEqual(info, [8, "bf16"])

    def test_float_negation_preserves_type(self):
        for dtype in ("bf16", "f16", "f32", "f64"):
            with V.set_kernel_handler(SimpleNamespace(var_info={"x": [8, dtype]})), patch(
                "PyTorchSimFrontend.mlir.mlir_ops.ops.to_dtype"
            ) as cast:
                code, info = Ops.neg("x")
            cast.assert_not_called()
            self.assertIn(f"arith.negf %x : vector<8x{dtype}>", code)
            self.assertEqual(info, [8, dtype])

    def test_bf16_constants(self):
        for value, bits in (("inf", 0x7F80), ("-inf", 0xFF80), ("nan", 0x7FC0)):
            code, info = Ops.constant(value, "bf16")
            self.assertIn(f"0x{bits:x}", code)
            self.assertEqual(MLIR_INF[value]["bf16"], bits)
            self.assertEqual(info, [1, "bf16"])
        code, _ = Ops.constant(1, "bf16")
        self.assertIn("1.000", code)

    def test_bf16_passes_are_opt_in_for_both_pipelines(self):
        for builder, args in ((mlir_compile_command, ("kernel", 128)),
                              (mlir_gem5_compile_command, ("kernel", "sample", "tog", 128))):
            normal = " ".join(builder(*args))
            bf16 = " ".join(builder(*args, uses_bf16=True))
            self.assertNotIn("include-bf16", normal)
            self.assertIn("include-bf16=true", bf16)
            self.assertIn("source-types=bf16", bf16)
            self.assertIn("-riscv-v-fixed-length-vector-lmul-max=4", bf16)
            self.assertNotIn("-riscv-v-fixed-length-vector-lmul-max", normal)
            self.assertLess(bf16.index("-test-memref-to-gemmini"), bf16.index("-arith-expand"))

    def test_plugin_pipeline_order_and_isolation(self):
        with patch.dict(os.environ, {"TORCHSIM_BF16_PLUGIN": "/scratch/plugin.so"}):
            for builder, args, count in (
                (mlir_compile_command, ("kernel", 128), 7),
                (mlir_gem5_compile_command, ("kernel", "sample", "tog", 128), 8),
            ):
                commands = builder(*args, uses_bf16=True)
                self.assertEqual(len(commands), count)
                self.assertIn("-global-idx", commands[0])
                self.assertNotIn("-arith-expand", commands[0])
                self.assertIn("--pass-pipeline=builtin.module(pytorchsim-bf16", commands[1])
                self.assertIn("--load-dialect-plugin=", commands[2])
                lower_index = 4 if builder is mlir_gem5_compile_command else 2
                self.assertIn("-arith-expand", commands[lower_index])
                self.assertLess(commands[lower_index].index("-test-memref-to-gemmini"), commands[lower_index].index("-arith-expand"))
                self.assertIn("pytorchsim-bf16-memory,instcombine", commands[lower_index + 2])
                self.assertIn(".bf16_memory.ll", commands[lower_index + 3])
                first, plugin, last = map(shlex.split, commands[:3])
                self.assertEqual(first[first.index("-o") + 1], plugin[plugin.index("-o") - 1])
                self.assertEqual(plugin[plugin.index("-o") + 1], last[last.index("-o") - 1])
                self.assertNotIn("plugin.so", " ".join(builder(*args)))

    def test_corrected_builtin_tog_selection_and_old_compiler_guard(self):
        args = ("kernel with space", "sample with space", "tog", 128)
        with patch("PyTorchSimFrontend.extension_codecache.extension_config.CONFIG_TORCHSIM_LLVM_PATH", "/riscv-llvm/bin"):
            with self.assertRaisesRegex(RuntimeError, "built-in mixed-width TOG fix"):
                mlir_gem5_compile_command(*args, uses_bf16=True)
            legacy = mlir_gem5_compile_command(*args)
            self.assertEqual(len(legacy), 3)
            self.assertIn("-test-tile-operation-graph=", " ".join(legacy))
            self.assertNotIn(".tog_pre.mlir", " ".join(legacy))
            mlir_compile_command("kernel", 128, uses_bf16=True)
        with patch.dict(os.environ, {"TORCHSIM_BF16_PLUGIN": "/scratch/bf16 plugin.so"}):
            commands = list(map(shlex.split, mlir_gem5_compile_command(*args, uses_bf16=True)))
            self.assertEqual(len(commands), 8)
            self.assertIn("-test-pytorchsim-to-vcix", " ".join(commands[2]))
            self.assertIn("-test-tile-operation-graph=", " ".join(commands[3]))
            self.assertFalse(any(token.startswith("--load-pass-plugin=") for token in commands[3]))
            self.assertNotIn("pytorchsim-tile-operation-graph{", " ".join(map(shlex.join, commands)))
            for first, second in zip(commands[:4], commands[1:5]):
                self.assertEqual(first[first.index("-o") + 1], second[second.index("-o") - 1])
            self.assertNotIn("-test-tile-operation-graph=", " ".join(mlir_compile_command("kernel", 128, uses_bf16=True)))

    def test_bf16_bitwise_operators_are_rejected(self):
        for operation in ("bitwise_and", "bitwise_or", "bitwise_xor"):
            with V.set_kernel_handler(SimpleNamespace(var_info={"x": [8, "bf16"], "y": [8, "bf16"]})):
                with self.assertRaises(ValueError):
                    getattr(Ops, operation)("x", "y")
        with V.set_kernel_handler(SimpleNamespace(var_info={"x": [8, "bf16"]})):
            with self.assertRaises(ValueError):
                Ops.bitwise_not("x")


class ImplicitTileTests(unittest.TestCase):
    def test_divisor_applies_to_its_symbol_not_dictionary_position(self):
        index0, index1 = sympy.symbols("index0 index1")
        constraints = {
            index1: {ModularIndexing(index1, 1, 1152)},
            index0: {ModularIndexing(index0, 7, 4)},
        }
        for items in (list(constraints.items()), list(reversed(list(constraints.items())))):
            with self.subTest(order=[str(key) for key, _ in items]):
                tile = MLIRMultiDimTile([256, 256], 128, 1, 2)
                tile.apply_constraints(dict(items), [28, 1152])
                self.assertEqual(tile.get_tile_size(), [7, 256])
                self.assertEqual([constraint.fixed for constraint in tile.tile_constraint], [True, False])

    def test_sparse_axis_constraints(self):
        index2 = sympy.Symbol("index2")
        tile = MLIRMultiDimTile([2, 256, 128], 128, 1, 2)
        tile.apply_constraints({index2: {ModularIndexing(index2, 3, 4)}}, [2, 28, 12])
        self.assertEqual(tile.get_tile_size(), [2, 256, 3])

    def test_real_decode_read_expression(self):
        c0, c1 = sympy.symbols("c0 c1")
        operand = SimpleNamespace(index=c1 + 1152 * FloorDiv(c0, 7), ranges={c0: 28, c1: 1152})
        constraints = BaseMLIRKernel.extract_dividers(None, [operand])
        tile = MLIRMultiDimTile([256, 256], 128, 1, 2)
        tile.apply_constraints(constraints, [28, 1152])
        self.assertEqual(tile.get_tile_size(), [7, 256])


class RawIOTests(unittest.TestCase):
    def test_all_bf16_bit_patterns_round_trip(self):
        # Includes signed zero, finite extremes, subnormals, infinities, NaNs.
        words = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
        values = words.view(torch.bfloat16)
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory:
            simulator = FunctionalSimulator(directory, "test")
            simulator.write_arg(values, directory, "input")
            path = Path(directory) / "input/0.raw"
            self.assertEqual(path.read_bytes(), words.numpy().tobytes())
            actual = torch.empty_like(values)
            simulator.load_tensor(actual, "output", None, path)
            torch.testing.assert_close(actual.view(torch.uint16), words, rtol=0, atol=0)

    def test_strided_bf16_output(self):
        words = np.array([0x3F80, 0x4000, 0xC040, 0x4080, 0x40A0, 0x40C0], dtype=np.uint16)
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory:
            path = Path(directory) / "output.raw"
            path.write_bytes(words.tobytes())
            actual = torch.empty_strided((3, 2), (1, 3), dtype=torch.bfloat16)
            FunctionalSimulator(directory, "test").load_tensor(actual, "output", None, path)
            expected = torch.from_numpy(words).view(torch.bfloat16).reshape(2, 3).t()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_other_float_readback_unchanged(self):
        for torch_dtype, np_dtype in ((torch.float16, np.float16), (torch.float32, np.float32)):
            with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory:
                path = Path(directory) / "output.raw"
                values = np.array([0, 1, -2, 3.5], dtype=np_dtype)
                path.write_bytes(values.tobytes())
                actual = torch.empty(4, dtype=torch_dtype)
                FunctionalSimulator(directory, "test").load_tensor(actual, "output", None, path)
                torch.testing.assert_close(actual, torch.from_numpy(values), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
