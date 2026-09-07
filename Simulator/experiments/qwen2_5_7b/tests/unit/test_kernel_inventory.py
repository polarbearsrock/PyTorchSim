import os
from pathlib import Path
import tempfile
import unittest

from Simulator.experiments.qwen2_5_7b.analysis.kernel_inventory import inventory, wrapper_inventory


class KernelInventoryTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(self.scratch.cleanup)
        self.run = Path(self.scratch.name)
        self.wrapper = self.run / "wrapper.py"
        self.source = "// M = 1\n// N = 4\n// K = 8\nlinalg.matmul"
        self.wrapper.write_text(
            f"mlir_kernel_0 = custom_async_compile.mlir({self.source!r}, origins={{'mm'}}, "
            "arg_attributes=[['x', [1, torch.bfloat16, 8, [1, 8], [8, 1]]]])\n"
            "def call(args):\n    mlir_kernel_0(buf0)\n    mlir_kernel_0(buf1)\n")
        generated = self.run / "generated" / "kernel"
        generated.mkdir(parents=True)
        (generated / "cabc.mlir").write_text(self.source)
        (self.run / "console.log").write_text(f"Starting decoder prefill: queries=1, cache=1\nWrapper Codegen Path = {self.wrapper}\n")
        (self.run / "logs").mkdir()
        (self.run / "logs" / "sample.trace").write_text("\n".join(
            f"LAUNCH_KERNEL,{index},0,0,{generated}/tile_graph.onnx,{generated}/runtime_{index:04d}/attribute/0,0"
            for index in range(2)))

    def test_reused_function_keeps_each_call_and_runtime(self):
        rows = inventory(self.run)
        self.assertEqual([r["call"] for r in rows], ["mlir_kernel_0(buf0)", "mlir_kernel_0(buf1)"])
        self.assertTrue(rows[1]["runtime_dir"].endswith("runtime_0001"))
        self.assertEqual(rows[0]["arguments"][0][1][1], "torch.bfloat16")
        self.assertEqual(rows[0]["dimensions"], {"M": 1, "N": 4, "K": 8})

    def test_source_mismatch_is_rejected(self):
        (self.run / "generated/kernel/cabc.mlir").write_text("different source")
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            inventory(self.run)

    def test_direct_external_operator_is_rejected(self):
        self.wrapper.write_text(self.wrapper.read_text() + "    extern_kernels.mm(buf0, buf1)\n")
        with self.assertRaisesRegex(ValueError, "Unaccounted direct operator"):
            wrapper_inventory(self.wrapper)


if __name__ == "__main__":
    unittest.main()
