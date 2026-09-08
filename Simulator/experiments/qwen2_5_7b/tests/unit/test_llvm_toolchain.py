"""Permanent compiler selection and build-integrity checks, without LLVM/PyTorch."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PyTorchSimFrontend.toolchain import (
    _has_mixed_width_tog, llvm_binary_directory, require_mixed_width_tog, selected_llvm_install,
)


REPO = Path(__file__).resolve().parents[5]
SCRIPTS = REPO / "scripts/toolchains/llvm"
spec = importlib.util.spec_from_file_location("llvm_build_record", SCRIPTS / "record_build.py")
records = importlib.util.module_from_spec(spec)
spec.loader.exec_module(records)


class LLVMToolchainTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        (self.root / "configs").mkdir()
        (self.root / "configs/toolchains.local.json").write_text(json.dumps({"llvm_install": "/local/llvm"}))
        _has_mixed_width_tog.cache_clear()

    def test_local_selection_and_explicit_override(self):
        self.assertEqual(selected_llvm_install(self.root, {}), "/local/llvm")
        self.assertEqual(llvm_binary_directory(self.root, {}), "/local/llvm/bin")
        self.assertEqual(llvm_binary_directory(self.root, {"TORCHSIM_LLVM_PATH": "/explicit/bin"}), "/explicit/bin")
        self.assertEqual(llvm_binary_directory(self.root, {
            "TORCHSIM_LLVM_PATH": "/image/bin", "TORCHSIM_LLVM_ROOT": "/explicit/llvm",
        }), "/explicit/llvm/bin")

    def test_diagnostic_opt_out_and_no_implicit_host_path(self):
        self.assertIsNone(selected_llvm_install(self.root, {"TORCHSIM_LLVM_ROOT": ""}))
        self.assertEqual(llvm_binary_directory(self.root, {"TORCHSIM_LLVM_ROOT": ""}), "/usr/bin")
        self.assertEqual(llvm_binary_directory(self.root / "missing", {}), "/usr/bin")
        with self.assertRaisesRegex(ValueError, "absolute"):
            selected_llvm_install(self.root, {"TORCHSIM_LLVM_ROOT": "relative"})

    def test_builtin_capability_is_required_and_cached(self):
        (self.root / "mlir-opt").write_bytes(b"compiler fixture")
        with patch("PyTorchSimFrontend.toolchain.subprocess.run", return_value=SimpleNamespace(stdout="old pass")):
            with self.assertRaisesRegex(RuntimeError, "built-in mixed-width TOG fix"):
                require_mixed_width_tog(self.root)
        _has_mixed_width_tog.cache_clear()
        with patch("PyTorchSimFrontend.toolchain.subprocess.run",
                   return_value=SimpleNamespace(stdout="test-tile-operation-graph (mixed-width-matmul)")) as run:
            require_mixed_width_tog(self.root)
            require_mixed_width_tog(self.root)
            self.assertEqual(run.call_count, 1)
            (self.root / "mlir-opt").write_bytes(b"replacement compiler fixture")
            require_mixed_width_tog(self.root)
            self.assertEqual(run.call_count, 2)

    def test_binary_and_source_fingerprints(self):
        install, source = self.root / "install", self.root / "source"
        for name in records.TOOLS:
            path = install / "bin" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        for name in records.SOURCE_FILES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        record = {"schema_version": 1, "features": ["mixed_width_tog"],
                  "binaries": {name: records.digest(install / "bin" / name) for name in records.TOOLS},
                  "source_files": {name: records.digest(source / name) for name in records.SOURCE_FILES}}
        manifest = install / records.MANIFEST
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps(record))
        records.verify(install, source)
        (install / "bin/opt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "binary differs"):
            records.verify(install)
        (install / "bin/opt").write_bytes(b"opt")
        (source / records.SOURCE_FILES[0]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            records.verify(install, source)

    def test_install_never_overwrites_an_existing_prefix(self):
        prefix = self.root / "existing"
        prefix.mkdir()
        sentinel = prefix / "sentinel"
        sentinel.write_text("preserve")
        result = subprocess.run(["bash", str(SCRIPTS / "install.sh"), str(self.root),
                                 str(self.root), str(prefix)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(sentinel.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
