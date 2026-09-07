"""Baseline selection and installed-package integrity checks; no PyTorch imports."""
import ast
import base64
import contextlib
import hashlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from Simulator.experiments.qwen2_5_7b import provenance, runner
from Simulator.experiments.qwen2_5_7b.analysis.trace_summary import phase_file


class ProvenanceTests(unittest.TestCase):
    def distribution(self):
        path = Path(provenance.__file__)
        digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b"=").decode()
        entries = []
        for name in provenance.MODEL_SOURCES:
            entry = MagicMock()
            entry.__str__.return_value = name
            entry.hash = SimpleNamespace(mode="sha256", value=digest)
            entries.append(entry)
        return SimpleNamespace(version=provenance.TRANSFORMERS_VERSION, files=entries,
                               locate_file=lambda _entry: path)

    def test_pinned_package_sources(self):
        with patch.object(provenance.metadata, "distribution", return_value=self.distribution()):
            result = provenance.transformers_provenance()
        self.assertEqual(result["version"], "4.43.4")
        self.assertEqual(set(result["sources"]), set(provenance.MODEL_SOURCES))

    def test_reject_version_drift(self):
        distribution = self.distribution()
        distribution.version = "0.0.0"
        with patch.object(provenance.metadata, "distribution", return_value=distribution):
            with self.assertRaisesRegex(RuntimeError, "requires transformers"):
                provenance.transformers_provenance()

    def test_reject_modified_or_unrecorded_sources(self):
        for missing in (False, True):
            distribution = self.distribution()
            if missing:
                distribution.files = []
            else:
                distribution.files[0].hash.value = "wrong"
            with self.subTest(missing=missing), patch.object(provenance.metadata, "distribution", return_value=distribution):
                with self.assertRaises(RuntimeError):
                    provenance.transformers_provenance()

    def test_default_is_standard_decoder(self):
        with patch("sys.argv", ["qwen", "--mode", "audit"]):
            self.assertEqual(runner.parse_args().component, "decoder")

    def test_decoder_timing_requires_numerical_validation(self):
        with patch("sys.argv", ["qwen", "--mode", "timing"]), contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit) as error:
                runner.parse_args()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("requires --validate-timing", errors.getvalue())

    def test_exploratory_option_is_explicit_and_scoped(self):
        with patch("sys.argv", ["qwen", "--mode", "audit"]):
            self.assertFalse(runner.parse_args().allow_numerical_mismatch)
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as output:
            valid = ["qwen", "--mode", "timing", "--validate-timing",
                     "--allow-numerical-mismatch", "--output-dir", output]
            with patch("sys.argv", valid):
                self.assertTrue(runner.parse_args().allow_numerical_mismatch)
            invalid = [
                ["qwen", "--mode", "audit", "--allow-numerical-mismatch"],
                ["qwen", "--mode", "timing", "--allow-numerical-mismatch"],
                valid + ["--component", "attention"],
            ]
            for arguments in invalid:
                with self.subTest(arguments=arguments), patch("sys.argv", arguments), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        runner.parse_args()

    def test_phase_metadata_is_explicit_and_backwards_compatible(self):
        self.assertEqual(phase_file({}), "attention_phases.json")
        self.assertEqual(phase_file({"probe": {"phase_file": "decoder_phases.json"}}), "decoder_phases.json")
        with self.assertRaises(ValueError):
            phase_file({"probe": {"phase_file": "../result.json"}})

    def test_decoder_has_no_local_model_class_or_forward(self):
        path = Path(runner.__file__).parent / "workloads/decoder.py"
        tree = ast.parse(path.read_text())
        self.assertFalse(any(isinstance(node, ast.ClassDef) for node in ast.walk(tree)))
        self.assertFalse(any(isinstance(node, ast.FunctionDef) and node.name == "forward" for node in ast.walk(tree)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, ("key_cache", "value_cache", "_seen_tokens"))


if __name__ == "__main__":
    unittest.main()
