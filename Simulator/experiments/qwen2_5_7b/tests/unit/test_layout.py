"""Entry-point and package-boundary checks that do not require a container."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


STUDY = Path(__file__).resolve().parents[2]
REPO = STUDY.parents[2]
PACKAGE = "Simulator.experiments.qwen2_5_7b"


def command(*args, cwd=REPO):
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=True)


def audit_payload(output):
    decoder = json.JSONDecoder()
    _, end = decoder.raw_decode(output.lstrip())
    return json.loads(output.lstrip()[end:].strip())


class LayoutTests(unittest.TestCase):
    def test_entry_point_imports_do_not_initialize_pytorch(self):
        command(sys.executable, "-B", "-c", f"import sys; import {PACKAGE}.runner; "
                f"import {PACKAGE}.analysis.trace_summary; "
                "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules")

    def test_launcher_and_legacy_alias_from_another_directory(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory:
            current = command("bash", str(STUDY / "run.sh"), "audit", "--context-tokens", "2048", cwd=directory)
            legacy = command("bash", str(REPO / "experiments/qwen2_5_7b/run.sh"), "audit", "--context-tokens", "2048", cwd=directory)
        self.assertEqual(audit_payload(current.stdout), audit_payload(legacy.stdout))
        self.assertEqual(audit_payload(current.stdout)["parameters"]["total"], 7615616512)

    def test_module_help_and_invalid_argument_exit(self):
        result = command(sys.executable, "-B", "-m", PACKAGE, "--help")
        self.assertIn("--decode-steps", result.stdout)
        result = subprocess.run([sys.executable, "-B", "-m", PACKAGE, "--mode", "audit", "--seq-len", "0"],
                                cwd=REPO, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)

    def test_analysis_module_entry_points(self):
        for module in ("dependency_audit", "trace_summary", "operator_timeline", "kernel_inventory"):
            with self.subTest(module=module):
                result = command(sys.executable, "-B", "-m", f"{PACKAGE}.analysis.{module}", "--help")
                self.assertIn("usage:", result.stdout)

    def test_shell_entry_points_parse(self):
        for script in STUDY.rglob("*.sh"):
            with self.subTest(script=script):
                command("bash", "-n", str(script))

    def test_local_documentation_links_resolve(self):
        for document in STUDY.rglob("*.md"):
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", document.read_text()):
                target = target.split("#", 1)[0]
                if not target or "://" in target or target.startswith("/"):
                    continue
                with self.subTest(document=document, target=target):
                    self.assertTrue((document.parent / target).exists())


if __name__ == "__main__":
    unittest.main()
