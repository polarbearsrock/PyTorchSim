"""Host-only checks for the regression contract and installation integrity."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from Simulator.experiments.qwen2_5_7b.tests.integration.gem5_matrix import CASES, assembly, queue_counts


STUDY = Path(__file__).resolve().parents[2]


class Gem5ToolchainTests(unittest.TestCase):
    def test_cases_balance_actual_elements_not_instruction_counts(self):
        for case in CASES:
            with self.subTest(case=case.name):
                self.assertEqual(case.feed_vl * case.pushes, case.pop_vl * case.pops)
                self.assertIn(".word 0x0800345b", assembly(case))
        s128 = next(case for case in CASES if case.name == "bf16_s128")
        self.assertEqual((s128.feed_vl, s128.pop_avl, s128.pop_vl), (16, 16, 8))
        self.assertNotEqual(s128.pushes, s128.pops)

    def test_debug_parser_retains_zero_length_transfers(self):
        parsed = queue_counts("pushInput: input size: 0\nSize before vpop(0): 0\nSize after vpop: 0\n")
        self.assertEqual(parsed, {"input_sizes": [0], "pop_sizes": [0], "remaining": [0]})

    def test_build_manifest_rejects_changed_binary(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as temporary:
            root = Path(temporary)
            (root / "bin").mkdir()
            binary = root / "bin/gem5.opt"
            binary.write_bytes(b"original simulator")
            (root / "build.json").write_text(json.dumps({"sha256": {"bin/gem5.opt": hashlib.sha256(binary.read_bytes()).hexdigest()}}))
            command = [sys.executable, "-B", str(STUDY / "tools/gem5/record_build.py"), "--verify", str(root)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            binary.write_bytes(b"different simulator")
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("does not match", failed.stderr)

    def test_installer_never_overwrites_existing_destination(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as temporary:
            result = subprocess.run(["bash", str(STUDY / "tools/gem5/install.sh"), temporary, temporary,
                                     "https://github.com/example/gem5.git"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("refusing to overwrite", result.stderr)

    def test_launcher_accepts_permanent_root_outside_scratch(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as temporary:
            root = Path(temporary)
            scratch, install, commands = root / "scratch", root / "permanent", root / "commands"
            scratch.mkdir()
            (install / "bin").mkdir(parents=True)
            commands.mkdir()
            binary = install / "bin/gem5.opt"
            binary.write_bytes(b"test executable; never executed")
            binary.chmod(0o755)
            manifest = {"sha256": {"bin/gem5.opt": hashlib.sha256(binary.read_bytes()).hexdigest()}}
            (install / "build.json").write_text(json.dumps(manifest))
            (root / "image.sif").touch()
            # Capture the launch arguments without executing a real container.
            fake = commands / "apptainer"
            fake.write_text("#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
            fake.chmod(0o755)
            env = {key: value for key, value in os.environ.items() if not key.startswith("QWEN_")}
            env.update(TMPDIR=str(scratch), QWEN_GEM5_ROOT=str(install),
                       QWEN_IMAGE=str(root / "image.sif"), PATH=str(commands) + os.pathsep + env["PATH"])
            result = subprocess.run(["bash", str(STUDY / "run.sh"), "toolchain", "true"],
                                    env=env, capture_output=True, text=True, check=True)
            run = Path(re.search(r"Run directory: (.+)", result.stdout).group(1))
            arguments = json.loads((run / "console.log").read_text())
            self.assertIn(str(binary) + ":/workspace/gem5-toolchain/bin/gem5.opt:ro", arguments)
            self.assertIn("GEM5_PATH=/workspace/gem5-toolchain/bin/gem5.opt", arguments)
            self.assertEqual(json.loads((run / "gem5-build.json").read_text()), manifest)


if __name__ == "__main__":
    unittest.main()
