"""Build TOGSim/tests using the image's existing dependency objects, offline.

Run through run.sh toolchain. With --test-only, test the image's original
objects (no QWEN_TOGSIM_BUILD) or a rebuilt simulator. All output is in TMPDIR.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    root = Path("/workspace/PyTorchSim/TOGSim")
    build = root / "build"
    scratch = Path(os.environ["TMPDIR"])
    database = json.loads((build / "compile_commands.json").read_text())
    commands = [item for item in database if Path(item["file"]).is_relative_to(root / "src")]
    assert commands
    if not args.test_only:
        # QWEN_TOGSIM_BUILD binds this directory to a copy under host TMPDIR.
        assert os.access(build, os.W_OK), "Select a writable scratch build"
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(subprocess.run, shlex.split(item["command"]),
                                   cwd=item["directory"], check=True) for item in commands]
            for future in futures:
                future.result()
        subprocess.run(shlex.split((build / "src/CMakeFiles/Simulator.dir/link.txt").read_text()),
                       cwd=build / "src", check=True)
        sources = sorted((root / "src").rglob("*.cc")) + sorted((root / "include").rglob("*.h"))
        manifest = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sources}
        manifest["build/bin/Simulator"] = hashlib.sha256((build / "bin/Simulator").read_bytes()).hexdigest()
        (build / "completion-fix-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    template = commands[0]
    compile_cmd = shlex.split(template["command"])
    compile_cmd[compile_cmd.index("-o") + 1] = str(scratch / "completion_test.o")
    compile_cmd[compile_cmd.index("-c") + 1] = str(root / "tests/completion_test.cc")
    subprocess.run(compile_cmd, cwd=template["directory"], check=True)
    link_cmd = shlex.split((build / "src/CMakeFiles/Simulator.dir/link.txt").read_text())
    link_cmd = [str(scratch / "completion_test.o") if arg.endswith("/main.cc.o") else arg
                for arg in link_cmd]
    link_cmd[link_cmd.index("-o") + 1] = str(scratch / "completion_test")
    subprocess.run(link_cmd, cwd=build / "src", check=True)
    subprocess.run([str(scratch / "completion_test")], check=True)


if __name__ == "__main__":
    main()
