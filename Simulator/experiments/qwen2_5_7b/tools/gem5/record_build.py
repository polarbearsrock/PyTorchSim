"""Record the exact source, backport, build inputs, and installed executable."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root, patch=None):
    recorded = json.loads((root / "build.json").read_text())
    if sha256(root / "bin/gem5.opt") != recorded["sha256"]["bin/gem5.opt"]:
        raise SystemExit("gem5 binary does not match build.json; refusing stale build metadata")
    if patch is not None and sha256(patch) != recorded["sha256"]["matrix-vl.patch"]:
        raise SystemExit("gem5 backport does not match build.json")
    return recorded


def main():
    if sys.argv[1] == "--verify":
        verify(Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
        return
    if sys.argv[1] == "--record-install":
        root = Path(sys.argv[2])
        recorded = verify(root, root / "pytorchsim-patches/matrix-vl.patch")
        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
        if git("rev-parse", "HEAD") != recorded["base_revision"]:
            raise SystemExit("Permanent checkout does not match the build's base revision")
        if sha256(root / "src/cpu/minor/execute.cc") != recorded["sha256"]["patched_execute.cc"]:
            raise SystemExit("Permanent checkout does not match the built matrix-unit source")
        install = {"origin": git("remote", "get-url", "origin"),
                   "upstream": git("remote", "get-url", "upstream"),
                   "branch": git("branch", "--show-current"),
                   "base_revision": git("rev-parse", "HEAD"),
                   "source_diff_sha256": hashlib.sha256(git("diff", "--", "src/cpu/minor/execute.cc").encode()).hexdigest(),
                   "binary_sha256": recorded["sha256"]["bin/gem5.opt"],
                   "source_matches_build": True}
        (root / "installation.json").write_text(json.dumps(install, indent=2) + "\n")
        return
    root = Path(sys.argv[1])
    scripts = Path(__file__).resolve().parent
    revision = "0511678eb5334c1102725cd928f2d3de8720d1bc"
    result = {
        "repository": "https://github.com/PSAL-POSTECH/gem5",
        "base_revision": revision,
        "base_release": "v1.0.1",
        "backport_from": "3ac8959462ad42e5096598594d165d5887ce5b39",
        "matrix_element_accounting": "instruction_vl",
        "jobs": int(sys.argv[2]),
        "python": sys.version,
        "cxx": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "build_arguments": ["build/RISCV/gem5.opt", "PYTHON_CONFIG=/usr/bin/python3-config", "--linker=lld", "--without-tcmalloc"],
        "sha256": {
            "bin/gem5.opt": sha256(root / "bin/gem5.opt"),
            "gem5-release.tar.gz": sha256(root / "gem5-release.tar.gz"),
            "scons_wheel": sha256(root / "build-deps/SCons-4.8.1-py3-none-any.whl"),
            "matrix-vl.patch": sha256(scripts / "matrix-vl.patch"),
            "patched_execute.cc": sha256(root / f"gem5-{revision}/src/cpu/minor/execute.cc"),
            "build_in_container.sh": sha256(scripts / "build_in_container.sh"),
        },
    }
    (root / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
