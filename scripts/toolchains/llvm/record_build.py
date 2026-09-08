"""Fingerprint and verify the installed, source-built PyTorchSim LLVM tools."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


TOOLS = ("mlir-opt", "mlir-translate", "llc", "opt", "llvm-config")
SOURCE_FILES = (
    "mlir/test/lib/Analysis/TestTileOperationGraph.cpp",
    "mlir/tools/mlir-opt/mlir-opt.cpp",
    "mlir/test/Analysis/tile-operation-graph-mixed-width.mlir",
    "mlir/test/Analysis/Inputs/tog-mixed-width.py",
)
MANIFEST = "share/pytorchsim/llvm-build.json"


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def verify(install, source=None):
    install = Path(install).resolve()
    record = json.loads((install / MANIFEST).read_text())
    if record.get("schema_version") != 1 or "mixed_width_tog" not in record.get("features", []):
        raise ValueError("LLVM installation lacks the mixed-width TOG build record")
    if set(record["binaries"]) != set(TOOLS):
        raise ValueError("Incomplete LLVM binary manifest")
    if set(record["source_files"]) != set(SOURCE_FILES):
        raise ValueError("Incomplete LLVM source manifest")
    for name, expected in record["binaries"].items():
        if digest(install / "bin" / name) != expected:
            raise ValueError(f"LLVM binary differs from its build record: {name}")
    if source:
        source = Path(source).resolve()
        for name, expected in record["source_files"].items():
            if digest(source / name) != expected:
                raise ValueError(f"LLVM source changed since this build: {name}; rebuild before selecting it")
    return record


def record(source, build):
    source, build = Path(source).resolve(), Path(build).resolve()
    install = build / "install"
    cache = {}
    for line in (build / "build/CMakeCache.txt").read_text().splitlines():
        if line.startswith(("CMAKE_BUILD_TYPE:", "CMAKE_CXX_COMPILER:", "LLVM_ENABLE_PROJECTS:",
                            "LLVM_TARGETS_TO_BUILD:", "LLVM_ENABLE_RTTI:", "LLVM_ENABLE_ASSERTIONS:",
                            "LLVM_USE_LINKER:", "BUILD_SHARED_LIBS:")):
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    result = {
        "schema_version": 1,
        "features": ["mixed_width_tog"],
        "upstream": git(source, "remote", "get-url", "upstream"),
        "origin": git(source, "remote", "get-url", "origin"),
        "source_path": str(source),
        "source_revision": git(source, "rev-parse", "HEAD"),
        "source_branch": git(source, "branch", "--show-current"),
        "source_dirty": bool(git(source, "status", "--porcelain", "--untracked-files=no")),
        "build_start_revision": (build / "source-revision.txt").read_text().strip(),
        "build_start_diff_sha256": digest(build / "source.diff"),
        "source_files": {name: digest(source / name) for name in SOURCE_FILES},
        "binaries": {name: digest(install / "bin" / name) for name in TOOLS},
        "cmake": cache,
        "build_directory": str(build),
        "runtime": "built inside the PyTorchSim Ubuntu 22.04 container; use its runtime environment",
    }
    destination = install / MANIFEST
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    verify(install, source)
    print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    create = subs.add_parser("record")
    create.add_argument("source", type=Path)
    create.add_argument("build", type=Path)
    check = subs.add_parser("verify")
    check.add_argument("install", type=Path)
    check.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.command == "record":
        record(args.source, args.build)
    else:
        result = verify(args.install, args.source)
        print("Verified LLVM:", result["source_revision"], result["source_branch"])


if __name__ == "__main__":
    main()
