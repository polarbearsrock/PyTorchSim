"""Local compiler selection and capability checks (no PyTorch dependency)."""
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess


def selected_llvm_install(repo, environ=None):
    environ = os.environ if environ is None else environ
    # An explicitly empty root opts out of the local setting for diagnostic
    # commands. BF16 timing still independently rejects the old compiler.
    if "TORCHSIM_LLVM_ROOT" in environ:
        value = environ["TORCHSIM_LLVM_ROOT"]
    else:
        local = Path(repo) / "configs/toolchains.local.json"
        value = json.loads(local.read_text()).get("llvm_install") if local.exists() else None
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("llvm_install / TORCHSIM_LLVM_ROOT must be an absolute installation path")
    return str(path)


def llvm_binary_directory(repo, environ=None):
    environ = os.environ if environ is None else environ
    if environ.get("TORCHSIM_LLVM_ROOT"):
        return str(Path(selected_llvm_install(repo, environ)) / "bin")
    if environ.get("TORCHSIM_LLVM_PATH"):
        return environ["TORCHSIM_LLVM_PATH"]
    root = selected_llvm_install(repo, environ)
    return str(Path(root) / "bin") if root else "/usr/bin"


@lru_cache(maxsize=8)
def _has_mixed_width_tog(compiler, size, mtime_ns):
    # A capability probe, not a version guess. File identity is part of the
    # cache key so replacing a compiler in the same process invalidates it.
    result = subprocess.run([compiler, "--help"], text=True, capture_output=True,
                            check=True, timeout=30)
    return "mixed-width-matmul" in result.stdout


def supports_mixed_width_tog(binary_directory):
    compiler = Path(binary_directory) / "mlir-opt"
    info = compiler.stat()
    return _has_mixed_width_tog(str(compiler), info.st_size, info.st_mtime_ns)


def require_mixed_width_tog(binary_directory):
    if not supports_mixed_width_tog(binary_directory):
        raise RuntimeError(
            "BF16 timing requires LLVM with the built-in mixed-width TOG fix. "
            "Select the source-built compiler using TORCHSIM_LLVM_ROOT or "
            "configs/toolchains.local.json; see scripts/toolchains/llvm/README.md. "
            "The old compiler misattributes FP32 matrix result reads to VPU."
        )
