"""Check and fingerprint the installed Transformers baseline, without importing it."""
import base64
import hashlib
from importlib import metadata
from pathlib import Path


TRANSFORMERS_VERSION = "4.43.4"
MODEL_SOURCES = (
    "transformers/models/qwen2/modeling_qwen2.py",
    "transformers/models/qwen2/configuration_qwen2.py",
    "transformers/cache_utils.py",
    "transformers/modeling_attn_mask_utils.py",
    "transformers/activations.py",
    "transformers/modeling_utils.py",
)


def transformers_provenance():
    distribution = metadata.distribution("transformers")
    if distribution.version != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"This baseline requires transformers=={TRANSFORMERS_VERSION}; "
            f"found {distribution.version}. Validate a version change explicitly."
        )
    entries = {str(entry): entry for entry in distribution.files or ()}
    sources = {}
    for name in MODEL_SOURCES:
        entry = entries.get(name)
        if entry is None or entry.hash is None or entry.hash.mode != "sha256":
            raise RuntimeError(f"No installed-package SHA256 record for {name}")
        path = Path(distribution.locate_file(entry)).resolve()
        digest = hashlib.sha256(path.read_bytes()).digest()
        encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        if encoded != entry.hash.value:
            raise RuntimeError(f"Transformers source differs from its installed-package record: {name}")
        sources[name] = {"path": str(path), "sha256": digest.hex()}
    return {
        "version": distribution.version,
        "source_integrity": "matches installed distribution RECORD (not a package-signature check)",
        "sources": sources,
    }
