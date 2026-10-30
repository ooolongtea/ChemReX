"""Resolve moved batch sources without embedding a machine-specific root."""

import os
from pathlib import Path


def resolve_source(path: str, data_root: str = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    root = data_root or os.getenv("CHEMREX_DATA_ROOT")
    if not root:
        raise FileNotFoundError(f"source not found: {path}; set CHEMREX_DATA_ROOT")
    marker = "chemrex_batch/"
    normalized = str(candidate).replace("\\", "/")
    if marker not in normalized:
        raise FileNotFoundError(f"source not found and has no chemrex_batch component: {path}")
    relative = normalized.split(marker, 1)[1]
    relocated = Path(root).expanduser() / relative
    if not relocated.exists():
        raise FileNotFoundError(f"source not found at {candidate} or {relocated}")
    return relocated.resolve()


def source_reference(path: str) -> dict:
    resolved = resolve_source(path)
    normalized = str(resolved)
    marker = "chemrex_batch/"
    return {
        "source": normalized,
        "source_relative": normalized.split(marker, 1)[1] if marker in normalized else None,
    }
