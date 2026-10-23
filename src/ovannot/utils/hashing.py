"""Stable hashes for cache keys, run ids and file integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ovannot.core.types import to_jsonable


def stable_json(obj: Any) -> str:
    """Canonical JSON (sorted keys, no whitespace): equal values give equal strings."""
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_bytes(data: bytes, length: int | None = None) -> str:
    digest = hashlib.sha256(data).hexdigest()
    return digest[:length] if length else digest


def hash_obj(obj: Any, length: int | None = 16) -> str:
    """Hash of the canonical JSON form of ``obj`` (dicts, lists, dataclasses, paths...)."""
    return hash_bytes(stable_json(obj).encode("utf-8"), length)


def hash_file(path: str | Path, length: int | None = None, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    digest = h.hexdigest()
    return digest[:length] if length else digest
