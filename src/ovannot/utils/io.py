"""File writing helpers that never leave a half-written file behind."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ovannot.core.types import to_jsonable

_UMASK = os.umask(0)
os.umask(_UMASK)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file and an atomic rename.

    Safe with concurrent readers and writers on the same file system (the last writer wins).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, 0o666 & ~_UMASK)  # mkstemp makes 0600 files; results are shared
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json(path: str | Path, obj: Any, indent: int | None = 2) -> None:
    atomic_write_text(path, json.dumps(to_jsonable(obj), indent=indent, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)
