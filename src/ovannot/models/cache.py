"""Disk cache of model calls, so that re-running an experiment recomputes only what changed.

The key is the hash of everything that determines the answer: model name, checkpoint and
revision, prompt or messages, image hashes and generation parameters. Entries are JSON files
under ``$OVANNOT_CACHE/<model>/<key[:2]>/<key>.json``, written atomically, so concurrent runs
can share the cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ovannot.paths import cache_root
from ovannot.utils.hashing import hash_obj
from ovannot.utils.io import read_json, write_json


class CallCache:
    def __init__(self, model: str, root: Path | None = None, enabled: bool = True) -> None:
        self.dir = (root or cache_root()) / model
        self.enabled = enabled

    @staticmethod
    def key(**parts: Any) -> str:
        return hash_obj(parts, 32)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(key)
        try:
            return read_json(path)
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:  # never written by write_json, but do not trust disks
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, value: Any) -> None:
        if self.enabled:
            write_json(self._path(key), value, indent=None)
