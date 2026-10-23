"""Default locations of the repository, the data and the caches.

Every location can be overridden from the environment, which is how the cluster points the
code at shared storage:

- ``OVANNOT_ROOT``: repository root (default: inferred from this file, editable install).
- ``OVANNOT_DATA``: datasets root (default: ``<root>/data``).
- ``OVANNOT_CACHE``: model-call cache (default: ``<root>/cache``).
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("OVANNOT_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[2]


def data_root() -> Path:
    env = os.environ.get("OVANNOT_DATA")
    return Path(env).resolve() if env else repo_root() / "data"


def cache_root() -> Path:
    env = os.environ.get("OVANNOT_CACHE")
    return Path(env).resolve() if env else repo_root() / "cache"


def results_root() -> Path:
    return repo_root() / "results"
