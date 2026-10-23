"""Access to models by name, with lazily created wrappers.

Each model has a config ``configs/models/<name>.yaml`` whose ``kind`` (e.g. an OpenAI-compatible
MLLM endpoint, SAM 3) selects the wrapper class registered with ``@register_model_kind``.
Pipelines ask the hub for a model by name and never see endpoints, checkpoints or devices.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ovannot.core.config import load_yaml
from ovannot.core.registry import Registry, RegistryError
from ovannot.paths import repo_root

MODEL_KINDS: Registry[Any] = Registry("model kind")


def register_model_kind(kind: str):
    """Register a wrapper class, built as ``cls(name, cfg)`` the first time it is needed."""

    def decorator(cls: type) -> type:
        MODEL_KINDS.add(kind, cls)
        return cls

    return decorator


class ModelHub:
    def __init__(self, model_cfgs: Mapping[str, Mapping[str, Any]]) -> None:
        self._cfgs = {name: dict(cfg) for name, cfg in model_cfgs.items()}
        self._instances: dict[str, Any] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_dir(cls, path: Path | None = None) -> ModelHub:
        """Load every ``*.yaml`` of ``configs/models/``; the file stem is the model name."""
        folder = path or repo_root() / "configs" / "models"
        return cls({p.stem: load_yaml(p) for p in sorted(folder.glob("*.yaml"))})

    def names(self) -> list[str]:
        return sorted(self._cfgs)

    def config(self, name: str) -> dict[str, Any]:
        try:
            return dict(self._cfgs[name])
        except KeyError:
            raise RegistryError(f"unknown model {name!r} (known: {self.names()})") from None

    def get(self, name: str) -> Any:
        """The wrapper for model ``name``, created on first use (thread-safe)."""
        with self._lock:
            if name not in self._instances:
                cfg = self.config(name)
                if "kind" not in cfg:
                    raise RegistryError(f"model config {name!r} has no 'kind'")
                self._instances[name] = MODEL_KINDS.get(cfg["kind"])(name, cfg)
            return self._instances[name]
