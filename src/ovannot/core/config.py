"""YAML configuration: composition with ``extends``, dotted overrides, stable hashing.

A config file may start with ``extends: <path>`` (or a list of paths, relative to the file).
The files it extends are loaded first, in order, then the file's own keys are deep-merged on
top. Dicts merge key by key; any other value (including lists) is replaced.

Overrides use dotted keys and YAML values, as on the command line::

    load_config("configs/experiments/x.yaml", ["pipeline_cfg.iou_threshold=0.6"])
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ovannot.utils.hashing import hash_obj
from ovannot.utils.io import atomic_write_text

EXTENDS_KEY = "extends"


class ConfigError(Exception):
    pass


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads ``1e-3`` as a float (YAML 1.1 requires ``1.0e-3``)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


def parse_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)  # a SafeLoader subclass: no arbitrary objects


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        data = parse_yaml(f.read())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def dump_yaml(cfg: Mapping[str, Any], path: str | Path) -> None:
    atomic_write_text(path, yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True))


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def parse_override(text: str) -> tuple[list[str], Any]:
    """``"a.b=0.5"`` -> ``(["a", "b"], 0.5)``; the value is parsed as YAML."""
    key, sep, raw = text.partition("=")
    if not sep or not key.strip():
        raise ConfigError(f"override must look like key.sub=value, got {text!r}")
    return key.strip().split("."), parse_yaml(raw)


def apply_overrides(cfg: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    result = deepcopy(dict(cfg))
    for text in overrides:
        keys, value = parse_override(text)
        node = result
        for k in keys[:-1]:
            child = node.setdefault(k, {})
            if not isinstance(child, dict):
                raise ConfigError(f"override {text!r}: {k!r} is not a mapping")
            node = child
        node[keys[-1]] = value
    return result


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load ``path`` with its ``extends`` chain resolved, then apply ``overrides``."""
    return apply_overrides(_load_with_extends(Path(path).resolve(), ()), overrides)


def _load_with_extends(path: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
    if path in chain:
        cycle = " -> ".join(str(p) for p in (*chain, path))
        raise ConfigError(f"cyclic 'extends': {cycle}")
    own = load_yaml(path)
    parents = own.pop(EXTENDS_KEY, None) or []
    if isinstance(parents, str):
        parents = [parents]
    merged: dict[str, Any] = {}
    for parent in parents:
        merged = deep_merge(
            merged, _load_with_extends((path.parent / parent).resolve(), (*chain, path))
        )
    return deep_merge(merged, own)


def config_hash(cfg: Mapping[str, Any], length: int = 8) -> str:
    """Short stable hash of a resolved config (key order does not matter)."""
    return hash_obj(cfg, length)
