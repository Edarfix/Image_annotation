"""Name -> class registries, and discovery of plugin folders by file path (decision D8).

Pipelines (``pipelines/<id>/pipeline.py``) and datasets (``datasets/<name>/loader.py``) are
plugin folders. Importing a plugin file runs its ``@register_*`` decorator. Plugins are
imported by path under a private module name, so that the folders need no packaging and
``datasets/`` cannot clash with the Hugging Face ``datasets`` library.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


class RegistryError(Exception):
    pass


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def add(self, name: str, cls: type[T]) -> None:
        existing = self._items.get(name)
        if existing is not None and existing is not cls:
            raise RegistryError(
                f"{self.kind} {name!r} is already registered by {existing.__module__}"
            )
        self._items[name] = cls

    def get(self, name: str) -> type[T]:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "none"
            raise RegistryError(f"unknown {self.kind} {name!r} (registered: {known})") from None

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def names(self) -> list[str]:
        return sorted(self._items)


def import_plugin(path: Path, module_name: str) -> None:
    """Import the Python file at ``path`` as module ``module_name`` (once per process)."""
    if module_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RegistryError(f"cannot import plugin file {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise


def discover_plugins(
    root: Path,
    entry_file: str,
    namespace: str,
    registry: Registry,
    include_templates: bool = False,
) -> list[str]:
    """Import ``<root>/<name>/<entry_file>`` for every plugin folder and check that it
    registered itself under the folder name.

    Folders starting with ``.`` are ignored, and folders starting with ``_`` (templates)
    unless ``include_templates`` is set. Returns the plugin names found.
    """
    found = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        name = folder.name
        if name.startswith(".") or (name.startswith("_") and not include_templates):
            continue
        entry = folder / entry_file
        if not entry.is_file():
            continue
        import_plugin(entry, f"{namespace}.{name}")
        if name not in registry:
            raise RegistryError(f"{entry} must register a {registry.kind} named {name!r}")
        found.append(name)
    return found


def make_register(registry: Registry[T], check: Callable[[type], None]) -> Callable:
    """Build a ``@register_xxx("name")`` decorator that validates the class first."""

    def register(name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            check(cls)
            registry.add(name, cls)
            cls.id = name  # type: ignore[attr-defined]
            return cls

        return decorator

    return register
