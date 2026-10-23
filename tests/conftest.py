import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def import_script(relpath: str):
    """Import a repository script (not part of the package) by path, once per session."""
    path = ROOT / relpath
    name = f"_script_{path.stem}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses look the module up while it executes
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture(scope="session")
def load_script():
    return import_script
