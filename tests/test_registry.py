import sys
import textwrap
import types
import uuid

import pytest

from ovannot.core.registry import Registry, RegistryError, discover_plugins, make_register


class Base:
    id = ""


@pytest.fixture
def plugin_env(tmp_path):
    """A fresh registry, importable by plugin files as module ``<name>``, and a namespace."""
    registry: Registry[Base] = Registry("thing")
    module_name = f"fake_registry_{uuid.uuid4().hex}"
    module = types.ModuleType(module_name)
    module.Base = Base
    module.register = make_register(registry, lambda cls: None)
    sys.modules[module_name] = module
    yield registry, module_name, f"ns_{uuid.uuid4().hex}", tmp_path
    del sys.modules[module_name]


def write_plugin(root, folder, registered_name, module_name):
    (root / folder).mkdir()
    (root / folder / "plugin.py").write_text(
        textwrap.dedent(
            f"""
            from {module_name} import Base, register

            @register({registered_name!r})
            class Thing(Base):
                pass
            """
        )
    )


def test_discovery_registers_plugins_and_skips_templates(plugin_env):
    registry, module_name, namespace, root = plugin_env
    write_plugin(root, "alpha", "alpha", module_name)
    write_plugin(root, "_template", "_template", module_name)
    (root / "not_a_plugin").mkdir()  # no entry file: ignored

    assert discover_plugins(root, "plugin.py", namespace, registry) == ["alpha"]
    assert registry.get("alpha").id == "alpha"
    assert "_template" not in registry
    # discovering again is a no-op (modules are imported once)
    assert discover_plugins(root, "plugin.py", namespace, registry) == ["alpha"]
    assert discover_plugins(root, "plugin.py", namespace, registry, include_templates=True) == [
        "_template",
        "alpha",
    ]


def test_plugin_must_register_its_folder_name(plugin_env):
    registry, module_name, namespace, root = plugin_env
    write_plugin(root, "beta", "something_else", module_name)
    with pytest.raises(RegistryError, match="must register a thing named 'beta'"):
        discover_plugins(root, "plugin.py", namespace, registry)


def test_duplicate_names_are_rejected_and_unknown_names_listed():
    registry: Registry[Base] = Registry("thing")
    register = make_register(registry, lambda cls: None)

    @register("x")
    class A(Base):
        pass

    with pytest.raises(RegistryError, match="already registered"):

        @register("x")
        class B(Base):
            pass

    with pytest.raises(RegistryError, match=r"registered: x"):
        registry.get("y")
