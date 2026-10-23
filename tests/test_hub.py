import pytest

from ovannot.core.config import dump_yaml
from ovannot.core.registry import RegistryError
from ovannot.models.hub import ModelHub, register_model_kind

created = []


@register_model_kind("test_dummy")
class Dummy:
    def __init__(self, name, cfg):
        self.name, self.cfg = name, cfg
        created.append(name)


def test_models_are_created_lazily_once(tmp_path):
    dump_yaml({"kind": "test_dummy", "endpoint": "http://x"}, tmp_path / "m1.yaml")
    dump_yaml({"kind": "test_dummy"}, tmp_path / "m2.yaml")
    hub = ModelHub.from_dir(tmp_path)
    assert hub.names() == ["m1", "m2"]
    assert created == []
    first = hub.get("m1")
    assert hub.get("m1") is first
    assert first.cfg["endpoint"] == "http://x"
    assert created == ["m1"]


def test_unknown_model_and_missing_kind_are_errors():
    hub = ModelHub({"nokind": {}})
    with pytest.raises(RegistryError, match="unknown model"):
        hub.get("absent")
    with pytest.raises(RegistryError, match="no 'kind'"):
        hub.get("nokind")
