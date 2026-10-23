import pytest

from ovannot.core.config import (
    ConfigError,
    apply_overrides,
    config_hash,
    deep_merge,
    dump_yaml,
    load_config,
)


def test_deep_merge_merges_dicts_and_replaces_other_values():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    merged = deep_merge(base, {"a": {"y": [3]}, "c": 2})
    assert merged == {"a": {"x": 1, "y": [3]}, "b": 1, "c": 2}
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": 1}  # inputs are not modified


def test_overrides_parse_yaml_values_and_create_keys():
    cfg = apply_overrides({"a": {"b": 1}}, ["a.b=0.5", "a.c=[1, 2]", "d.e=true", "f=text"])
    assert cfg == {"a": {"b": 0.5, "c": [1, 2]}, "d": {"e": True}, "f": "text"}
    with pytest.raises(ConfigError):
        apply_overrides({}, ["no_equal_sign"])
    with pytest.raises(ConfigError):
        apply_overrides({"a": 1}, ["a.b=2"])


def test_extends_chain_is_merged_in_order(tmp_path):
    (tmp_path / "models").mkdir()
    dump_yaml({"model": {"name": "m", "temperature": 0.0}, "seed": 1}, tmp_path / "models/m.yaml")
    dump_yaml({"seed": 2, "workers": 4}, tmp_path / "runner.yaml")
    dump_yaml(
        {"extends": ["models/m.yaml", "runner.yaml"], "model": {"temperature": 0.7}},
        tmp_path / "exp.yaml",
    )
    cfg = load_config(tmp_path / "exp.yaml", ["workers=8"])
    assert cfg == {"model": {"name": "m", "temperature": 0.7}, "seed": 2, "workers": 8}


def test_extends_cycle_is_an_error(tmp_path):
    dump_yaml({"extends": "b.yaml"}, tmp_path / "a.yaml")
    dump_yaml({"extends": "a.yaml"}, tmp_path / "b.yaml")
    with pytest.raises(ConfigError, match="cyclic"):
        load_config(tmp_path / "a.yaml")


def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": {"c": 2, "d": 3}}) == config_hash(
        {"b": {"d": 3, "c": 2}, "a": 1}
    )
    assert config_hash({"a": 1}) != config_hash({"a": 2})
