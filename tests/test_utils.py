import jinja2
import pytest

from ovannot.utils.hashing import hash_bytes, hash_file, hash_obj, stable_json
from ovannot.utils.io import atomic_write_text, read_json, write_json
from ovannot.utils.prompts import prompt_variables, render_prompt


def test_stable_json_and_hashes():
    assert stable_json({"b": 1, "a": (1, 2)}) == '{"a":[1,2],"b":1}'
    assert hash_obj({"b": 1, "a": 2}) == hash_obj({"a": 2, "b": 1})
    assert len(hash_obj({"a": 1})) == 16


def test_hash_file_matches_hash_bytes(tmp_path):
    data = b"x" * 3_000_000  # several chunks
    (tmp_path / "f.bin").write_bytes(data)
    assert hash_file(tmp_path / "f.bin", chunk_size=1 << 20) == hash_bytes(data)


def test_atomic_write_creates_parents_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "a" / "b" / "out.json"
    write_json(target, {"x": 1})
    atomic_write_text(target, '{"x": 2}')
    assert read_json(target) == {"x": 2}
    assert [p.name for p in target.parent.iterdir()] == ["out.json"]


def test_prompts_fail_on_missing_variables(tmp_path):
    path = tmp_path / "p.jinja"
    path.write_text("Is this a {{ class_name }}?{% if hint %} Hint: {{ hint }}{% endif %}\n")
    assert prompt_variables(path) == {"class_name", "hint"}
    assert render_prompt(path, class_name="crack", hint="") == "Is this a crack?\n"
    with pytest.raises(jinja2.UndefinedError):
        render_prompt(path, hint="")
