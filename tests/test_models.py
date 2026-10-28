"""Model wrappers against fake HTTP servers: request format, box conversion, cache, errors."""

import json

import httpx
import pytest
from PIL import Image

from ovannot.core.errors import EmptyOutputError, ModelError, ParseError
from ovannot.models.mllm_client import OpenAIMLLM, parse_json_list
from ovannot.models.sam3 import SAM3

QWEN = {
    "kind": "openai_mllm",
    "checkpoint": "Qwen/x",
    "revision": "abc",
    "licence": "apache-2.0",
    "licence_source": "test",
    "source": "test",
    "endpoint": "http://model/v1",
    "served_model_name": "x",
    "coord_format": "qwen3vl_rel1000",
    "generation": {"temperature": 0.0},
    "max_retries": 1,
}


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "img.png"
    Image.new("RGB", (200, 100), "gray").save(path)
    return path


def fake(handler):
    calls = []

    def wrapped(request):
        calls.append(request)
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(wrapped), base_url="http://model/v1"), calls


def chat_answer(text, finish="stop"):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


def test_chat_sends_image_then_text_and_caches(tmp_path, image, monkeypatch):
    monkeypatch.setenv("OVANNOT_CACHE", str(tmp_path / "cache"))
    mllm = OpenAIMLLM("qwen", QWEN)
    mllm._client, calls = fake(lambda r: chat_answer('[{"bbox_2d": [0, 0, 500, 1000]}]'))
    result = mllm.chat([{"role": "user", "content": "find"}], images=[image])
    body = json.loads(calls[0].content)
    content = body["messages"][-1]["content"]
    assert [c["type"] for c in content] == ["image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert body["temperature"] == 0.0 and body["model"] == "x"
    assert (result.prompt_tokens, result.completion_tokens, result.cached) == (11, 7, False)

    again = mllm.chat([{"role": "user", "content": "find"}], images=[image])
    assert again.cached and again.text == result.text and len(calls) == 1
    mllm.chat([{"role": "user", "content": "find"}], images=[image], temperature=0.5)
    assert len(calls) == 2  # generation parameters are part of the key


def test_qwen_boxes_are_converted_from_relative_1000(tmp_path, monkeypatch):
    monkeypatch.setenv("OVANNOT_CACHE", str(tmp_path))
    mllm = OpenAIMLLM("qwen", QWEN)
    text = (
        '```json\n[{"bbox_2d": [100, 200, 500, 1000], "label": "dent"},\n'
        ' {"bbox_2d": [600, 50, 300, 10], "label": "crack"}]\n```'
    )
    assert mllm.boxes_from_text(text, width=200, height=100) == [
        ("dent", (20.0, 20.0, 100.0, 100.0)),
        ("crack", (60.0, 1.0, 120.0, 5.0)),  # inverted corners are reordered
    ]


@pytest.mark.parametrize(
    ("text", "n"),
    [
        ('[{"bbox_2d": [1, 2, 3, 4]}]', 1),
        ('{"bbox_2d": [1, 2, 3, 4]}', 1),  # a single object
        ('[{"bbox_2d": [1, 2, 3, 4], "label": "a"}, {"bbox_2d": [5, 6', 1),  # cut by max_tokens
        ("Here you go:\n```json\n[]\n```", 0),
    ],
)
def test_parse_json_list(text, n):
    assert len(parse_json_list(text)) == n


def test_parse_errors_and_plain_text_answers():
    with pytest.raises(EmptyOutputError):
        parse_json_list("  ")
    assert parse_json_list("There is no object belonging to the provided categories.") == []
    with pytest.raises(ParseError):
        parse_json_list('[{"bbox_2d": [1, 2,')


def test_malformed_lists_keep_the_objects_that_parse():
    text = (
        '[{"bbox_2d": [1, 2, 3, 4], "label": "a"}, {"bbox_2d": [5 6 7 8]},'
        ' {"bbox": [9, 9, 10, 10]}]'
    )
    assert parse_json_list(text) == [
        {"bbox_2d": [1, 2, 3, 4], "label": "a"},
        {"bbox": [9, 9, 10, 10]},
    ]


def test_bbox_key_is_an_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("OVANNOT_CACHE", str(tmp_path))
    mllm = OpenAIMLLM("qwen", QWEN)
    assert mllm.boxes_from_text('[{"bbox": [0, 0, 500, 500], "label": "x"}]', 200, 100) == [
        ("x", (0.0, 0.0, 100.0, 50.0))
    ]


def test_http_errors_become_model_errors(tmp_path, image, monkeypatch):
    monkeypatch.setenv("OVANNOT_CACHE", str(tmp_path))
    mllm = OpenAIMLLM("qwen", QWEN)
    mllm._client, calls = fake(lambda r: httpx.Response(400, text="context too long"))
    with pytest.raises(ModelError, match="400"):
        mllm.chat([{"role": "user", "content": "find"}], images=[image])
    assert len(calls) == 1  # 4xx are not retried


def test_endpoint_env_overrides_config(monkeypatch, tmp_path):
    monkeypatch.setenv("OVANNOT_CACHE", str(tmp_path))
    monkeypatch.setenv("OVANNOT_ENDPOINT_TEST", "http://elsewhere:9/v1")
    mllm = OpenAIMLLM("qwen", {**QWEN, "endpoint_env": "OVANNOT_ENDPOINT_TEST"})
    assert str(mllm._client.base_url).startswith("http://elsewhere:9/v1")


def test_sam3_client(tmp_path, image, monkeypatch):
    monkeypatch.setenv("OVANNOT_CACHE", str(tmp_path))
    cfg = {
        k: v
        for k, v in QWEN.items()
        if k not in ("served_model_name", "coord_format", "generation")
    }
    sam3 = SAM3("sam3", {**cfg, "kind": "sam3_http"})
    answer = {
        "width": 200,
        "height": 100,
        "results": [
            {"phrase": "dent", "boxes": [[1, 2, 30, 40]], "scores": [0.8]},
            {"phrase": "crack", "boxes": [], "scores": []},
        ],
    }
    sam3._client, calls = fake(lambda r: httpx.Response(200, json=answer))
    out = sam3.detect(image, ["dent", "crack"], threshold=0.1)
    assert [(g.phrase, g.boxes, g.scores) for g in out] == [
        ("dent", [(1, 2, 30, 40)], [0.8]),
        ("crack", [], []),
    ]
    body = json.loads(calls[0].content)
    assert body["image_path"] == str(image.resolve()) and body["threshold"] == 0.1
    sam3.detect(image, ["dent", "crack"], threshold=0.1)
    assert len(calls) == 1  # cached
