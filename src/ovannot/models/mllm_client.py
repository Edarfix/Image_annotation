"""MLLMs served behind an OpenAI-compatible chat endpoint (vLLM), with a disk cache.

``chat`` sends text messages plus images (attached to the last user message) and returns the
answer with its token counts. ``boxes_from_text`` converts the boxes of a grounding answer
from the model's coordinate format to absolute pixels: it is the only place where that format
is known (``coord_format`` in the model config).

Coordinate formats:

- ``qwen3vl_rel1000``: JSON objects ``{"bbox_2d": [x1, y1, x2, y2], "label": ...}`` with
  coordinates relative to the image, in [0, 1000]. Checked 2026-10-28 in the official
  cookbook ``QwenLM/Qwen3-VL/cookbooks/2d_grounding.ipynb`` (``bbox_2d[0] / 1000 * width``) and
  the Qwen3-VL report §3.2.4 (normalised to [0, 1000]).
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from ovannot.core.errors import EmptyOutputError, ParseError
from ovannot.core.types import BBox
from ovannot.models.base import ModelConfig, http_client, post_json
from ovannot.models.cache import CallCache
from ovannot.models.hub import register_model_kind
from ovannot.utils.hashing import hash_file


class MLLMConfig(ModelConfig):
    served_model_name: str
    coord_format: Literal["qwen3vl_rel1000"] | None = None
    generation: dict[str, Any] = Field(default_factory=dict)  # e.g. temperature, max_tokens


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    logprobs: list[dict[str, Any]] | None = None
    cached: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


@register_model_kind("openai_mllm")
class OpenAIMLLM:
    def __init__(self, name: str, cfg: dict[str, Any]) -> None:
        self.name = name
        self.cfg = MLLMConfig.model_validate(cfg)
        self.cache = CallCache(name, enabled=self.cfg.cache)
        self._client = http_client(self.cfg)

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        images: Sequence[str | Path] = (),
        json_schema: dict | None = None,
        logprobs: bool = False,
        **generation: Any,
    ) -> ChatResult:
        """One chat completion. ``messages`` are ``{"role", "content"}`` dicts with text
        content; ``images`` go before the text of the last message, which must be a user one."""
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("the last message must come from the user")
        gen = {**self.cfg.generation, **generation}
        key = self.cache.key(
            checkpoint=self.cfg.checkpoint,
            revision=self.cfg.revision,
            messages=list(messages),
            images=[hash_file(p) for p in images],
            json_schema=json_schema,
            logprobs=logprobs,
            generation=gen,
        )
        hit = self.cache.get(key)
        if hit is not None:
            return ChatResult(**{**hit, "cached": True})

        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": _data_url(p)}} for p in images
        ]
        content.append({"type": "text", "text": messages[-1]["content"]})
        body: dict[str, Any] = {
            "model": self.cfg.served_model_name,
            "messages": [*messages[:-1], {"role": "user", "content": content}],
            **gen,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": json_schema},
            }
        if logprobs:
            body.update(logprobs=True, top_logprobs=gen.get("top_logprobs", 5))
        answer = post_json(self._client, "/chat/completions", body, self.cfg.max_retries)
        choice = answer["choices"][0]
        usage = answer.get("usage") or {}
        result = ChatResult(
            text=choice["message"].get("content") or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason"),
            logprobs=(choice.get("logprobs") or {}).get("content") if logprobs else None,
        )
        self.cache.put(key, {k: v for k, v in result.__dict__.items() if k != "cached"})
        return result

    def boxes_from_text(self, text: str, width: int, height: int) -> list[tuple[str, BBox]]:
        """``(label, xyxy pixel box)`` pairs from a grounding answer, in the order given.

        Raises :class:`ParseError` if the text holds no parsable JSON. Boxes are not clipped or
        validated here; that is the caller's policy.
        """
        if self.cfg.coord_format != "qwen3vl_rel1000":
            raise ValueError(f"model {self.name!r} declares no supported coord_format")
        out = []
        for item in parse_json_list(text):
            if not isinstance(item, dict):
                continue
            # Qwen3-VL sometimes writes "bbox" (seen on the dev subset, 2026-10-28)
            values = item.get("bbox_2d", item.get("bbox"))
            if not (isinstance(values, list) and len(values) == 4):
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in values)
            except (TypeError, ValueError):
                continue
            x1, x2 = sorted((x1 / 1000 * width, x2 / 1000 * width))
            y1, y2 = sorted((y1 / 1000 * height, y2 / 1000 * height))
            out.append((str(item.get("label", "")), (x1, y1, x2, y2)))
        return out


_FENCE = re.compile(r"```(?:json)?\s*(.*?)(?:```|$)", re.S)


_OBJECT = re.compile(r"\{[^{}]*\}")


def parse_json_list(text: str) -> list[Any]:
    """A JSON list from a model answer.

    - Markdown fences are stripped.
    - An answer without any JSON bracket (e.g. "There is no object of these categories") is
      an empty list: the model found nothing.
    - If the list does not parse (cut by the token limit, or a malformed object), every flat
      ``{...}`` object that parses on its own is kept, in order.
    """
    if not text.strip():
        raise EmptyOutputError("empty answer")
    fenced = _FENCE.search(text)
    body = (fenced.group(1) if fenced else text).strip()
    if "[" not in body and "{" not in body:
        return []
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        data = []
        for match in _OBJECT.finditer(body):
            try:
                data.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue
        if not data:
            raise ParseError(f"unparsable JSON ({exc}): {text[:200]!r}") from None
    return data if isinstance(data, list) else [data]


def _data_url(path: str | Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode('ascii')}"
