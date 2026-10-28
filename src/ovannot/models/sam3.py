"""SAM 3 text-prompted grounder, reached over HTTP (server: ``serving/sam3_server.py``).

``detect`` returns, for each phrase, the boxes (absolute pixel xyxy, as the server returns
them) and scores above ``threshold``. Calls are cached on disk like MLLM calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ovannot.core.types import BBox
from ovannot.models.base import ModelConfig, http_client, post_json
from ovannot.models.cache import CallCache
from ovannot.models.hub import register_model_kind
from ovannot.utils.hashing import hash_file


class SAM3Config(ModelConfig):
    mask_threshold: float = 0.5


@dataclass
class Grounding:
    phrase: str
    boxes: list[BBox]
    scores: list[float]


@register_model_kind("sam3_http")
class SAM3:
    def __init__(self, name: str, cfg: dict[str, Any]) -> None:
        self.name = name
        self.cfg = SAM3Config.model_validate(cfg)
        self.cache = CallCache(name, enabled=self.cfg.cache)
        self._client = http_client(self.cfg)

    def detect(
        self, image: str | Path, phrases: Sequence[str], threshold: float
    ) -> list[Grounding]:
        key = self.cache.key(
            checkpoint=self.cfg.checkpoint,
            revision=self.cfg.revision,
            image=hash_file(image),
            phrases=list(phrases),
            threshold=threshold,
            mask_threshold=self.cfg.mask_threshold,
        )
        answer = self.cache.get(key)
        if answer is None:
            answer = post_json(
                self._client,
                "/detect",
                {
                    "image_path": str(Path(image).resolve()),
                    "phrases": list(phrases),
                    "threshold": threshold,
                    "mask_threshold": self.cfg.mask_threshold,
                },
                self.cfg.max_retries,
            )
            self.cache.put(key, answer)
        return [
            Grounding(
                phrase=r["phrase"],
                boxes=[(b[0], b[1], b[2], b[3]) for b in r["boxes"]],
                scores=[float(s) for s in r["scores"]],
            )
            for r in answer["results"]
        ]
