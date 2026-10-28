"""SigLIP / SigLIP 2 image-text scorer, reached over HTTP (server: ``serving/siglip_server.py``).

``logits(image, texts)`` returns ``logit_scale * <image, text> + logit_bias`` per text, the
model's own zero-shot logits (``sigmoid`` gives SigLIP's independent probabilities). Text and
image embeddings are cached on disk; within a process, text embeddings are also kept in
memory, so a dataset's class prompts are encoded once.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from ovannot.core.errors import ModelError
from ovannot.models.base import ModelConfig, http_client, post_json
from ovannot.models.cache import CallCache
from ovannot.models.hub import register_model_kind
from ovannot.utils.hashing import hash_file


class SigLIPConfig(ModelConfig):
    pass


@register_model_kind("siglip_http")
class SigLIP:
    def __init__(self, name: str, cfg: dict[str, Any]) -> None:
        self.name = name
        self.cfg = SigLIPConfig.model_validate(cfg)
        self.cache = CallCache(name, enabled=self.cfg.cache)
        self._client = http_client(self.cfg)
        self._text: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._scale_bias: tuple[float, float] | None = None

    def scale_bias(self) -> tuple[float, float]:
        if self._scale_bias is None:
            key = self.cache.key(checkpoint=self.cfg.checkpoint, revision=self.cfg.revision)
            hit = self.cache.get(key)
            if hit is None:
                try:
                    hit = self._client.get("/health").json()
                except httpx.HTTPError as exc:
                    raise ModelError(f"{self.name}: /health failed: {exc}") from exc
                self.cache.put(key, hit)
            self._scale_bias = (float(hit["logit_scale"]), float(hit["logit_bias"]))
        return self._scale_bias

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        with self._lock:
            missing = [t for t in dict.fromkeys(texts) if t not in self._text]
        if missing:
            key = self.cache.key(
                checkpoint=self.cfg.checkpoint, revision=self.cfg.revision, texts=missing
            )
            emb = self.cache.get(key)
            if emb is None:
                emb = post_json(
                    self._client, "/embed_text", {"texts": missing}, self.cfg.max_retries
                )["embeddings"]
                self.cache.put(key, emb)
            with self._lock:
                self._text.update(zip(missing, np.asarray(emb, dtype=np.float32), strict=True))
        return np.stack([self._text[t] for t in texts])

    def embed_image(self, image: str | Path) -> np.ndarray:
        key = self.cache.key(
            checkpoint=self.cfg.checkpoint, revision=self.cfg.revision, image=hash_file(image)
        )
        emb = self.cache.get(key)
        if emb is None:
            emb = post_json(
                self._client,
                "/embed_image",
                {"image_paths": [str(Path(image).resolve())]},
                self.cfg.max_retries,
            )["embeddings"][0]
            self.cache.put(key, emb)
        return np.asarray(emb, dtype=np.float32)

    def logits(self, image: str | Path, texts: Sequence[str]) -> np.ndarray:
        scale, bias = self.scale_bias()
        return scale * (self.embed_texts(texts) @ self.embed_image(image)) + bias
