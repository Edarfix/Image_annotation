"""HTTP server for SigLIP / SigLIP 2 embeddings, run inside the vLLM container like SAM 3.

API (JSON):

- ``GET /health`` -> ``{"status", "checkpoint", "logit_scale", "logit_bias"}``
- ``POST /embed_text`` ``{"texts": [str]}`` -> ``{"embeddings": [[float]]}``
- ``POST /embed_image`` ``{"image_paths": [str]}`` -> ``{"embeddings": [[float]]}``

Embeddings are L2-normalised ``pooler_output`` of ``get_text_features`` /
``get_image_features``, so ``logit = logit_scale * <image, text> + logit_bias`` reproduces the
model's ``logits_per_image`` (checked on 2026-10-28 with transformers 5.13 and
google/siglip2-so400m-patch14-384). Texts are padded to ``max_length=64``, as SigLIP was
trained. Launched by ``serving/launch_siglip.sh``.
"""

from __future__ import annotations

import argparse
import threading

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps
from pydantic import BaseModel
from transformers import AutoModel, AutoProcessor

BATCH = 256


class TextRequest(BaseModel):
    texts: list[str]


class ImageRequest(BaseModel):
    image_paths: list[str]


def _normalise(out) -> torch.Tensor:
    x = out if torch.is_tensor(out) else out.pooler_output
    return torch.nn.functional.normalize(x.float(), dim=-1)


def build_app(checkpoint: str, device: str) -> FastAPI:
    model = AutoModel.from_pretrained(checkpoint).to(device).eval()
    processor = AutoProcessor.from_pretrained(checkpoint)
    lock = threading.Lock()
    app = FastAPI()

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "checkpoint": checkpoint,
            "logit_scale": float(model.logit_scale.exp()),
            "logit_bias": float(model.logit_bias),
        }

    @app.post("/embed_text")
    def embed_text(req: TextRequest) -> dict:
        out = []
        with lock, torch.inference_mode():
            for i in range(0, len(req.texts), BATCH):
                batch = processor(
                    text=req.texts[i : i + BATCH],
                    padding="max_length",
                    max_length=64,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                out += _normalise(model.get_text_features(input_ids=batch["input_ids"])).tolist()
        return {"embeddings": out}

    @app.post("/embed_image")
    def embed_image(req: ImageRequest) -> dict:
        try:
            images = []
            for path in req.image_paths:
                with Image.open(path) as im:
                    images.append(ImageOps.exif_transpose(im).convert("RGB"))
        except OSError as exc:
            raise HTTPException(status_code=422, detail=f"cannot read image: {exc}") from exc
        with lock, torch.inference_mode():
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            out = _normalise(model.get_image_features(pixel_values=pixels)).tolist()
        return {"embeddings": out}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="local snapshot folder")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    uvicorn.run(build_app(args.checkpoint, args.device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
