"""HTTP server for SAM 3 text-prompted detection, run inside the vLLM container (torch and
transformers come from there, so the ovannot environment stays light, D11).

API (JSON):

- ``GET /health`` -> ``{"status": "ok", "checkpoint": ...}``
- ``POST /detect`` with ``{"image_path": str, "phrases": [str], "threshold": float}`` ->
  ``{"width", "height", "results": [{"phrase", "boxes": [[x0, y0, x1, y1]], "scores": []}]}``.
  Boxes are absolute pixels of the original image, from ``post_process_instance_segmentation``
  (transformers 5.13, checked 2026-10-28 in the container). The image is encoded once and its
  vision features reused for every phrase.

The image is read from a path on the shared file system: model traffic stays on the cluster.
Launched by ``serving/launch_sam3.sh``.
"""

from __future__ import annotations

import argparse
import threading

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps
from pydantic import BaseModel
from transformers import Sam3Model, Sam3Processor


class DetectRequest(BaseModel):
    image_path: str
    phrases: list[str]
    threshold: float = 0.05
    mask_threshold: float = 0.5


def build_app(checkpoint: str, device: str) -> FastAPI:
    model = Sam3Model.from_pretrained(checkpoint).to(device).eval()
    processor = Sam3Processor.from_pretrained(checkpoint)
    lock = threading.Lock()  # one GPU, one request at a time
    app = FastAPI()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "checkpoint": checkpoint}

    @app.post("/detect")
    def detect(req: DetectRequest) -> dict:
        try:
            with Image.open(req.image_path) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
        except OSError as exc:
            raise HTTPException(status_code=422, detail=f"cannot read image: {exc}") from exc
        width, height = image.size
        results = []
        with lock, torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt").to(device)
            vision_embeds = model.get_vision_features(pixel_values=inputs.pixel_values)
            for phrase in req.phrases:
                text = processor(text=phrase, return_tensors="pt").to(device)
                outputs = model(vision_embeds=vision_embeds, **text)
                post = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=req.threshold,
                    mask_threshold=req.mask_threshold,
                    target_sizes=[(height, width)],
                )[0]
                results.append(
                    {
                        "phrase": phrase,
                        "boxes": post["boxes"].float().cpu().tolist(),
                        "scores": post["scores"].float().cpu().tolist(),
                    }
                )
        return {"width": width, "height": height, "results": results}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="local snapshot folder")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    uvicorn.run(build_app(args.checkpoint, args.device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
