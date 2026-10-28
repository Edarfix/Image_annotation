"""Proposal stages: turn a grounder's answer into :class:`Detection` objects of dataset classes.

Shared by the detection pipelines (p00 SAM 3, p01 MLLM; p02+ build on them).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from ovannot.core.errors import InvalidBoxError
from ovannot.core.types import BBox, Detection, ImageAnnotation, ImageRecord
from ovannot.utils.boxes import clip_xyxy, is_valid_xyxy


def clean_boxes(
    boxes: Iterable[tuple[BBox, float]], img: ImageRecord
) -> tuple[list[tuple[BBox, float]], int]:
    """Clip boxes to the image and drop the invalid ones; returns kept boxes and the count
    dropped. A box is invalid when it is not finite or has no area after clipping."""
    kept, dropped = [], 0
    for box, score in boxes:
        clipped = clip_xyxy(box, img.width, img.height) if is_finite(box) else box
        if is_valid_xyxy(clipped, img.width, img.height):
            kept.append((clipped, score))
        else:
            dropped += 1
    return kept, dropped


def is_finite(box: Sequence[float]) -> bool:
    return len(box) == 4 and all(v == v and abs(v) != float("inf") for v in box)


def top_k(detections: list[Detection], k: int) -> list[Detection]:
    """The ``k`` best detections by score (stable: ties keep their order)."""
    return sorted(detections, key=lambda d: -d.score)[:k]


def mllm_grounding(
    mllm,
    img: ImageRecord,
    prompt: str,
    class_names: Sequence[str],
    ann: ImageAnnotation,
    score: float = 1.0,
) -> list[Detection]:
    """Ask ``mllm`` to ground the classes in ``img``; keep the boxes whose label is a class.

    Labels match class names exactly, else case-, underscore- and space-insensitively; other
    labels are dropped. MLLMs give no confidence: every box gets ``score`` (1.0, as in the
    RF100-VL and Qwen3-VL evaluations). Each detection records in ``extras`` whether the answer
    was cut by the token limit. Raises :class:`InvalidBoxError` if the answer had boxes of
    known classes and all were invalid.
    """
    result = mllm.chat([{"role": "user", "content": prompt}], images=[img.path])
    ann.record_usage(
        mllm.name, prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens
    )
    by_key = {_norm(n): n for n in class_names}
    named = [
        (name, box)
        for label, box in mllm.boxes_from_text(result.text, img.width, img.height)
        if (name := label if label in class_names else by_key.get(_norm(label)))
    ]
    detections = []
    for name, box in named:
        kept, _ = clean_boxes([(box, score)], img)
        detections += [
            Detection(
                b,
                name,
                s,
                provenance=[mllm.name],
                extras={"truncated": result.finish_reason == "length"},
            )
            for b, s in kept
        ]
    if named and not detections:
        raise InvalidBoxError(f"all {len(named)} boxes invalid")
    return detections


def sam3_grounding(
    sam3,
    img: ImageRecord,
    phrases: Mapping[str, Sequence[str]],
    threshold: float,
    ann: ImageAnnotation,
) -> list[Detection]:
    """Prompt SAM 3 with each phrase of each class; every box keeps SAM 3's score and records
    its phrase in ``extras``. ``phrases`` maps class names to their phrases."""
    flat = [(name, p) for name, ps in phrases.items() for p in ps]
    groundings = sam3.detect(img.path, [p for _, p in flat], threshold)
    ann.record_usage(sam3.name, calls=1)
    detections = []
    for (name, phrase), g in zip(flat, groundings, strict=True):
        kept, _ = clean_boxes(zip(g.boxes, g.scores, strict=True), img)
        detections += [
            Detection(box, name, float(s), provenance=[sam3.name], extras={"phrase": phrase})
            for box, s in kept
        ]
    return detections


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").split())
