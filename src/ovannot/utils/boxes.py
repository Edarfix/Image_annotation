"""Box geometry in absolute-pixel ``(x_min, y_min, x_max, y_max)`` coordinates.

COCO files store ``(x, y, width, height)``; convert at the boundary with :func:`xywh_to_xyxy`
and :func:`xyxy_to_xywh` and use xyxy everywhere else.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ovannot.core.types import BBox


def xywh_to_xyxy(box: Sequence[float]) -> BBox:
    x, y, w, h = (float(v) for v in box)
    return (x, y, x + w, y + h)


def xyxy_to_xywh(box: Sequence[float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in box)
    return (x0, y0, x1 - x0, y1 - y0)


def box_area(box: Sequence[float]) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def clip_xyxy(box: Sequence[float], width: float, height: float) -> BBox:
    x0, y0, x1, y1 = box
    return (
        min(max(float(x0), 0.0), width),
        min(max(float(y0), 0.0), height),
        min(max(float(x1), 0.0), width),
        min(max(float(y1), 0.0), height),
    )


def is_valid_xyxy(
    box: Sequence[float],
    width: float | None = None,
    height: float | None = None,
    tol: float = 1.0,
) -> bool:
    """True if the box has 4 finite coordinates, a positive area, and (when the image size is
    given) lies inside the image up to ``tol`` pixels."""
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return False
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return False
    if width is not None and (x0 < -tol or x1 > width + tol):
        return False
    return not (height is not None and (y0 < -tol or y1 > height + tol))


def expand_xyxy(box: Sequence[float], margin: float, width: float, height: float) -> BBox:
    """Grow each side by ``margin`` times the box size (e.g. 0.25), clipped to the image.

    Used to crop a candidate with some context before showing it to a verifier.
    """
    x0, y0, x1, y1 = box
    dx, dy = margin * (x1 - x0), margin * (y1 - y0)
    return clip_xyxy((x0 - dx, y0 - dy, x1 + dx, y1 + dy), width, height)


def iou_matrix(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Pairwise IoU between boxes ``a`` (N, 4) and ``b`` (M, 4), as an (N, M) array."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nms(boxes: ArrayLike, scores: ArrayLike, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression; returns the kept indices, best score first.

    A box is suppressed when its IoU with an already kept box is strictly above
    ``iou_threshold``. Ties in score keep the lower index first (stable sort).
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    ious = iou_matrix(boxes, boxes)
    suppressed = np.zeros(len(boxes), dtype=bool)
    keep: list[int] = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(int(i))
        suppressed |= ious[i] > iou_threshold
    return keep
