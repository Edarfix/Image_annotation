"""Reading COCO ground truth and writing COCO-format predictions.

COCO stores boxes as ``[x, y, width, height]`` and uses integer image and category ids; the
rest of the code uses xyxy boxes, string image ids and class names. The conversion happens
here and nowhere else.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ovannot.core.types import Detection, ImageAnnotation, ImageRecord
from ovannot.utils.boxes import xywh_to_xyxy, xyxy_to_xywh
from ovannot.utils.io import read_json


@dataclass
class CocoIndex:
    images: dict[int, dict[str, Any]]
    annotations_by_image: dict[int, list[dict[str, Any]]]
    category_names: dict[int, str]  # category id -> name

    @property
    def category_ids(self) -> dict[str, int]:
        return {name: cid for cid, name in self.category_names.items()}


def load_coco(path: str | Path) -> dict[str, Any]:
    coco = read_json(path)
    for key in ("images", "annotations", "categories"):
        if key not in coco:
            raise ValueError(f"{path}: not a COCO detection file (missing {key!r})")
    return coco


def index_coco(coco: Mapping[str, Any]) -> CocoIndex:
    names = [c["name"] for c in coco["categories"]]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate category names {duplicates}: class names must be unique")
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)
    return CocoIndex(
        images={img["id"]: img for img in coco["images"]},
        annotations_by_image=dict(by_image),
        category_names={c["id"]: c["name"] for c in coco["categories"]},
    )


def image_records(
    coco: Mapping[str, Any], image_dir: str | Path, id_prefix: str = ""
) -> list[ImageRecord]:
    """One :class:`ImageRecord` per COCO image, sorted by COCO id.

    ``image_id`` is ``f"{id_prefix}{coco_id}"``; use a split prefix (e.g. ``"test/"``) when
    COCO ids restart at 0 in every split, so that ids stay unique within a dataset.
    """
    image_dir = Path(image_dir)
    return [
        ImageRecord(
            image_id=f"{id_prefix}{img['id']}",
            path=image_dir / img["file_name"],
            width=int(img["width"]),
            height=int(img["height"]),
        )
        for img in sorted(coco["images"], key=lambda i: i["id"])
    ]


def detections_to_coco_results(
    annotations: Iterable[ImageAnnotation],
    image_ids: Mapping[str, int],
    category_ids: Mapping[str, int],
    with_extras: bool = True,
) -> list[dict[str, Any]]:
    """COCO results list (one dict per detection), ready for ``COCO.loadRes``.

    ``image_ids`` maps :attr:`ImageRecord.image_id` to COCO image ids and ``category_ids``
    maps class names to COCO category ids. Provenance and extras go under ``"extras"``.
    """
    results = []
    for ann in annotations:
        coco_image_id = image_ids[ann.image_id]
        for det in ann.detections:
            if det.class_name not in category_ids:
                raise KeyError(
                    f"image {ann.image_id}: predicted class {det.class_name!r} is not a "
                    f"dataset class ({sorted(category_ids)})"
                )
            result: dict[str, Any] = {
                "image_id": coco_image_id,
                "category_id": category_ids[det.class_name],
                "bbox": [round(v, 2) for v in xyxy_to_xywh(det.bbox_xyxy)],
                "score": float(det.score),
            }
            if det.mask_rle is not None:
                result["segmentation"] = det.mask_rle
            if with_extras:
                result["extras"] = {"provenance": det.provenance, **det.extras}
            results.append(result)
    return results


def coco_results_to_detections(
    results: Iterable[Mapping[str, Any]],
    category_names: Mapping[int, str],
    image_ids: Mapping[str, int],
) -> dict[str, list[Detection]]:
    """Inverse of :func:`detections_to_coco_results`, keyed by ``ImageRecord.image_id``."""
    by_coco_id = {coco_id: image_id for image_id, coco_id in image_ids.items()}
    out: dict[str, list[Detection]] = defaultdict(list)
    for r in results:
        extras = dict(r.get("extras", {}))
        provenance = list(extras.pop("provenance", []))
        out[by_coco_id[r["image_id"]]].append(
            Detection(
                bbox_xyxy=xywh_to_xyxy(r["bbox"]),
                class_name=category_names[r["category_id"]],
                score=float(r["score"]),
                mask_rle=r.get("segmentation"),
                provenance=provenance,
                extras=extras,
            )
        )
    return dict(out)
