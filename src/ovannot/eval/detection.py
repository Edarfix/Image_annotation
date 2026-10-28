"""Detection metrics: the official RF100-VL protocol, and precision / recall / F1 at IoU 0.5.

- mAP@[.5:.95], AP50 and AP75 come from pycocotools ``COCOeval`` on boxes with
  ``maxDets = (1, 10, 500)``: RF100-VL raises COCO's 100 detections per image to 500 (paper
  §3.2). Each dataset is scored on its own; averaging over datasets happens in the runner.
- Precision, recall and F1 count the detections scored at least ``score_threshold`` (the
  pipeline's operating point), matched greedily by score to unmatched ground truth of the same
  class with IoU >= 0.5.

Frozen after M1: any change needs an entry in ``CHANGELOG.md`` and a re-run of the pipelines.
"""

from __future__ import annotations

import contextlib
import io
from collections import defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ovannot.core.coco_io import detections_to_coco_results, index_coco
from ovannot.core.dataset import DetectionGroundTruth
from ovannot.core.types import ImageAnnotation
from ovannot.utils.boxes import iou_matrix, xywh_to_xyxy

MAX_DETS = (1, 10, 500)
F1_IOU = 0.5


def restrict_coco(coco: Mapping[str, Any], coco_image_ids: Iterable[int]) -> dict[str, Any]:
    """The COCO dict restricted to ``coco_image_ids`` (e.g. a run with ``--limit``)."""
    keep = set(coco_image_ids)
    return {
        **coco,
        "images": [img for img in coco["images"] if img["id"] in keep],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in keep],
    }


def coco_map(gt_coco: Mapping[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    """COCO box AP with RF100-VL's ``maxDets``; ``per_class_AP`` is None for classes without
    ground truth in these images."""
    names = {c["id"]: c["name"] for c in gt_coco["categories"]}
    has_gt = {a["category_id"] for a in gt_coco["annotations"]}
    if not results or not gt_coco["annotations"]:
        per_class = {n: (0.0 if cid in has_gt else None) for cid, n in names.items()}
        return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "per_class_AP": per_class}
    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints its progress
        gt = COCO()
        gt.dataset = deepcopy(dict(gt_coco))
        gt.createIndex()
        dt = gt.loadRes(deepcopy(results))
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = list(MAX_DETS)
        ev.evaluate()
        ev.accumulate()
    # Not ev.summarize(): its mAP uses maxDets=100 whatever params.maxDets says, which gives -1
    # with (1, 10, 500). Read the precision array at maxDets=500 for every number instead, as
    # detectron2's COCOevalMaxDets does.
    # precision: [IoU thresholds, recall points, classes, area ranges, maxDets]
    precision = ev.eval["precision"][:, :, :, 0, -1]
    iou = ev.params.iouThrs

    def mean_ap(p: np.ndarray) -> float:
        return float(p[p > -1].mean()) if (p > -1).any() else 0.0

    per_class = {
        names[cid]: mean_ap(precision[:, :, k]) if cid in has_gt else None
        for k, cid in enumerate(ev.params.catIds)
    }
    return {
        "mAP": mean_ap(precision),
        "AP50": mean_ap(precision[np.isclose(iou, 0.5)]),
        "AP75": mean_ap(precision[np.isclose(iou, 0.75)]),
        "per_class_AP": per_class,
    }


def f1_at_iou(
    gt_coco: Mapping[str, Any],
    annotations: Iterable[ImageAnnotation],
    image_ids: Mapping[str, int],
    score_threshold: float,
    iou_threshold: float = F1_IOU,
) -> dict[str, Any]:
    """Micro-averaged and per-class precision, recall and F1 at the operating threshold."""
    index = index_coco(gt_coco)
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    n_gt: dict[str, int] = defaultdict(int)
    for anns in index.annotations_by_image.values():
        for a in anns:
            n_gt[index.category_names[a["category_id"]]] += 1
    for ann in annotations:
        gts = index.annotations_by_image.get(image_ids[ann.image_id], [])
        for name in {d.class_name for d in ann.detections}:
            dets = sorted(
                (d for d in ann.detections if d.class_name == name and d.score >= score_threshold),
                key=lambda d: -d.score,
            )
            gt_boxes = [
                xywh_to_xyxy(a["bbox"])
                for a in gts
                if index.category_names[a["category_id"]] == name
            ]
            if not dets:
                continue
            if not gt_boxes:
                fp[name] += len(dets)
                continue
            ious = iou_matrix([d.bbox_xyxy for d in dets], gt_boxes)
            matched = np.zeros(len(gt_boxes), dtype=bool)
            for i in range(len(dets)):
                cand = np.where(~matched & (ious[i] >= iou_threshold))[0]
                if len(cand):
                    matched[cand[np.argmax(ious[i, cand])]] = True
                    tp[name] += 1
                else:
                    fp[name] += 1

    def prf(t: int, f: int, g: int) -> dict[str, float]:
        p = t / (t + f) if t + f else 0.0
        r = t / g if g else 0.0
        return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}

    names = [c["name"] for c in gt_coco["categories"]]
    return {
        **prf(sum(tp.values()), sum(fp.values()), sum(n_gt.values())),
        "per_class": {
            n: {**prf(tp[n], fp[n], n_gt[n]), "n_gt": n_gt[n], "tp": tp[n], "fp": fp[n]}
            for n in names
        },
    }


def evaluate_detection(
    gt: DetectionGroundTruth,
    annotations: list[ImageAnnotation],
    score_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Metrics of ``annotations`` against ``gt``, restricted to the annotated images.

    Returns the metrics and the COCO results list (the run's ``predictions.json``).
    """
    image_ids = {a.image_id: gt.image_ids[a.image_id] for a in annotations}
    coco = restrict_coco(gt.coco, image_ids.values())
    category_ids = {c["name"]: c["id"] for c in coco["categories"]}
    results = detections_to_coco_results(annotations, gt.image_ids, category_ids)
    ap = coco_map(coco, [{k: v for k, v in r.items() if k != "extras"} for r in results])
    f1 = f1_at_iou(coco, annotations, gt.image_ids, score_threshold)
    per_class = {
        name: {"AP": ap["per_class_AP"][name], **f1["per_class"][name]} for name in category_ids
    }
    metrics = {
        "mAP": ap["mAP"],
        "AP50": ap["AP50"],
        "AP75": ap["AP75"],
        "precision@0.5": f1["precision"],
        "recall@0.5": f1["recall"],
        "f1@0.5": f1["f1"],
        "score_threshold": score_threshold,
        "n_images": len(annotations),
        "n_ground_truth": len(coco["annotations"]),
        "n_predictions": len(results),
        "per_class": per_class,
    }
    return metrics, results
