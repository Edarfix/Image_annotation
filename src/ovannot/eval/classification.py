"""Task 1 metrics: single-label classification where "none of the classes" is an answer (D2, D9).

Ground truth gives each image one known class, or ``None`` when its class was withheld from
the query set (open-set images, D16). Predictions are :class:`ClassPrediction` objects.

- ``accuracy``: share of images whose decision is right, "none" included.
- ``closed_accuracy``: on images of known classes only, share whose top known class is right,
  ignoring the "none" decision (comparable with closed-set numbers).
- ``none_precision`` / ``none_recall`` / ``none_f1``: the "none" decision as a detector of
  open-set images.
- ``macro_f1``: mean F1 over the known classes and "none".
- Open-set recognition (Miller et al., ECCV 2024, arXiv 2403.16528, §4.1): positives are the
  correct predictions on known images (TPs), negatives the open-set errors (OSEs, every
  open-set image, since its class is not in the query set), ranked by the confidence of the
  top known class. Misclassified known images are left out, as in the paper. ``aupr`` is the
  step-wise average precision; ``p_at_95r`` the precision at the first threshold reaching 95%
  recall; ``r_at_95p`` the largest recall at precision >= 95% (None if never reached).

The confidence of the top known class is ``max(class_scores)`` when the pipeline gives
``class_scores``, else ``score`` if it predicted a class and 0 if it predicted "none".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ovannot.core.dataset import ClassificationGroundTruth
from ovannot.core.types import ImageAnnotation

NONE = "__none__"


def known_confidence(ann: ImageAnnotation) -> tuple[str | None, float]:
    """Top known class and its confidence, whatever the final decision."""
    label = ann.label
    if label is None:
        return None, 0.0
    if label.class_scores:
        name = max(label.class_scores, key=label.class_scores.get)
        return name, float(label.class_scores[name])
    return label.class_name, float(label.score) if label.class_name else 0.0


def open_set_curve(pos: np.ndarray, neg: np.ndarray) -> dict[str, float | None]:
    """AuPR, P@95R and R@95P of positives ``pos`` against negatives ``neg`` (confidences)."""
    if len(pos) == 0:
        return {"aupr": 0.0, "p_at_95r": 0.0, "r_at_95p": None}
    scores = np.concatenate([pos, neg])
    is_pos = np.concatenate([np.ones(len(pos), bool), np.zeros(len(neg), bool)])
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(is_pos[order])
    fp = np.cumsum(~is_pos[order])
    # evaluate only at the last element of each run of tied scores
    last = np.r_[scores[order][1:] != scores[order][:-1], True]
    tp, fp = tp[last], fp[last]
    precision = tp / (tp + fp)
    recall = tp / len(pos)
    recall_steps = np.diff(np.r_[0.0, recall])
    reach = np.where(recall >= 0.95)[0]
    good = precision >= 0.95
    return {
        "aupr": float((precision * recall_steps).sum()),
        "p_at_95r": float(precision[reach[0]]) if len(reach) else 0.0,
        "r_at_95p": float(recall[good].max()) if good.any() else None,
    }


def evaluate_classification(
    gt: ClassificationGroundTruth, annotations: list[ImageAnnotation], class_names: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Metrics of ``annotations`` against ``gt`` (restricted to the annotated images) and the
    predictions list written to ``predictions.json``."""
    truth, pred, pos, neg = [], [], [], []
    predictions = []
    closed_right = closed_n = 0
    for ann in annotations:
        y = gt.labels[ann.image_id]
        decided = ann.label.class_name if ann.label else None
        top, conf = known_confidence(ann)
        truth.append(y or NONE)
        pred.append(decided or NONE)
        if y is None:
            neg.append(conf)
        else:
            closed_n += 1
            closed_right += top == y
            if top == y:
                pos.append(conf)
        predictions.append(
            {
                "image_id": ann.image_id,
                "class_name": decided,
                "score": ann.label.score if ann.label else 0.0,
                "class_scores": ann.label.class_scores if ann.label else {},
                "extras": ann.label.extras if ann.label else {},
            }
        )
    truth_a, pred_a = np.array(truth), np.array(pred)
    labels = [*class_names, NONE]
    per_class = {}
    for name in labels:
        t, p = truth_a == name, pred_a == name
        tp = int((t & p).sum())
        prec = tp / p.sum() if p.sum() else 0.0
        rec = tp / t.sum() if t.sum() else 0.0
        per_class[name] = {
            "precision": prec,
            "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "n": int(t.sum()),
        }
    present = [n for n in labels if per_class[n]["n"]]
    none = per_class[NONE]
    metrics = {
        "accuracy": float((truth_a == pred_a).mean()) if len(truth_a) else 0.0,
        "closed_accuracy": closed_right / closed_n if closed_n else 0.0,
        "macro_f1": float(np.mean([per_class[n]["f1"] for n in present])) if present else 0.0,
        "none_precision": none["precision"],
        "none_recall": none["recall"],
        "none_f1": none["f1"],
        **open_set_curve(np.array(pos), np.array(neg)),
        "n_images": len(annotations),
        "n_open_set": int((truth_a == NONE).sum()),
        "per_class": {("none" if k == NONE else k): v for k, v in per_class.items()},
    }
    return metrics, predictions
