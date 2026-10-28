"""Sanity tests of the detection evaluation (brief, "Evaluation")."""

import itertools

import pytest

from ovannot.core.dataset import DetectionGroundTruth
from ovannot.core.types import Detection, ImageAnnotation
from ovannot.eval.detection import coco_map, evaluate_detection, f1_at_iou
from ovannot.utils.boxes import xywh_to_xyxy

GT = {
    "images": [{"id": i, "file_name": f"{i}.jpg", "width": 200, "height": 200} for i in range(3)],
    "annotations": [
        {
            "id": 1,
            "image_id": 0,
            "category_id": 0,
            "bbox": [10, 10, 40, 30],
            "area": 1200,
            "iscrowd": 0,
        },
        {
            "id": 2,
            "image_id": 0,
            "category_id": 1,
            "bbox": [100, 50, 60, 80],
            "area": 4800,
            "iscrowd": 0,
        },
        {
            "id": 3,
            "image_id": 1,
            "category_id": 0,
            "bbox": [20, 120, 50, 50],
            "area": 2500,
            "iscrowd": 0,
        },
        {
            "id": 4,
            "image_id": 2,
            "category_id": 1,
            "bbox": [5, 5, 100, 20],
            "area": 2000,
            "iscrowd": 0,
        },
    ],
    "categories": [
        {"id": 0, "name": "crack"},
        {"id": 1, "name": "dent"},
        {"id": 2, "name": "rust"},
    ],
}
NAMES = {0: "crack", 1: "dent", 2: "rust"}
GROUND_TRUTH = DetectionGroundTruth(coco=GT, image_ids={f"test/{i}": i for i in range(3)})


def annotations(shift: float = 0.0, score: float = 0.9) -> list[ImageAnnotation]:
    anns = {f"test/{i}": ImageAnnotation(f"test/{i}") for i in range(3)}
    for a in GT["annotations"]:
        box = tuple(v + shift for v in xywh_to_xyxy(a["bbox"]))
        anns[f"test/{a['image_id']}"].detections.append(
            Detection(box, NAMES[a["category_id"]], score)
        )
    return list(anns.values())


def test_ground_truth_as_predictions_scores_one():
    metrics, predictions = evaluate_detection(GROUND_TRUTH, annotations(), score_threshold=0.5)
    assert metrics["mAP"] == pytest.approx(1.0)
    assert metrics["AP50"] == pytest.approx(1.0)
    assert metrics["f1@0.5"] == pytest.approx(1.0)
    assert metrics["per_class"]["crack"]["AP"] == pytest.approx(1.0)
    assert metrics["per_class"]["rust"]["AP"] is None  # no ground truth
    assert len(predictions) == 4 and "extras" in predictions[0]


def test_empty_predictions_score_zero():
    empty = [ImageAnnotation(f"test/{i}") for i in range(3)]
    metrics, predictions = evaluate_detection(GROUND_TRUTH, empty, score_threshold=0.5)
    assert predictions == []
    assert metrics["mAP"] == 0.0 and metrics["f1@0.5"] == 0.0 and metrics["recall@0.5"] == 0.0


def test_shifted_boxes_lower_the_map_monotonically():
    maps = [evaluate_detection(GROUND_TRUTH, annotations(s), 0.5)[0]["mAP"] for s in (0, 2, 5, 10)]
    assert maps[0] == pytest.approx(1.0)
    assert all(a > b for a, b in itertools.pairwise(maps))


def test_operating_threshold_filters_f1_but_not_map():
    metrics, _ = evaluate_detection(GROUND_TRUTH, annotations(score=0.3), score_threshold=0.5)
    assert metrics["mAP"] == pytest.approx(1.0)
    assert metrics["recall@0.5"] == 0.0


def test_f1_counts_duplicates_and_wrong_classes_as_false_positives():
    anns = annotations()
    anns[0].detections.append(Detection(anns[0].detections[0].bbox_xyxy, "crack", 0.8))  # dup
    anns[1].detections.append(Detection((0, 0, 10, 10), "rust", 0.9))  # no rust here
    f1 = f1_at_iou(GT, anns, GROUND_TRUTH.image_ids, score_threshold=0.5)
    assert f1["recall"] == pytest.approx(1.0)
    assert f1["precision"] == pytest.approx(4 / 6)
    assert f1["per_class"]["rust"]["fp"] == 1


def test_limit_restricts_ground_truth_to_annotated_images():
    metrics, _ = evaluate_detection(GROUND_TRUTH, annotations()[:1], score_threshold=0.5)
    assert metrics["n_images"] == 1 and metrics["n_ground_truth"] == 2
    assert metrics["mAP"] == pytest.approx(1.0)


def test_coco_map_without_ground_truth():
    gt = {**GT, "annotations": []}
    assert (
        coco_map(gt, [{"image_id": 0, "category_id": 0, "bbox": [0, 0, 1, 1], "score": 1}])["mAP"]
        == 0.0
    )
