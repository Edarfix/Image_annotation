import pytest

from ovannot.core.coco_io import (
    coco_results_to_detections,
    detections_to_coco_results,
    image_records,
    index_coco,
    load_coco,
)
from ovannot.core.types import Detection, ImageAnnotation
from ovannot.utils.io import write_json

COCO = {
    "images": [
        {"id": 1, "file_name": "b.jpg", "width": 640, "height": 480},
        {"id": 0, "file_name": "a.jpg", "width": 320, "height": 240},
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 0,
            "category_id": 0,
            "bbox": [10, 20, 30, 40],
            "area": 1200,
            "iscrowd": 0,
        },
        {"id": 2, "image_id": 0, "category_id": 1, "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0},
    ],
    "categories": [{"id": 0, "name": "crack"}, {"id": 1, "name": "dent"}],
}


def test_index_and_image_records(tmp_path):
    write_json(tmp_path / "gt.json", COCO)
    index = index_coco(load_coco(tmp_path / "gt.json"))
    assert index.category_ids == {"crack": 0, "dent": 1}
    assert len(index.annotations_by_image[0]) == 2 and 1 not in index.annotations_by_image

    records = image_records(COCO, tmp_path / "images", id_prefix="test/")
    assert [r.image_id for r in records] == ["test/0", "test/1"]  # sorted by COCO id
    assert records[0].path == tmp_path / "images" / "a.jpg"
    assert (records[0].width, records[0].height) == (320, 240)


def test_load_coco_rejects_non_coco(tmp_path):
    write_json(tmp_path / "x.json", {"images": []})
    with pytest.raises(ValueError, match="missing"):
        load_coco(tmp_path / "x.json")


def test_duplicate_category_names_are_rejected():
    coco = {**COCO, "categories": [{"id": 0, "name": "a"}, {"id": 1, "name": "a"}]}
    with pytest.raises(ValueError, match="duplicate"):
        index_coco(coco)


def test_results_round_trip():
    ann = ImageAnnotation(
        "test/0",
        detections=[
            Detection((10.0, 20.0, 40.0, 60.0), "crack", 0.9, provenance=["sam3"], extras={"k": 1}),
            Detection((1.0, 1.0, 2.0, 2.0), "dent", 0.1),
        ],
    )
    image_ids, category_ids = {"test/0": 0}, {"crack": 0, "dent": 1}
    results = detections_to_coco_results([ann], image_ids, category_ids)
    assert results[0] == {
        "image_id": 0,
        "category_id": 0,
        "bbox": [10.0, 20.0, 30.0, 40.0],
        "score": 0.9,
        "extras": {"provenance": ["sam3"], "k": 1},
    }
    assert "extras" not in detections_to_coco_results([ann], image_ids, category_ids, False)[0]

    back = coco_results_to_detections(results, {0: "crack", 1: "dent"}, image_ids)
    assert back == {"test/0": ann.detections}


def test_unknown_predicted_class_is_an_error():
    ann = ImageAnnotation("test/0", detections=[Detection((0, 0, 1, 1), "scratch", 0.5)])
    with pytest.raises(KeyError, match="scratch"):
        detections_to_coco_results([ann], {"test/0": 0}, {"crack": 0})
