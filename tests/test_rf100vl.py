"""RF100-VL conversion and loader, on a synthetic two-image dataset."""

import io

import pytest
from PIL import Image

from ovannot.core.dataset import DATASETS, discover_datasets, restrict_to_setting
from ovannot.core.types import Setting
from ovannot.utils.io import read_json, write_json

discover_datasets()


def png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "gray").save(buf, format="PNG")
    return buf.getvalue()


def row(image_id, name, w, h, cats, bboxes):
    return {
        "image": {"bytes": png(w, h), "path": None},
        "image_id": image_id,
        "width": w,
        "height": h,
        "file_name": name,
        "dataset_id": "68",
        "dataset_name": "ufba-425",
        "annotations": {
            "id": [f"68_{i}" for i in range(len(cats))],
            "category_id": cats,
            "dataset_category_id": [f"68_{c}" for c in cats],
            "bbox": bboxes,
            "area": [b[2] * b[3] for b in bboxes],
            "segmentation": [[] for _ in cats],
            "iscrowd": [0 for _ in cats],
        },
        "original_category_ids": cats,
    }


CATS = [{"id": 0, "name": "tooth", "supercategory": "x"}, {"id": 1, "name": "caries"}]


@pytest.fixture
def root(tmp_path, load_script):
    dl = load_script("datasets/rf100vl/download.py")
    for split in ("train", "test"):
        rows = [
            row("68_10", "b.png", 40, 30, [1], [[1.0, 2.0, 10.0, 5.0]]),
            row("68_2", "a.png", 20, 10, [0, 0], [[0.0, 0.0, 5.0, 5.0], [5.0, 5.0, 4.0, 4.0]]),
        ]
        coco, info = dl.to_coco(rows, CATS, tmp_path / "ufba-425" / split)
        write_json(tmp_path / "ufba-425" / split / "_annotations.coco.json", {**coco, "info": info})
    return tmp_path


def test_conversion_orders_images_and_keeps_boxes(root):
    coco = read_json(root / "ufba-425" / "test" / "_annotations.coco.json")
    assert [i["hf_image_id"] for i in coco["images"]] == ["68_2", "68_10"]  # numeric order
    assert coco["categories"] == [{"id": 0, "name": "tooth"}, {"id": 1, "name": "caries"}]
    assert [a["image_id"] for a in coco["annotations"]] == [0, 0, 1]
    assert coco["annotations"][2]["bbox"] == [1.0, 2.0, 10.0, 5.0]
    assert coco["info"]["image_size_mismatch"] == 0
    assert coco["images"][0]["original_file_name"] == "a.png"
    assert (root / "ufba-425" / "test" / "images" / "000000.png").exists()  # opaque name


def test_loader_spec_images_and_ground_truth(root):
    loader = DATASETS.get("rf100vl")({"name": "ufba-425", "root": str(root)})
    spec = loader.spec()
    assert spec.class_names == ["tooth", "caries"]
    assert spec.domain == "Lab Imaging"
    assert spec.splits == {"train": ["train/0", "train/1"], "test": ["test/0", "test/1"]}
    tooth = spec.classes[0].exemplars
    assert [e.image_id for e in tooth] == ["train/0", "train/0"]
    assert tooth[1].bbox_xyxy == (5.0, 5.0, 9.0, 9.0)
    zero = restrict_to_setting(spec, Setting.ZERO_SHOT)
    assert all(not c.exemplars for c in zero.classes) and zero.domain == ""
    few = restrict_to_setting(spec, Setting.FEW_SHOT)
    assert few.classes[0].exemplars and few.domain == ""  # domain is never passed (D4)

    images = loader.images("test")
    assert [(i.image_id, i.width, i.height) for i in images] == [
        ("test/0", 20, 10),
        ("test/1", 40, 30),
    ]
    gt = loader.ground_truth("test")
    assert gt.image_ids == {"test/0": 0, "test/1": 1}


def test_unknown_dataset_is_an_error():
    with pytest.raises(ValueError, match="unknown"):
        DATASETS.get("rf100vl")({"name": "not-a-dataset"})
