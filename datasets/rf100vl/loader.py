"""RF100-VL loader: one dataset of the benchmark, from the COCO folders written by download.py.

Config keys: ``name`` (the dataset, e.g. ``"ufba-425"``) and optionally ``root`` (default
``$OVANNOT_DATA/rf100vl``). Image ids are ``"<split>/<coco id>"``.

Few-shot material (D4, D10): the exemplars are the boxes of the FSOD ``train`` split, which
holds the official 10-shot examples. Annotator instructions are not available yet
(TODO(verify), see README.md), so ``annotator_instructions`` stays ``None``.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from ovannot.core.coco_io import image_records, index_coco, load_coco
from ovannot.core.dataset import DatasetLoader, DetectionGroundTruth, register_dataset
from ovannot.core.types import ClassSpec, DatasetSpec, Exemplar, ImageRecord, Task
from ovannot.paths import data_root
from ovannot.utils.boxes import xywh_to_xyxy
from ovannot.utils.io import read_json

SPLITS = ("train", "valid", "test")
EXEMPLAR_SPLIT = "train"


@cache
def domains() -> dict[str, str]:
    return read_json(Path(__file__).with_name("domains.json"))["domains"]


@register_dataset("rf100vl")
class RF100VL(DatasetLoader):
    task = Task.DETECTION

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.name: str = cfg["name"]
        if self.name not in domains():
            raise ValueError(f"unknown RF100-VL dataset {self.name!r}")
        self.root = Path(cfg.get("root") or data_root() / "rf100vl") / self.name
        self._coco: dict[str, dict] = {}

    def coco(self, split: str) -> dict[str, Any]:
        if split not in self._coco:
            path = self.root / split / "_annotations.coco.json"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing: run datasets/rf100vl/download.py")
            self._coco[split] = load_coco(path)
        return self._coco[split]

    def spec(self) -> DatasetSpec:
        index = index_coco(self.coco(EXEMPLAR_SPLIT))
        exemplars: dict[int, list[Exemplar]] = {cid: [] for cid in index.category_names}
        for coco_id, anns in sorted(index.annotations_by_image.items()):
            img = index.images[coco_id]
            for a in anns:
                exemplars[a["category_id"]].append(
                    Exemplar(
                        image_id=f"{EXEMPLAR_SPLIT}/{coco_id}",
                        path=self.root / EXEMPLAR_SPLIT / "images" / img["file_name"],
                        bbox_xyxy=xywh_to_xyxy(a["bbox"]),
                    )
                )
        classes = [
            ClassSpec(name=name, exemplars=exemplars[cid])
            for cid, name in sorted(index.category_names.items())
        ]
        test_names = [c["name"] for c in sorted(self.coco("test")["categories"], key=_id)]
        if [c.name for c in classes] != test_names:
            raise ValueError(f"{self.name}: train and test categories differ")
        return DatasetSpec(
            name=self.name,
            task=Task.DETECTION,
            classes=classes,
            splits={s: [r.image_id for r in self.images(s)] for s in SPLITS if self._has(s)},
            annotator_instructions=None,
            domain=domains()[self.name],
        )

    def images(self, split: str) -> list[ImageRecord]:
        return image_records(self.coco(split), self.root / split / "images", id_prefix=f"{split}/")

    def ground_truth(self, split: str) -> DetectionGroundTruth:
        coco = self.coco(split)
        return DetectionGroundTruth(
            coco=coco, image_ids={f"{split}/{img['id']}": img["id"] for img in coco["images"]}
        )

    def _has(self, split: str) -> bool:
        return (self.root / split / "_annotations.coco.json").exists()


def _id(category: dict) -> int:
    return category["id"]
