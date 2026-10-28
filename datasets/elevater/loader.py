"""ELEVATER image classification with an open-set class split (task 1, D9, D16).

Config keys:

- ``name``: one of the datasets in ``knowledge.json`` (eurosat, dtd, gtsrb, fer2013,
  fgvc_aircraft);
- ``open_set`` (default true): withhold a seeded share of the classes. Pipelines only see the
  known classes; images of withheld classes must be labelled "none";
- ``withheld_fraction`` (default 0.2, at least one class, at least two known), ``seed``
  (default 0): the class split;
- ``max_images`` (default none): a seeded sample of at most this many images per split;
- ``root`` (default ``$OVANNOT_DATA/elevater``).

Image ids are ``<split>/<index in the split>``. Class descriptions come from ELEVATER's
knowledge files (WordNet, Wiktionary and GPT-3 descriptions), one sentence per line,
duplicates removed; the ``described`` and ``few_shot`` settings pass them to pipelines.
"""

from __future__ import annotations

import json
import random
from functools import cache
from pathlib import Path
from typing import Any

from PIL import Image

from ovannot.core.dataset import ClassificationGroundTruth, DatasetLoader, register_dataset
from ovannot.core.types import ClassSpec, DatasetSpec, ImageRecord, Task
from ovannot.paths import data_root
from ovannot.utils.io import read_json
from ovannot.utils.sampling import seeded_subset

SPLITS = ("val", "test")
DOMAINS = {  # for per-domain means in reports; never shown to pipelines (D4)
    "eurosat": "Remote sensing",
    "dtd": "Textures",
    "gtsrb": "Traffic signs",
    "fer2013": "Faces",
    "fgvc_aircraft": "Fine-grained objects",
}


@cache
def knowledge() -> dict[str, Any]:
    return read_json(Path(__file__).with_name("knowledge.json"))["datasets"]


def description(c: dict[str, Any]) -> str:
    lines = [c["def_wn"], c["def_wiki"], *c["gpt3"]]
    seen, out = set(), []
    for line in (x.strip() for x in lines):
        key = line.lower().rstrip(".")
        if line and key not in seen:
            seen.add(key)
            out.append(line)
    return "\n".join(out)


def sample_key(name: str, split: str) -> str:
    """Seed of the image sample of a split (shared with download.py, which extracts it)."""
    return f"elevater-{name}-{split}"


def class_split(n: int, fraction: float, seed: int) -> list[int]:
    """Indices of the withheld classes (sorted): seeded, at least 1, leaving at least 2."""
    k = min(max(1, round(fraction * n)), n - 2)
    return sorted(random.Random(f"elevater-open-set-{seed}").sample(range(n), k))


@register_dataset("elevater")
class Elevater(DatasetLoader):
    task = Task.CLASSIFICATION

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        self.name: str = cfg["name"]
        if self.name not in knowledge():
            raise ValueError(f"unknown ELEVATER dataset {self.name!r} (see knowledge.json)")
        self.classes = knowledge()[self.name]["classes"]
        self.root = Path(cfg.get("root") or data_root() / "elevater") / self.name
        self.open_set = bool(cfg.get("open_set", True))
        withheld = class_split(
            len(self.classes), float(cfg.get("withheld_fraction", 0.2)), int(cfg.get("seed", 0))
        )
        self.withheld = set(withheld) if self.open_set else set()
        self.max_images = cfg.get("max_images")

    def _rows(self, split: str) -> list[tuple[str, dict]]:
        path = self.root / split / "index.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run datasets/elevater/download.py")
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        ids = seeded_subset(sample_key(self.name, split), len(rows), self.max_images)
        return [(f"{split}/{i}", rows[i]) for i in ids]

    def spec(self) -> DatasetSpec:
        known = [c for i, c in enumerate(self.classes) if i not in self.withheld]
        return DatasetSpec(
            name=self.name,
            task=Task.CLASSIFICATION,
            classes=[ClassSpec(name=c["name"], description=description(c)) for c in known],
            splits={s: [i for i, _ in self._rows(s)] for s in SPLITS if self._has(s)},
            domain=DOMAINS.get(self.name, ""),
            allows_none=self.open_set,
        )

    def images(self, split: str) -> list[ImageRecord]:
        records = []
        for image_id, row in self._rows(split):
            path = self.root / split / "images" / row["file"]
            w, h = _size(path)
            records.append(ImageRecord(image_id, path, w, h))
        return records

    def ground_truth(self, split: str) -> ClassificationGroundTruth:
        return ClassificationGroundTruth(
            labels={
                image_id: None
                if row["label"] in self.withheld
                else self.classes[row["label"]]["name"]
                for image_id, row in self._rows(split)
            }
        )

    def _has(self, split: str) -> bool:
        return (self.root / split / "index.jsonl").exists()


def _size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size
