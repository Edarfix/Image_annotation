"""Convert the RF100-VL-FSOD Hugging Face mirror into one COCO folder per dataset and split.

Usage (download on the login node, which has internet; convert anywhere)::

    hf download probicheaux/rf100-vl-fsod --repo-type dataset \\
        --revision 4bcf86c4cc58f3afecfb6fe7b3a541b33bfbb301
    uv run python datasets/rf100vl/download.py --snapshot <path printed by hf download>

Output, under ``$OVANNOT_DATA/rf100vl/`` (default ``data/rf100vl/``)::

    <dataset>/<split>/_annotations.coco.json   # COCO, integer ids, cleaned categories
    <dataset>/<split>/images/<file_name>
    manifest.json                               # source revision, counts, checks

The mirror (decision D10, Q3 in question.txt) stores one row per image with its boxes in COCO
``[x, y, w, h]`` pixels, and the category lists per dataset and split in
``dataset_metadata.json``. Its category ids are already cleaned like the official ``rf100vl``
package (Roboflow's placeholder category 0 removed, ids shifted to start at 0). Idempotent: a
split whose annotation file exists is skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from ovannot.paths import data_root
from ovannot.utils.io import read_json, write_json

SPLITS = ("train", "valid", "test")
HF_REPO = "probicheaux/rf100-vl-fsod"


def convert(snapshot: Path, out: Path, only: set[str] | None, force: bool) -> dict:
    meta = read_json(snapshot / "dataset_metadata.json")
    names = {ds_id: d["name"] for ds_id, d in meta["datasets"].items()}
    manifest: dict = {"source": HF_REPO, "snapshot": snapshot.name, "datasets": {}}
    for split in SPLITS:
        rows_by_ds: dict[str, list[dict]] = defaultdict(list)
        for shard in sorted((snapshot / "data").glob(f"{split}-*.parquet")):
            table = pq.read_table(shard)
            for row in table.to_pylist():
                if only is None or names[row["dataset_id"]] in only:
                    rows_by_ds[row["dataset_id"]].append(row)
        for ds_id, rows in sorted(rows_by_ds.items()):
            name = names[ds_id]
            split_dir = out / name / split
            ann_path = split_dir / "_annotations.coco.json"
            if ann_path.exists() and not force:
                manifest["datasets"].setdefault(name, {})[split] = read_json(ann_path)["info"]
                continue
            coco, info = to_coco(rows, meta["categories"][ds_id][split]["categories"], split_dir)
            coco["info"] = info
            write_json(ann_path, coco, indent=None)
            manifest["datasets"].setdefault(name, {})[split] = info
            print(f"{name}/{split}: {info}", flush=True)
    return manifest


def to_coco(rows: list[dict], categories: list[dict], split_dir: Path) -> tuple[dict, dict]:
    """COCO dict of one dataset split; writes its images under ``split_dir/images``."""
    rows = sorted(rows, key=lambda r: int(r["image_id"].rsplit("_", 1)[1]))
    image_dir = split_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    images, annotations, size_mismatch = [], [], 0
    file_names = [r["file_name"] for r in rows]
    if len(set(file_names)) != len(file_names):
        raise ValueError(f"{split_dir}: duplicate file names")
    for coco_id, row in enumerate(rows):
        data = row["image"]["bytes"]
        (image_dir / row["file_name"]).write_bytes(data)
        with Image.open(io.BytesIO(data)) as im:
            if im.size != (row["width"], row["height"]):
                size_mismatch += 1  # e.g. EXIF rotation; boxes follow width/height
        images.append(
            {
                "id": coco_id,
                "file_name": row["file_name"],
                "width": row["width"],
                "height": row["height"],
                "hf_image_id": row["image_id"],
            }
        )
        a = row["annotations"]
        for cat, bbox, area, crowd in zip(
            a["category_id"], a["bbox"], a["area"], a["iscrowd"], strict=True
        ):
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": coco_id,
                    "category_id": int(cat),
                    "bbox": [float(v) for v in bbox],
                    "area": float(area),
                    "iscrowd": int(crowd),
                }
            )
    cats = [{"id": c["id"], "name": c["name"]} for c in categories]
    known = {c["id"] for c in cats}
    unknown = sum(a["category_id"] not in known for a in annotations)
    if unknown:
        raise ValueError(f"{split_dir}: {unknown} annotations of unknown categories")
    info = {
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(cats),
        "image_size_mismatch": size_mismatch,
    }
    return {"images": images, "annotations": annotations, "categories": cats}, info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", type=Path, required=True, help="HF snapshot folder")
    parser.add_argument("--out", type=Path, default=None, help="default: $OVANNOT_DATA/rf100vl")
    parser.add_argument("--only", nargs="+", metavar="DATASET", help="convert these only")
    parser.add_argument("--force", action="store_true", help="rewrite existing splits")
    args = parser.parse_args(argv)
    out = args.out or data_root() / "rf100vl"
    manifest = convert(args.snapshot, out, set(args.only) if args.only else None, args.force)
    if not args.only:
        write_json(out / "manifest.json", manifest)
    mismatched = {
        f"{n}/{s}": i["image_size_mismatch"]
        for n, splits in manifest["datasets"].items()
        for s, i in splits.items()
        if i["image_size_mismatch"]
    }
    print(f"{len(manifest['datasets'])} datasets; image size mismatches: {mismatched or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
