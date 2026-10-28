"""Extract the ELEVATER images used by task 1 from the Hugging Face mirror (D9, D16).

Usage (download on the login node, extract anywhere)::

    hf download ytaek-oh/elevater --repo-type dataset \\
        --revision 48bf249c204612310cb56f0a51ca7a414acf63ad --include '<folder>/*'
    uv run python datasets/elevater/download.py --snapshot <snapshot folder>

For each dataset of ``knowledge.json`` and each split in ``SPLITS``, the mirror's index file
(``<split>.txt``, lines ``<zip>@<path in zip> <label id>``) gives the images and labels. Images
are extracted to ``$OVANNOT_DATA/elevater/<dataset>/<split>/images/<line number>.<ext>``, under
opaque names because the mirror's paths contain the class (PPM files, which model servers do
not read, become PNG). They are listed in ``<split>/index.jsonl`` as
``{"file", "label": <label id>, "source": <mirror path>}``; only the evaluator reads labels.
Idempotent unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

from ovannot.paths import data_root
from ovannot.utils.io import atomic_write_text, read_json
from ovannot.utils.sampling import seeded_subset

SPLITS = ("val", "test")
KNOWLEDGE = Path(__file__).with_name("knowledge.json")


def convert_split(folder: Path, name: str, split: str, out: Path, k: int | None) -> int:
    """Index every image of the split; extract only the seeded sample of ``k`` images the
    loader will use (``max_images``; all if None). Returns the number extracted."""
    lines = [
        ln.rsplit(" ", 1) for ln in (folder / f"{split}.txt").read_text().splitlines() if ln.strip()
    ]
    # same key as loader.py's sample_key (tests/test_classification.py checks they agree)
    rows, wanted = [], set(seeded_subset(f"elevater-{name}-{split}", len(lines), k))
    by_zip: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for i, (ref, label) in enumerate(lines):
        zip_name, member = ref.split("@", 1)
        # opaque file names: the mirror's paths contain the class (e.g. "banded/banded_0002.jpg")
        # and pipelines can read ImageRecord.path
        suffix = ".png" if member.lower().endswith(".ppm") else Path(member).suffix.lower()
        target = Path(f"{i:06d}{suffix}")
        rows.append({"file": str(target), "label": int(label), "source": ref})
        if i in wanted:
            by_zip[zip_name].append((member, out / "images" / target))
    for zip_name, members in sorted(by_zip.items()):
        with zipfile.ZipFile(folder / zip_name) as zf:
            for member, target in members:
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as f:
                    if member.lower().endswith(".ppm"):  # model servers do not read PPM
                        Image.open(f).save(target)
                    else:
                        target.write_bytes(f.read())
    atomic_write_text(out / "index.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    return sum(len(m) for m in by_zip.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="default: $OVANNOT_DATA/elevater")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--max-images",
        nargs="+",
        default=["val=200", "test=500"],
        metavar="SPLIT=N",
        help="extract only the loader's seeded sample (default: val=200 test=500)",
    )
    args = parser.parse_args(argv)
    out = args.out or data_root() / "elevater"
    limits = {s: int(n) for s, n in (x.split("=") for x in args.max_images)}
    for name, d in read_json(KNOWLEDGE)["datasets"].items():
        for split in SPLITS:
            split_out = out / name / split
            if (split_out / "index.jsonl").exists() and not args.force:
                continue
            n = convert_split(
                args.snapshot / d["mirror_folder"], name, split, split_out, limits.get(split)
            )
            print(f"{name}/{split}: {n} images extracted", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
