# Datasets

One folder per dataset. `loader.py` registers a `DatasetLoader` under the folder name
(see [AGENTS.md](../AGENTS.md#add-a-dataset)), and `README.md` documents the source, licence,
splits, protocol and settings. Data files live under `$OVANNOT_DATA` (default `data/`, a
git-ignored symlink to cluster storage), never in git.

## Unified format

A loader exposes three things, which the runner keeps apart (`src/ovannot/core/dataset.py`):

| Method | Content | Who sees it |
|---|---|---|
| `spec()` | `DatasetSpec`: task, classes (name, description, exemplars), splits as lists of image ids, annotator instructions, domain, whether "none" is allowed | The pipeline, restricted to the run's setting (D4) |
| `images(split)` | `ImageRecord` list: id, path, width, height | The pipeline |
| `ground_truth(split)` | Detection: a COCO dict and the id mapping. Classification: one label or `None` per image | The evaluator only |

Image ids are unique across splits (e.g. `"test/42"`). Boxes are absolute-pixel xyxy.

## Datasets

| Folder | Task | Role | State |
|---|---|---|---|
| [`rf100vl/`](rf100vl/) | detection | Phase 1 benchmark (RF100-VL, FSOD variant, D10) | loader and dev subset done |
| `elevater/` | classification | Phase 1 benchmark for task 1, open-set split (D9, to confirm) | M1 |
| `_template_proprietary/` | either | Template for phase 2 datasets, with a confidentiality checklist | M1 |
