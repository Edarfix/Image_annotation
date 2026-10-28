# RF100-VL

Roboflow100-VL (Robicheaux et al., NeurIPS 2025 D&B, arXiv 2505.20612; reading note in
[papers/README.md](../../papers/README.md#rf100vl)). 100 detection datasets from Roboflow
Universe in 7 domains, licensed Apache 2.0.

**State: loader, conversion and dev subset done (2026-10-28, D12).** Annotator instructions are
still missing (see "Open points").

## Use

```bash
# login node (internet): fetch the mirror at a pinned revision
hf download probicheaux/rf100-vl-fsod --repo-type dataset \
    --revision 4bcf86c4cc58f3afecfb6fe7b3a541b33bfbb301
# anywhere: one COCO folder per dataset and split under $OVANNOT_DATA/rf100vl/
uv run python datasets/rf100vl/download.py --snapshot <snapshot folder>
```

`loader.py` registers `rf100vl`; its config is `{name: <dataset>}`. Image ids are
`<split>/<coco id>`. Few-shot exemplars are the boxes of the `train` split (the official
10-shot examples). The domains in `domains.json` are copied from the official package.

## Dev subset (D3, D12)

In each of five domains, the dataset with the largest `valid` split (a rule fixed before any
run, so that the choice cannot follow results): `configs/datasets/rf100vl_dev.yaml`.

| Domain | Dataset | Valid images | Classes |
|---|---|---:|---:|
| Lab Imaging | `ufba-425` | 105 | 32 |
| Aerial | `apoce-aerial-photographs-for-object-detection-of-construction-equipment` | 52 | 7 |
| Industrial | `2024-frc` | 89 | 11 |
| Flora/Fauna | `into-the-vale` | 51 | 6 |
| Document | `paper-parts` | 175 | 19 |

Global settings are tuned on these `valid` splits only. Test numbers are reported over all
100 datasets and over the 95 others (`configs/datasets/rf100vl_all.yaml`).

## Facts checked on 2026-10-23

## Source and download

- Official tooling: the [`rf100vl`](https://github.com/roboflow/rf100-vl) package (Apache 2.0).
  It downloads COCO JSON exports from Roboflow Universe and needs a free `ROBOFLOW_API_KEY`
  (read from the environment).
  - Its workspaces: `rf100-vl` (full), `rf100-vl-fsod` (few-shot variant), and the 20-dataset
    subsets `rf20-vl` and `rf20-vl-fsod`.
  - It downloads the *latest* version of each Roboflow project, so the download script must
    record the version ids it got.
  - Each dataset has `train/`, `valid/` and `test/` folders, each with
    `_annotations.coco.json`. The package's cleaning step drops Roboflow's placeholder
    category 0 and shifts the category ids.
- Mirror by the first author on Hugging Face, no key needed: `probicheaux/rf100-vl-fsod`
  (parquet, Apache 2.0, about 6.9 GB). It has 4,437 train, 3,652 valid and 14,446 test
  images. Use it as a fallback and cross-check, after verifying it matches the official export.
- We download the FSOD variant only (D10). Its train split holds the official 10-shot
  examples, and its test set is the benchmark's test set (paper §4.1). Check the latter at M1
  by comparing image lists.
- Domains: the package maps each dataset to one of 7 categories (Flora/Fauna 23, Industrial 22,
  Misc 15, Lab Imaging 13, Aerial 11, Document 10, Sport 6).

## Evaluation protocol (paper §3.2, §4.1, App. A–B)

- pycocotools `COCOeval` on boxes, with **maxDets = 500** instead of 100.
- Each dataset is evaluated on its own; results are averaged per domain and over all
  100 datasets.
- No per-dataset tuning of zero-shot prompts or parameters. Settings: zero-shot gets class
  names; few-shot gets 10 shots, annotator instructions and images.
- MLLM boxes without a score get 1.0. A failed request yields empty predictions for that
  image.
- The repository has no evaluation script, only a test that ground truth scored as
  predictions gives mAP 1 (`test/test_accuracy.py`). The CVPR'25 challenge baseline repository
  (`anishmadan23/foundational_fsod`, branch `fsod_rf20vl`) uses detectron2's `COCOEvaluator`.
- Baseline predictions for RF20-VL are linked from rf100-vl.org (`predictions_rf20vl.zip`).
  Reproducing the paper's numbers from them is the planned check of our evaluation (M1, on
  the cluster).

## Open points

- TODO(verify): where the annotator instructions are published. They are not in the pip
  package, the HF mirrors, or the GitHub repositories; probably on the Roboflow Universe FSOD
  projects (needs the API key).
- Cross-check the mirror against an official Roboflow export (needs `ROBOFLOW_API_KEY`).
