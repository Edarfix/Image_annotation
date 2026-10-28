# ELEVATER (task 1 benchmark)

ELEVATER image classification (Li et al., NeurIPS 2022 D&B, arXiv 2204.08790; reading note in
[papers/README.md](../../papers/README.md#elevater)), used for task 1 with an open-set class
split (decisions D9, D16).

## Source and licence

- **Images and labels.** The Hugging Face mirror `ytaek-oh/elevater`, revision `48bf249c`
  (46 GB, same layout as the official registry; the official host no longer resolves). Each
  split has an index file `<split>.txt` with lines `<zip>@<path in zip> <label id>`.
- **Class names and descriptions.** The official toolkit `Elevater_Toolkit_IC` (MIT), commit
  `00d0af7`. Its files `resources/knowledge/external/*_knowledge.tsv` (WordNet and Wiktionary
  definitions) and `resources/knowledge/gpt3/GPT3_*.tsv` (5 GPT-3 descriptions per class)
  are copied into `knowledge.json` in label-id order, which is how the toolkit pairs them.
- **Licences.** Each dataset keeps its own licence (EuroSAT, DTD, GTSRB, FER-2013,
  FGVC-Aircraft); research use only. TODO(verify): list them per dataset before any release.

## Datasets used (D16)

| Name | Content | Classes | Known / withheld (seed 0) |
|---|---|---:|---|
| `eurosat` | Sentinel-2 satellite land use | 10 | 8 / 2 |
| `dtd` | describable textures | 47 | 38 / 9 |
| `gtsrb` | German traffic signs | 43 | 34 / 9 |
| `fer2013` | 48×48 grayscale facial expressions | 7 | 6 / 1 |
| `fgvc_aircraft` | aircraft variants (fine-grained) | 100 | 80 / 20 |

They were chosen as the ELEVATER datasets least like web photos. PatchCamelyon is left out: with
2 classes, an open-set split leaves one known class.

## Protocol

- **Open set.** `round(0.2 × classes)` classes are withheld per dataset (at least one, at
  least two known), drawn with a fixed seed (`loader.class_split`). Pipelines see only the
  known classes (`DatasetSpec.allows_none = true`). Images of withheld classes have the
  ground truth `None` and must be labelled "none". `open_set: false` gives the closed-set
  variant, comparable with published ELEVATER numbers.
- **Splits.** dev = `val`, test = `test`. Each uses a seeded sample: 200 images per dataset for
  dev, 500 for test (`configs/datasets/elevater_dev.yaml`). A sample is the prefix of a seeded
  permutation, so it grows without changing earlier images.
- **Settings.** `described` (D17): class names with their descriptions, one line per
  description sentence. `zero_shot`: names only. Few-shot (the train split) is not wired yet.
- **Metrics** (`src/ovannot/eval/classification.py`):
  - accuracy (with "none"), closed-set accuracy, macro-F1, "none" P/R/F1;
  - the open-set metrics of Miller et al. (ECCV 2024): AuPR, P@95R and R@95P, with true
    positives against open-set errors.

## Use

```bash
# login node: images, labels and index files of the five datasets (about 2.6 GB)
hf download ytaek-oh/elevater --repo-type dataset \
    --revision 48bf249c204612310cb56f0a51ca7a414acf63ad \
    --include 'eurosat_clip_20210930/*' 'dtd_20211007/*' 'gtsrb_20210923/*' \
              'fer_2013_20211008/*' 'fgvc_aircraft_2013b_variants102_20211007/*'
# anywhere: extract the seeded samples (val 200, test 500) under opaque file names
uv run python datasets/elevater/download.py --snapshot <snapshot folder>
```

Image files get opaque names (`000042.png`): the mirror's paths contain the class, and pipelines
can read `ImageRecord.path`.
