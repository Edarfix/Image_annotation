# Decision log

One entry per decision that changes the brief ([`BRIEF.md`](BRIEF.md)) or fixes something it
left open. Entries are numbered `D<n>`, never renumbered; a reversed decision gets a new entry
that points to the old one.

## D1 (2026-10-23): repository root and package name

**Decision.** The repository root is `Image_annotation/` (the existing git repository) and the
Python package is `ovannot` ("open-vocabulary annotation") instead of `ovdannot`.

**Why.** The project also covers classification (D2); "ovd" (open-vocabulary detection) would
be misleading.

## D2 (2026-10-23): task 1, classification with a "none" answer

**Decision.** Besides detection (task 2), the testbed supports single-label image
classification where "none of the classes" is a valid answer. A dataset declares
`task: classification | detection` and whether "none" is allowed; a pipeline declares the task
it solves; outputs are `ClassPrediction` (task 1) or `Detection` (task 2). Classification
pipelines use the prefix `c` (`c00_…`); detection keeps `p00`–`p04`.

**Why.** Requested by the owner.

## D3 (2026-10-23): RF100-VL dev / test protocol

**Decision.** Only global settings (shared by all datasets) are tuned, on the `valid` split of
the 5 dev datasets. Final numbers are reported on the `test` split of all 100 datasets and,
separately, on the 95 datasets never used for tuning.

**Why.** The RF100-VL baselines "do not tune any parameters or modify zero-shot prompts
per-dataset" (paper §3.2). Reporting the 95 held-out datasets makes any leakage from tuning on
the dev datasets visible.

## D4 (2026-10-23): what each setting may use

**Decision.** `zero_shot` gets the class names only, as in the RF100-VL zero-shot baselines.
`few_shot` gets class names, annotator instructions and the official 10-shot examples (train
split of RF100-VL-FSOD). Any other input (dataset name, domain, descriptions generated from
extra context) defines a separate setting that must be declared explicitly. The runner strips
from the `DatasetSpec` everything the declared setting does not allow, before the pipeline
sees it.

**Why.** Keeps results comparable with the paper and makes accidental leakage impossible. This
was proposed as a default; the owner did not object.

## D5 (2026-10-23): model size cap

**Decision.** Kimi K3 is dropped; models above about 1T parameters are out of scope.

**Why.** `moonshotai/Kimi-K3` has about 2.78T parameters (Hugging Face safetensors metadata,
checked 2026-10-23). It would need several nodes just to act as an optional arbiter.

## D6 (2026-10-23): diagram rendering

**Decision.** mermaid-cli is installed locally under `.tools/` (git-ignored) by
`scripts/setup_mermaid.sh`. The rendered `diagram.svg` files are committed.

**Why.** Requested by the owner. Because the SVGs are committed, the cluster does not need
Node.js or a headless browser.

## D7 (2026-10-23): git workflow

**Decision.** Small atomic commits directly on `main`; there is no remote yet.

## D8 (2026-10-23): Pipeline base class and plugin discovery

**Decision.** The `Pipeline` abstract class and `register_pipeline` live in
`src/ovannot/core/pipeline.py`, not in `pipelines/base.py`. Each folder under `pipelines/` and
`datasets/` is a self-contained plugin (code, README, config, prompts). The registry imports
plugins by file path; folders whose name starts with `_` are templates and are skipped. A
plugin folder must register exactly the name of the folder.

**Why.** A top-level `datasets` package would clash with the Hugging Face `datasets` library.
Importing plugins by path means `pipelines/` and `datasets/` need no packaging, while the base
class stays importable from the installed package by pipelines, tests and scripts alike.

## D9 (2026-10-23): task 1 benchmark (proposal, to be confirmed at M1)

**Decision (proposed).**
- The public task 1 benchmark is ELEVATER image classification: 20 datasets (Li et al.,
  NeurIPS 2022 D&B, arXiv 2204.08790).
- It is run with an open-set protocol so that "none" is a valid, evaluated answer. A seeded
  subset of classes is withheld per dataset; pipelines only ever see the known classes; images
  of withheld classes must be labelled "none".
- The "none" decision is scored with the metrics of Miller et al. (ECCV 2024,
  arXiv 2403.16528): precision–recall of correct predictions against open-set errors (AuPR,
  P@95R, R@95P), alongside accuracy.
- Closed-set results (all classes given) are reported as well, to compare with published
  ELEVATER numbers.

**Correction (2026-10-23, after reading the PDF).** Miller et al. do not withhold classes per
dataset. They test each image twice: once with all classes, and once with the image's own
classes removed from the query set. With our pipelines, which prepare per dataset on the full
class list (e.g. concept cards), a per-image query set would reveal the answer (the missing
class). So we keep a fixed class split and reuse only their metrics.

**Why.**
- No public multi-domain classification benchmark with a "none" answer turned up in the
  2026-10-23 searches.
- ELEVATER is the standard "classification in the wild" suite. It has official splits, a
  few-shot protocol (5, 20 or 50 shots, three seeds) and class knowledge (WordNet, Wiktionary,
  GPT-3). That mirrors RF100-VL's structure of names, instructions and exemplars.

**Caveats.**
- The official download host (`cvinthewildeus.blob.core.windows.net`) no longer resolves
  (checked 2026-10-23). A third-party mirror (`ytaek-oh/elevater` on Hugging Face, 46 GB, same
  layout as the official registry) must be checked against the official per-split image counts
  before use.
- Several ELEVATER datasets are low-resolution or in-distribution for modern VLMs (e.g. CIFAR,
  MNIST). The exact open-set split and the OOD subset are fixed at M1, before any run.

## D10 (2026-10-23): RF100-VL variant to download

**Decision.** Download only RF100-VL-FSOD (Roboflow workspace `rf100-vl-fsod`). Its train
split holds the official 10-shot examples. Per the paper (§4.1), its test set is the full
benchmark's: "all methods across data regimes are evaluated on the same fully annotated test
set". This is to be checked at M1 by comparing image lists.

**Why.** We never fine-tune, so the full training sets are not needed. FSOD is about 6.9 GB on
the Hugging Face mirror `probicheaux/rf100-vl-fsod` (14,446 test images).

## D11 (2026-10-23): heavy dependencies

**Decision.** GPU and serving dependencies (torch, vLLM, SAM 3, transformers) will be optional
extras, installed on the cluster only. The base install is CPU-only and light.

**Why.** The laptop has no GPU, and the cluster compute nodes are aarch64 while the login node
is x86_64.

## D12 (2026-10-28): RF100-VL from the Hugging Face mirror; dev subset by rule

**Decision.**
- The data come from `probicheaux/rf100-vl-fsod` (revision `4bcf86c4`), converted to one COCO
  folder per dataset and split by `datasets/rf100vl/download.py`. Category names are in its
  `dataset_metadata.json`, already cleaned like the official package (placeholder removed, ids
  from 0). Domains are copied from the official package (`domains.json`, commit `451c6dd`).
- Dev subset: in each of Lab Imaging, Aerial, Industrial, Flora/Fauna and Document, the
  dataset with the largest `valid` split (472 images in total).

**Why.** No Roboflow key is needed, and the counts match the paper (14,446 test images). The
mechanical rule picks the dev datasets before any run, so they cannot be cherry-picked; the
larger `valid` splits give less noisy dev metrics. Confirmed by the owner (question.txt, Q1,
Q3). Not yet cross-checked against an official Roboflow export.

## D13 (2026-10-28): models served over HTTP from the shared vLLM container

**Decision.** MLLMs run in `vllm serve` and SAM 3 in a small FastAPI server
(`serving/sam3_server.py`), both inside the cluster's vLLM Apptainer image
(`shared/vllm-sifs/vllm_cuda13.sif`: vLLM 0.23.1, transformers 5.13, torch 2.11). Wrappers
reach them over HTTP without the cluster proxy. A SLURM job starts the servers it needs on
its own GPUs, then runs the experiment (`scripts/slurm/run_experiment.sbatch`).

**Why.** The ovannot environment stays CPU-only (D11) and identical on the laptop and the
cluster; torch and CUDA come from an image that already works on aarch64 B200 nodes.

## D14 (2026-10-28): p01 uses Qwen3-VL-30B-A3B for now

**Decision.** `p01_mllm_grounding_raw` uses Qwen3-VL-30B-A3B-Instruct instead of the brief's
235B-A22B, until the latter is downloaded. MLLM boxes get the score 1.0.

**Why.** 62 GB instead of about 470 GB, one GPU; same family and box format. Agreed by the
owner (question.txt, Q2). The constant score is the protocol of the RF100-VL and Qwen3-VL
evaluations.

## D15 (2026-10-28): COCO mAP read at maxDets = 500

**Decision.** `src/ovannot/eval/detection.py` reads every AP from `COCOeval.eval["precision"]`
at `maxDets = 500` instead of calling `COCOeval.summarize`.

**Why.** `summarize` computes its headline mAP with `maxDets=100` whatever `params.maxDets`
holds, which returns -1 when the list is `(1, 10, 500)`. detectron2, used by the RF100-VL
reference code, patches it the same way (`COCOevalMaxDets`, `coco_evaluation.py:684`).

## D16 (2026-10-28): task 1 benchmark, first version (completes D9)

**Decision.**
- Five ELEVATER datasets, chosen to be far from everyday photos and to span domains:
  - EuroSAT (satellite land use, 10 classes);
  - DTD (textures, 47);
  - GTSRB (traffic signs, 43);
  - FER-2013 (48×48 grayscale faces, 7);
  - FGVC-Aircraft (aircraft variants, 100).
- Images come from the mirror `ytaek-oh/elevater` (revision `48bf249c`). Class names and
  descriptions come from the official toolkit (`knowledge.json`, commit `00d0af7`, MIT):
  WordNet, Wiktionary and GPT-3 lines.
- Open set: 20% of each dataset's classes (at least one) are withheld with a fixed seed.
  Pipelines see only the known classes; images of withheld classes must be labelled "none".
- Samples: dev = 200 seeded images of `val`, test = 500 seeded images of `test`, per dataset.
  The samples are prefixes of a seeded permutation, so a smaller sample sits inside a larger
  one.
- Metrics:
  - accuracy (with "none"), closed-set accuracy, macro-F1, "none" P/R/F1;
  - Miller et al.'s AuPR, P@95R and R@95P, with true positives against open-set errors,
    ranked by the top known class's confidence (definitions checked in the paper, §4.1).

**Why.**
- The owner asked for a working task 1 and delegated the choice ("do the best").
- These are the ELEVATER datasets least like web photos, small enough to run tonight.
- PatchCamelyon is left out: with 2 classes, an open-set split leaves a single known class.
- The samples keep MLLM runs cheap; their size can grow without changing the earlier images.

**Caveats.**
- The mirror's per-split counts match the toolkit's registry, but the images were not checked
  against the original host, which is dead.
- Some GPT-3 descriptions are poor (e.g. GTSRB).
- None of these datasets is truly out of distribution for modern VLMs.

## D17 (2026-10-28): `described` setting

**Decision.** A third setting, `described`, passes the class names and their descriptions, and
nothing else: no exemplars, no instructions, no domain. Task 1 runs use it.

**Why.** Task 1 is "classes with their descriptions → one class or none". D4 requires any input
beyond names to be a declared setting; `few_shot` would also pass exemplars.
