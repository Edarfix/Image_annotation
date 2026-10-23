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
- It is run with an open-set protocol following Miller et al. (ECCV 2024, arXiv 2403.16528),
  so that "none" is a valid, evaluated answer. A seeded subset of classes is withheld per
  dataset, and images of withheld classes must be labelled "none".
- Closed-set results (all classes given) are reported as well, to compare with published
  ELEVATER numbers.

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
