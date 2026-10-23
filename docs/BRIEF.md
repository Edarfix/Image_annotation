# Project brief

> English translation of the brief written by the project owner on 2026-10-23 (original in
> French). Additions agreed the same day are marked **[added]**. This file is the reference
> specification; every change to it is logged in [`decisions.md`](decisions.md) (entries `D<n>`).

## Mission

Build a research repository: a **testbed for automatic open-vocabulary image-annotation
pipelines, targeting out-of-distribution (OOD) datasets**.

**Target tasks.**
- **Task 1: image classification [added].** Input: a dataset of images and the classes of
  interest (names, optionally descriptions and a few labelled examples). Output: for each image,
  exactly one of the classes **or "none"**, with a score and traceability metadata.
- **Task 2: detection.** Input: a dataset of images and, for each class of interest, a textual
  description, plus optionally a few annotated examples. Output: for each image, every instance
  of the classes of interest, with a bounding box (mandatory), a mask (optional), a score and
  traceability metadata.

**Success criteria, in order:**
1. Phase 1: performance on public benchmarks, following their official protocols:
   Roboflow100-VL (RF100-VL, NeurIPS 2025 Datasets & Benchmarks) for task 2; for task 1, the
   open benchmark chosen in D9.
2. Phase 2, the real evaluation: performance on proprietary benchmarks, added later through a
   dataset template.

The repository must make it **efficient to compare several pipelines, to add one without
touching the rest, and to understand each one without reading its code**. Structure and
documentation matter as much as scores.

Code, docstrings and documentation are in English: the repository will accompany a paper.

## Non-negotiable rules

- **Nothing invented.** Never invent a result, a Hugging Face model id, an arXiv id, an API
  signature or a coordinate format. Check the source (model card, official repository, arXiv
  page). If that is impossible, write `TODO(verify)` with an explanation.
- **Proprietary data.** No call to any external API; only locally served models. Never commit
  proprietary data, or predictions on it, to git.
- **Strict dev / test separation.**
  - Prompts, thresholds and hyper-parameters are tuned on a dev split.
  - The test split is used only for final reporting.
  - The evaluation code is frozen after M1. Any later change requires a changelog entry and a
    re-run of the affected pipelines.
- **RF100-VL setting.** Each run declares its setting (`zero_shot`, or `few_shot` with
  annotator instructions and examples) and records it in its metadata. Settings are never mixed
  in one table without saying so.
- **Licences.** Check the licence of every checkpoint before integrating it and record it in
  `configs/models/*.yaml`.
- **No notebooks in production.** Every reported number is produced by a script in `scripts/`
  from a versioned run.
- **Commits and milestones.** Small, atomic commits. At each milestone (M0 to M5), stop and
  write a recap: what works, what remains TODO, numbers obtained, decisions taken and why.

## Repository layout

The root is `Image_annotation/` rather than `ovd-annot/` and the package is `ovannot` rather
than `ovdannot` (D1). The `Pipeline` base class lives in `src/ovannot/core/pipeline.py` rather
than `pipelines/base.py` (D8). `docs/` is an addition.

```
Image_annotation/
├── README.md                  # project goal, quickstart, how to read the leaderboard
├── AGENTS.md                  # conventions for agents/devs: add a pipeline, a dataset, a model; key commands
├── pyproject.toml             # uv, python >= 3.11, ruff, pytest, pydantic
├── .gitignore                 # data/, papers/pdf/, results/runs/, caches, *.ckpt
├── docs/                      # [added] this brief, decision log, milestone recaps
├── papers/
│   ├── README.md              # table of papers + one reading note per paper
│   ├── papers.yaml            # source of truth: arxiv_id, expected title, role in the project
│   ├── download_papers.py
│   └── pdf/                   # git-ignored
├── data/                      # git-ignored: raw/, processed/ (or symlinks to cluster storage)
├── datasets/
│   ├── README.md              # unified format, list of datasets, how to add one
│   ├── rf100vl/               # download.py, loader.py, README.md (protocol, splits, settings, dev subset)
│   └── _template_proprietary/ # loader.py, README.md, confidentiality checklist
├── src/ovannot/
│   ├── core/                  # types.py (dataclasses), registry.py, coco_io.py, config.py
│   ├── models/                # uniform wrappers: mllm_client.py (OpenAI-compatible), sam3.py, rex_omni.py, hub.py (lazy loading)
│   ├── stages/                # reusable steps: concept_card.py, propose_sam3.py, propose_mllm.py, fuse.py, verify_binary.py, verify_checklist.py
│   ├── eval/                  # rf100vl_official.py, coco_metrics.py, f1_at_iou.py, error_analysis.py
│   ├── runner/                # run.py (parallelism, resume, cache, timing, failure capture)
│   └── utils/                 # boxes.py, viz.py, tiling.py, prompts.py, hashing.py
├── pipelines/
│   ├── README.md              # index: one line per pipeline (status, parent, idea, best score), auto-generated table section
│   ├── _template/             # pipeline.py, config.yaml, README.md, diagram.mmd, diagram.svg, prompts/
│   ├── p00_sam3_raw/
│   ├── p01_mllm_grounding_raw/
│   ├── p02_sam3_concept_card/
│   ├── p03_concept_card_verifier_binary/
│   └── p04_concept_card_verifier_checklist/
├── configs/
│   ├── models/                # one yaml per model: checkpoint, licence, endpoint, precision, coordinate format
│   ├── datasets/              # one yaml per dataset / subset
│   └── experiments/           # one yaml per experiment = pipeline + dataset + setting + overrides
├── serving/
│   ├── README.md              # GPU needs per model, how to launch, health checks
│   └── launch_*.sh            # one script per model (vLLM OpenAI-compatible, cluster scheduler)
├── scripts/
│   ├── run.py                 # python scripts/run.py --exp configs/experiments/xxx.yaml
│   ├── compare_runs.py        # metric and error diff between two runs
│   ├── make_leaderboard.py    # → results/leaderboard.md + auto section of pipelines/README.md + tables of pipeline READMEs
│   └── render_diagrams.py     # diagram.mmd → diagram.svg (mermaid-cli)
├── results/
│   ├── runs/                  # git-ignored except metrics.json and the resolved config
│   ├── concept_cards/         # versioned concept cards, editable by hand
│   ├── leaderboard.md
│   └── reports/
└── tests/                     # unit tests (boxes, coco_io, fuse), evaluation sanity tests, smoke test per pipeline
```

## Code contract

Common types, in `src/ovannot/core/types.py`. This is the contract as specified; the implemented
version adds the classification task (`Task`, `ClassPrediction`, `DatasetSpec.task`,
`DatasetSpec.allows_none`, D2) and an image `path` on `Exemplar`, so that pipelines can read
exemplar images without knowing the dataset layout.

```python
@dataclass
class Exemplar:
    image_id: str
    bbox_xyxy: tuple[float, float, float, float]

@dataclass
class ClassSpec:
    name: str
    description: str
    exemplars: list[Exemplar]  # empty in zero-shot

@dataclass
class DatasetSpec:
    name: str
    classes: list[ClassSpec]
    splits: dict[str, list[str]]
    annotator_instructions: str | None  # RF100-VL instruction text, if any
    domain: str

@dataclass
class ImageRecord:
    image_id: str
    path: Path
    width: int
    height: int

@dataclass
class Detection:
    bbox_xyxy: tuple[float, float, float, float]  # absolute pixels, always
    class_name: str
    score: float  # mandatory, must allow ranking (mAP)
    mask_rle: dict | None
    provenance: list[str]  # models that produced / validated the detection
    extras: dict  # e.g. attribute scores, verifier rationale

@dataclass
class Failure:
    stage: str
    kind: str  # parse_error | timeout | oom | empty_output | model_error | invalid_box | other
    message: str

@dataclass
class ImageAnnotation:
    image_id: str
    detections: list[Detection]
    timings_s: dict[str, float]  # per stage
    model_calls: dict[str, int]
    tokens: dict[str, int]
    failures: list[Failure]
```

Pipeline interface:

```python
class Pipeline(ABC):
    id: str

    def __init__(self, cfg: dict, hub: ModelHub): ...

    def prepare(self, ds: DatasetSpec, workdir: Path) -> None:
        """Called once per dataset (e.g. concept cards). The result is cached."""

    def annotate(self, img: ImageRecord) -> ImageAnnotation:
        """Never raises: every error becomes a Failure."""
```

- **Registry.** `@register_pipeline("p02_sam3_concept_card")`. Adding a pipeline means copying
  `_template/`, implementing it and adding an experiment yaml, with no change anywhere else.
- **Model wrappers.** They expose uniform interfaces:
  - `Grounder.detect(image, phrases, exemplars=None) -> list[Detection]`
  - `Segmenter.masks(image, boxes)`
  - `MLLM.chat(messages, images, json_schema=None, logprobs=False)`

  Every coordinate conversion (relative 0–1000 format, quantised tokens, etc.) happens in the
  wrapper, following the format checked in the model card. Pipelines only ever see absolute
  pixel xyxy.
- **Disk cache** of every model call, keyed by the hash of (model, version, prompt, image hash,
  generation parameters). Re-running an experiment recomputes only what changed.
- **MLLMs served through OpenAI-compatible endpoints (vLLM).** The address comes from
  `configs/models/*.yaml`, so a pipeline does not know where the model runs.
- **Shared stages.** Pipelines compose functions from `src/ovannot/stages/`. No copy-paste
  between pipelines: p03 and p04 reuse p02's stages.

## Runner and outputs

`python scripts/run.py --exp <yaml> [--subset dev|test] [--limit N] [--workers K]`

Expected behaviour:
- resumable runs (images already done are skipped);
- asynchronous, parallel MLLM calls;
- structured logs.

Each run writes `results/runs/<date>_<pipeline>_<dataset>_<setting>_<hash>/` containing:
- `config.resolved.yaml`, with the git commit, the model versions and the setting;
- `predictions.json`, in COCO format, with the additional fields in `extras`;
- `metrics.json`:
  - mAP@[.5:.95] and AP50 following the official RF100-VL protocol;
  - precision, recall and F1 at IoU 0.5 at the operating threshold;
  - per-class metrics;
- `timings.json`: mean, p50, p95 and max per image and per stage, cost of `prepare`, calls and
  tokens per image;
- `failures.jsonl`, plus the failure rate per kind and per stage;
- `errors.json`: decomposition of detection errors (classification, localisation, duplicate,
  background, missed), TIDE-style;
- `viz/`: 20 random images and the 20 worst images, with ground truth and predictions overlaid;
- `run.log`.

For task 1 **[added]**, `predictions.json` holds one label per image and `metrics.json` holds
classification metrics (defined at M1).

## Mandatory documentation of a pipeline (`pipelines/<id>/README.md`)

1. **Summary.** Three lines: idea, hypothesis tested, parent pipeline and difference with it.
2. **Diagram.** `diagram.mmd` (Mermaid), rendered to `diagram.svg` by
   `scripts/render_diagrams.py` and included as an image. It shows the stages, the models and
   the data flows.
3. **Models.** A table with: role, model, exact checkpoint, licence, precision / quantisation,
   GPU memory, serving mode.
4. **Prompts.** They live in `prompts/*.txt` or `*.jinja`, never hard-coded in the `.py`. The
   section lists each prompt with its role, its variables and the expected JSON output schema.
5. **Step-by-step walkthrough.** Inputs and outputs of each stage, with their formats.
6. **Rationale.** Why this pipeline should work, which results from the literature motivate it
   (link to `papers/README.md`).
7. **Failure structure.** For each stage:
   - the possible failure modes (parse error, empty output, hallucinated box, confusion with a
     confuser, missed small object, duplicate, timeout, OOM...);
   - how each mode is detected;
   - what the pipeline does in that case (retry, fallback, give up);
   - the rates observed in the last run.
8. **Cost.** Mean, p95 and max time per image; calls and tokens per image; cost of `prepare`;
   GPU resources.
9. **Results.** Table generated automatically by `make_leaderboard.py`, never edited by hand.
10. **Known limitations and next steps.**
11. **Changelog.**

A pipeline is **done** when:
- it is registered and has its config;
- its README is complete, with the rendered diagram;
- its prompts are in files;
- its smoke test passes (3 images);
- it has been run on the dev subset;
- its results appear in the leaderboard.

## `papers/` folder

1. Create `papers/papers.yaml` with the list below. For each entry, check the exact title on the
   arXiv page `https://arxiv.org/abs/<id>` and fix the yaml if needed. If an id does not match
   the expected title, report it; do not guess.
2. `download_papers.py`:
   - downloads `https://arxiv.org/pdf/<id>` to `papers/pdf/<year>_<acronym>_<short-title>.pdf`;
   - is idempotent;
   - checks that the title found on the first page of the PDF matches the expected title;
   - logs failures.
3. Fill `papers/README.md`:
   - a table: title, year, venue, link, role in the project, download status;
   - then a 5 to 10 line note per paper: core idea, what we reuse (model, prompt, protocol,
     trick), relevant numbers with table or page number, pipelines of the repo that draw on it.

   The notes must be written **from the downloaded PDFs**, not from memory.

Main papers:

| arXiv | Title | Role |
|---|---|---|
| 2511.16719 | SAM 3: Segment Anything with Concepts | Main grounder, data engine with AI verifiers, SAM 3 Agent |
| 2505.20612 | Roboflow100-VL: A Multi-Domain Object Detection Benchmark for Vision-Language Models | Target benchmark, protocol, annotator instructions |
| 2510.12798 | Detect Anything via Next Point Prediction (Rex-Omni) | Second proposer, detection-specialised MLLM |
| 2511.21631 | Qwen3-VL Technical Report | Grounding of the main MLLM, coordinate format |
| 2507.10844 | LAOD (MLLM-guided autonomous object detection) | The MLLM decides what, the detector decides where |
| 2603.25645 | Colon-Bench: An Agentic Workflow for Scalable Dense Lesion Annotation | Detection and verification agents in an OOD domain |
| 2512.04585 | SAM3-I: Segment Anything with Instructions | Limits of the agent loop on long instructions |
| 2502.16486 | MQADet | Multimodal QA to strengthen an open-vocabulary detector |
| 2505.14362 | DeepEyes: Incentivizing "Thinking with Images" via RL | Tool-based zoom, small objects |
| 2602.02564 | Label Curation Using Agentic AI (AURA) | Probabilistic aggregation of several AI annotators |

Complementary papers:

| arXiv | Title | Role |
|---|---|---|
| 2603.27494 | Learning to Focus and Precise Cropping | Grounding rewards for cropping |
| 2511.19046 | MedSAM3 | SAM 3 agent in a specialised domain |
| 2604.02032 | IndoorCrowd | Auto-labelling comparison, SAM3 vs GroundingSAM |
| 2303.05499 | Grounding DINO | Reference open-vocabulary baseline |

Task 1 papers **[added]** (D9):

| arXiv | Title | Role |
|---|---|---|
| 2204.08790 | ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models | Task 1 benchmark: 20 classification datasets, zero/few-shot protocol, class knowledge |
| 2403.16528 | Open-Set Recognition in the Age of Vision-Language Models | Protocol for the "none" answer: open-set evaluation of VLM classifiers |

## Initial models

Check the exact ids, licences, output formats and memory needs in the model cards before
writing the configs.

| Model | Role | Notes |
|---|---|---|
| SAM 3 (Meta, gated checkpoint, SAM License) | Text and exemplar grounder, masks | SAM 3.1 variant optional, gains mostly on video |
| Rex-Omni (IDEA-Research, 3B) | Second proposer | Not before M4, and only if recall plateaus |
| Qwen3.5-397B-A17B | Concept-card compiler | Few calls |
| Qwen3.5-122B-A10B or Qwen3.8-27B | Verifier | High throughput |
| Qwen3-VL-235B-A22B-Instruct | Direct MLLM grounding baseline | |
| ~~Kimi K3~~ | ~~Arbiter (optional)~~ | **Dropped** (D5): about 2.8T parameters; models stay under 1T parameters |

## Initial pipelines

Start simple. Each pipeline adds a single building block to its parent.

**`p00_sam3_raw`**: SAM 3 prompted with the raw name of each class. This is the floor.

**`p01_mllm_grounding_raw`**: direct grounding by the MLLM, JSON output, prompt with name and
description. The score comes from logprobs if possible, otherwise from a documented method.

**`p02_sam3_concept_card`**:
- `prepare()`: the compiler MLLM produces one concept card per class, as JSON validated by
  pydantic, with:
  - short visual phrases for SAM 3 (no jargon);
  - the discriminative attributes;
  - the confusers;
  - the known pitfalls.
- Cards are saved in `results/concept_cards/<dataset>/`, editable by hand and versioned.
- Annotation: SAM 3 is prompted with all the phrases of a class, then the detections are fused
  per class (NMS or WBF, configurable).

**`p03_concept_card_verifier_binary`**:
- Builds on p02.
- Each candidate is shown to the verifier in two views: a crop with a context margin, and the
  full image with the box drawn.
- The verifier answers yes or no. The score combines the SAM 3 score and the probability of
  "yes".

**`p04_concept_card_verifier_checklist`**:
- Builds on p02.
- The verifier rates each attribute of the card and each confuser.
- The decision is a deterministic rule whose weights and thresholds are calibrated on dev.
- The attribute scores are stored in `extras` for explainability.

Task 1 pipelines **[added]** use the prefix `c` (`c00_…`) and are defined at M3; detection
keeps `p00`–`p04`.

## Evaluation

- Reproduce the official RF100-VL evaluation protocol and code (find the official repository and
  its download tooling; a Roboflow API key may be needed, read from an environment variable).
- If the paper or the repository provides reproducible baseline numbers, reproduce at least one.
- Mandatory sanity tests:
  - the ground truth submitted as predictions gives an mAP of 1;
  - an empty prediction gives 0;
  - boxes shifted by a few pixels lower the mAP consistently.
- Dev subset to iterate on: 5 RF100-VL datasets from varied domains (e.g. medical, aerial,
  industrial / defects, biology, document or thermal). Justify the choice in
  `datasets/rf100vl/README.md`.
- `make_leaderboard.py` produces a pipeline × dataset table with:
  - mAP and AP50;
  - precision, recall and F1 at IoU 0.5;
  - mean and p95 time per image;
  - MLLM calls per image;
  - failure rate;
  - setting.

## Milestones

- **M0**: repository skeleton, pyproject, ruff, pytest, README and AGENTS.md; papers downloaded
  and `papers/README.md` filled.
- **M1**: RF100-VL downloaded and loaded in the unified format; annotator instructions and
  few-shot examples retrieved; dev subset defined; official evaluation wired and sanity tests
  green; proprietary dataset template ready. **The evaluation is frozen from here on.**
  [added] Task 1 benchmark loader and classification metrics, frozen at the same time.
- **M2**: serving scripts, model wrappers, cache; smoke test on one image per model, with
  visualisation.
- **M3**: full runner; p00 and p01 run on the dev subset; first leaderboard.
- **M4**: p02, p03 and p04; comparative error analysis; leaderboard updated.
- **M5**: full run on the RF100-VL test set, in the declared setting, for the best pipeline and
  two baselines; report in `results/reports/rf100vl_<date>.md`.
- **Phase 2**, later: plug in the proprietary datasets through the template, without any
  external call.

## Hardware context

- Cluster with B200 GPUs. Compute nodes are aarch64, the login ("frontal") node is x86_64.
  Scheduler, storage paths, internet access from compute nodes and quotas are to be filled in
  on the cluster (the owner will set this up there once the main code exists).
- Local development machine: x86_64, no GPU, limited storage. Only code, tests on synthetic
  fixtures and the papers live there.
- The Hugging Face token (gated SAM 3 checkpoints) and the Roboflow API key are provided by the
  owner through environment variables, never committed.
