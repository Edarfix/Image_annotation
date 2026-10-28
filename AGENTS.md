# Working in this repository (agents and developers)

Read first: [docs/BRIEF.md](docs/BRIEF.md) (the specification),
[docs/decisions.md](docs/decisions.md) (what changed since, and why), then this file. The
latest milestone recap in [docs/milestones/](docs/milestones/) says where the work stands.

## Rules that are never bent

1. **Nothing invented.** Never write a result, a Hugging Face id, an arXiv id, an API
   signature, a licence or a coordinate format without checking the source (model card,
   official repository, paper PDF). If it cannot be checked, write `TODO(verify)` with what
   is missing.
2. **Proprietary data stays local.** No external API calls on it. No data, predictions,
   class names or metrics about it in git: they live in `*_private/` folders, which git
   ignores.
3. **Dev / test separation.**
   - Prompts, thresholds and hyper-parameters are tuned on dev splits only.
   - Test splits are for final reporting.
   - On RF100-VL, only settings shared by all datasets are tuned (D3).
4. **The evaluation is frozen after M1.** Changing `src/ovannot/eval/` needs an entry in its
   changelog and a re-run of the affected pipelines.
5. **Settings are explicit.** Every run declares `zero_shot` or `few_shot` (D4), and the
   runner hides from the pipeline what the setting does not allow. Never put
   `DatasetSpec.name` in a prompt.
6. **Licences.** Check every checkpoint's licence before using it and record it in
   `configs/models/<name>.yaml`.
7. **No notebooks for results.** Every reported number comes from a script in `scripts/` run
   on a versioned configuration.
8. **Small, atomic commits.** At the end of each milestone, write
   `docs/milestones/M<n>.md`: what works, what is TODO, numbers, decisions and why.

## Conventions

- **Boxes** are `(x_min, y_min, x_max, y_max)` in absolute pixels of the original image,
  everywhere. Only model wrappers convert to and from model formats (0–1000 relative,
  quantised tokens...). COCO `[x, y, w, h]` appears only in `core/coco_io.py`.
- **Image ids** are strings, unique within a dataset across splits (e.g. `"test/42"`).
- **Shared types** are in `src/ovannot/core/types.py`. Ground truth never reaches a pipeline:
  loaders keep `spec()`, `images()` and `ground_truth()` apart (`core/dataset.py`).
- **Plugins.** `pipelines/<id>/pipeline.py` and `datasets/<name>/loader.py` are imported by
  path and must register exactly their folder name. Folders starting with `_` are templates
  (D8).
- **Failures.** `Pipeline.annotate` never raises. Wrap each stage in
  `with stage(ann, "<name>") as st:` and check `st.failed`. Raise the errors of
  `core/errors.py` so failures get the right kind.
- **Configs** are YAML. `extends:` composes files, `key.sub=value` overrides from the command
  line, and pydantic models validate them (`extra="forbid"`: a typo is an error).
- **Prompts** live in `pipelines/<id>/prompts/*.jinja` (or `.txt`), rendered with
  `self.prompt(...)`, which fails on any missing variable.
- **Paths** come from configs or the environment: `OVANNOT_DATA` (datasets), `OVANNOT_CACHE`
  (model-call cache), `OVANNOT_ROOT` (repository). Secrets come from the environment only:
  `HF_TOKEN`, `ROBOFLOW_API_KEY`.

## Add a pipeline

1. Copy `pipelines/_template/` to `pipelines/<id>/`. Ids are `pNN_<idea>` for detection and
   `cNN_<idea>` for classification. Register the class with `@register_pipeline("<id>")`.
2. Declare hyper-parameters in the config class and their defaults in `config.yaml`.
3. Write the prompts in `prompts/`. Reuse stages from `src/ovannot/stages/` and add new
   reusable stages there, not in the pipeline. Never copy code between pipelines.
4. Fill the 11 sections of `README.md`, edit `diagram.mmd`, then run
   `uv run python scripts/render_diagrams.py`.
5. Add `configs/experiments/<id>_<dataset>_<setting>.yaml` (format fixed at M3).
6. Run `uv run pytest`: the contract tests check the folder, and the smoke test runs on
   3 images (`-m gpu` on the cluster for pipelines that need models).
7. Run on the dev subset, then `scripts/make_leaderboard.py` (M3).

A pipeline is **done** when:
- it is registered and configured;
- its README is complete with the rendered diagram;
- its prompts are in files;
- its smoke test passes;
- it has run on the dev subset;
- its results are in the leaderboard.

## Add a dataset

1. Create `datasets/<name>/` with `loader.py`, a subclass of `DatasetLoader` registered with
   `@register_dataset("<name>")`, and a `README.md` with the source, licence, splits,
   protocol and settings.
2. Add `configs/datasets/<name>.yaml` (and one file per subset, e.g. the dev subset).
3. For proprietary data, start from `datasets/_template_proprietary/` (M1) and follow its
   confidentiality checklist.

## Add a model

1. Check the model card: exact id and revision, licence, input and output format (coordinate
   convention), memory needs. Record them with their source in `configs/models/<name>.yaml`,
   including a `kind` that selects the wrapper.
2. Wrappers are classes registered with `@register_model_kind("<kind>")` (M2). Pipelines get
   them with `self.hub.get("<name>")` and never see endpoints or devices.
3. Add `serving/launch_<name>.sh` and a health check for served models (M2).

## Key commands

```bash
uv sync                                     # install (CPU-only base; GPU extras on the cluster)
uv run pytest                               # local tests (excludes -m gpu and -m network)
uv run pytest -m gpu                        # cluster: smoke tests that need model servers
uv run ruff check . && uv run ruff format .
uv run python papers/download_papers.py     # papers + status table in papers/README.md
uv run python scripts/render_diagrams.py    # diagrams (needs scripts/setup_mermaid.sh once)
uv run python scripts/run.py --exp configs/experiments/<name>.yaml [--limit N]  # resumable
uv run python scripts/make_leaderboard.py   # results/leaderboard.md and the README tables
# cluster: serve the models on the job's GPUs, smoke-test, then run (see serving/README.md)
SERVE="vllm:qwen3_vl_30b" sbatch scripts/slurm/run_experiment.sbatch --smoke --exp <yaml>
```

## Concurrency

The runner annotates images on a thread pool (`--workers`, 32 on the cluster): several
`annotate` calls of the **same pipeline object** run at the same time. Keep per-image state in
local variables of `annotate`, never in `self`; whatever `prepare` builds is read-only
afterwards. Model wrappers are thread-safe (one HTTP client each, `ModelHub.get` is locked).
The runner does not use processes: plugin classes are imported by path and cannot be pickled.

## Where things run

The laptop (x86_64, no GPU, little storage) holds the code, tests on synthetic data and the
papers. Models, datasets and runs live on the B200 cluster: compute nodes are aarch64, the
login node is x86_64. Keep GPU dependencies in optional extras, and every path and endpoint in
configs or environment variables.
