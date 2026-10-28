"""Run an experiment: one pipeline, a group of datasets, one setting, then score each dataset.

An experiment file (``configs/experiments/*.yaml``) holds::

    pipeline: p01_mllm_grounding_raw
    setting: zero_shot                      # D4
    dataset: configs/datasets/rf100vl_dev.yaml
    pipeline_cfg: {score_threshold: 0.5}    # overrides of the pipeline's config.yaml
    models: {qwen3_vl_30b: {...}}           # optional overrides of configs/models/*.yaml

Run folder ``results/runs/<date>_<pipeline>_<datasets>_<setting>_<subset>_<hash>/``:

- ``config.resolved.yaml``: everything above resolved, the git commit, the models used;
- ``metrics.json``: per-dataset metrics, their mean and the mean per domain;
- ``<dataset>/annotations.jsonl``: one :class:`ImageAnnotation` per line, appended as images
  finish. A run started again with the same config resumes from it;
- ``<dataset>/predictions.json`` (COCO results), ``metrics.json``, ``timings.json``,
  ``failures.jsonl``;
- ``run.log``.

Images are annotated by a thread pool (``workers``): pipelines must be thread-safe, which they
are when they keep per-image state local. Threads, not processes, because plugin classes are
imported by path and cannot be pickled.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from ovannot.core.config import config_hash, deep_merge, dump_yaml, load_config, load_yaml
from ovannot.core.dataset import DATASETS, discover_datasets, restrict_to_setting
from ovannot.core.pipeline import PIPELINES, discover_pipelines
from ovannot.core.types import ImageAnnotation, Setting, Task
from ovannot.eval.classification import evaluate_classification
from ovannot.eval.detection import evaluate_detection
from ovannot.models.hub import ModelHub
from ovannot.paths import repo_root, results_root
from ovannot.utils.io import write_json

log = logging.getLogger("ovannot.run")


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline: str
    setting: Setting
    dataset: str  # dataset-group config, relative to the repository root
    pipeline_cfg: dict[str, Any] = {}
    models: dict[str, dict[str, Any]] = {}


class DatasetGroupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loader: str
    datasets: list[str]
    splits: dict[str, str]  # subset (dev / test) -> split name
    loader_cfg: dict[str, Any] = {}
    split_loader_cfg: dict[str, dict[str, Any]] = {}  # split name -> extra loader config


def run_experiment(
    exp_path: str | Path,
    subset: str = "dev",
    limit: int | None = None,
    workers: int = 8,
    only: list[str] | None = None,
    overrides: list[str] = (),
    runs_root: Path | None = None,
) -> Path:
    exp_path = Path(exp_path)
    exp = ExperimentConfig.model_validate(load_config(exp_path, overrides))
    group = DatasetGroupConfig.model_validate(load_config(repo_root() / exp.dataset))
    if subset not in group.splits:
        raise ValueError(f"subset {subset!r} not in {sorted(group.splits)}")
    names = [n for n in group.datasets if not only or n in only]
    if only and len(names) != len(only):
        raise ValueError(f"unknown datasets in --only: {sorted(set(only) - set(names))}")

    discover_pipelines()
    discover_datasets()
    pipeline_cls = PIPELINES.get(exp.pipeline)
    loader_cls = DATASETS.get(group.loader)
    if pipeline_cls.task is not loader_cls.task:
        raise ValueError(
            f"{exp.pipeline} solves {pipeline_cls.task}, datasets are {loader_cls.task}"
        )
    pipeline_cfg = deep_merge(load_yaml(pipeline_cls.folder() / "config.yaml"), exp.pipeline_cfg)
    pipeline_cls.Config.model_validate(pipeline_cfg)  # fail before any model call
    hub = ModelHub.from_dir()
    if unknown := sorted(set(exp.models) - set(hub.names())):
        raise ValueError(f"models overridden but not in configs/models/: {unknown}")
    hub = ModelHub({n: deep_merge(hub.config(n), exp.models.get(n, {})) for n in hub.names()})

    identity = {
        "experiment": exp.model_dump(mode="json"),
        "datasets": names,
        "group": group.model_dump(),
        "pipeline_cfg": pipeline_cfg,
        "models": {n: hub.config(n) for n in hub.names()},
        "subset": subset,
        "limit": limit,
        # the code is part of a run's identity: a new commit starts a new run (cached model
        # calls make that cheap); uncommitted edits resume the current one
        "code": _git_state()["commit"],
    }
    run_hash = config_hash(identity)
    tag = f"{exp.pipeline}_{Path(exp.dataset).stem}_{exp.setting.value}_{subset}_{run_hash}"
    root = runs_root or results_root() / "runs"
    existing = sorted(root.glob(f"*_{tag}"))
    run_dir = existing[-1] if existing else root / f"{dt.date.today():%Y-%m-%d}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = _log_to(run_dir / "run.log")
    try:
        log.info("run %s (%s)", run_dir.name, "resumed" if existing else "new")
        per_dataset = {}
        for name in names:
            split = group.splits[subset]
            loader = loader_cls(
                {**group.loader_cfg, **group.split_loader_cfg.get(split, {}), "name": name}
            )
            per_dataset[name] = _run_dataset(
                loader,
                split,
                exp,
                pipeline_cls,
                pipeline_cfg,
                hub,
                run_dir / name,
                limit,
                workers,
            )
        resolved = {
            **identity,
            "run_hash": run_hash,
            "git": _git_state(),
            "models_used": hub.used(),
            "finished": dt.datetime.now().isoformat(timespec="seconds"),
        }
        dump_yaml(resolved, run_dir / "config.resolved.yaml")
        write_json(
            run_dir / "metrics.json", _summarise(exp, pipeline_cls.task, subset, per_dataset)
        )
        log.info("done: %s", run_dir)
    finally:
        logging.getLogger("ovannot").removeHandler(handler)
        handler.close()
    return run_dir


def _run_dataset(loader, split, exp, pipeline_cls, pipeline_cfg, hub, out: Path, limit, workers):
    out.mkdir(parents=True, exist_ok=True)
    full_spec = loader.spec()
    spec = restrict_to_setting(full_spec, exp.setting)
    images = loader.images(split)[:limit]
    pipeline = pipeline_cls(pipeline_cfg, hub)
    t0 = time.perf_counter()
    pipeline.prepare(spec, out / "prepare")
    prepare_s = time.perf_counter() - t0

    jsonl = out / "annotations.jsonl"
    done = _read_annotations(jsonl)
    todo = [img for img in images if img.image_id not in done]
    log.info("%s/%s: %d images, %d to annotate", spec.name, split, len(images), len(todo))
    lock = threading.Lock()

    def annotate(img):
        start = time.perf_counter()
        ann = pipeline.safe_annotate(img)
        ann.timings_s["_total"] = time.perf_counter() - start
        return ann

    with ThreadPoolExecutor(max_workers=workers) as pool, jsonl.open("a", encoding="utf-8") as f:
        futures = [pool.submit(annotate, img) for img in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            ann = fut.result()
            with lock:
                f.write(json.dumps(ann.to_dict(), ensure_ascii=False) + "\n")
                f.flush()
                done[ann.image_id] = ann
            if i % 25 == 0 or i == len(todo):
                log.info("%s: %d/%d", spec.name, i, len(todo))

    anns = [done[img.image_id] for img in images]
    if pipeline_cls.task is Task.DETECTION:
        threshold = float(getattr(pipeline.cfg, "score_threshold", 0.0))
        metrics, predictions = evaluate_detection(loader.ground_truth(split), anns, threshold)
    else:
        metrics, predictions = evaluate_classification(
            loader.ground_truth(split), anns, spec.class_names
        )
    metrics["domain"] = full_spec.domain  # the evaluator may know it; the pipeline did not
    metrics["failures"] = _failure_rates(anns)
    timings = _timings(anns, prepare_s)
    metrics["cost"] = {
        "time_per_image_s": {k: timings["per_image_s"].get(k, 0.0) for k in ("mean", "p95")},
        "model_calls_per_image": sum(timings["calls_per_image"].values()),
        "prepare_s": prepare_s,
    }
    write_json(out / "predictions.json", predictions, indent=None)
    write_json(out / "metrics.json", metrics)
    write_json(out / "timings.json", timings)
    with (out / "failures.jsonl").open("w", encoding="utf-8") as f:
        for a in anns:
            for fl in a.failures:
                f.write(json.dumps({"image_id": a.image_id, **fl.__dict__, "kind": fl.kind.value}))
                f.write("\n")
    log.info(
        "%s: %s, failure rate %.3f",
        spec.name,
        ", ".join(f"{k} {metrics[k]:.4f}" for k in SUMMARY_KEYS[pipeline_cls.task]),
        metrics["failures"]["rate"],
    )
    return metrics


def _read_annotations(path: Path) -> dict[str, ImageAnnotation]:
    done: dict[str, ImageAnnotation] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                ann = ImageAnnotation.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError):
                continue  # a line cut by a crash: the image is annotated again
            done[ann.image_id] = ann
    return done


def _failure_rates(anns: list[ImageAnnotation]) -> dict[str, Any]:
    n = max(len(anns), 1)
    by_kind: dict[str, int] = {}
    by_stage: dict[str, int] = {}
    for a in anns:
        for kind in {f.kind.value for f in a.failures}:
            by_kind[kind] = by_kind.get(kind, 0) + 1
        for st in {f.stage for f in a.failures}:
            by_stage[st] = by_stage.get(st, 0) + 1
    return {
        "rate": sum(bool(a.failures) for a in anns) / n,
        "by_kind": {k: v / n for k, v in sorted(by_kind.items())},
        "by_stage": {k: v / n for k, v in sorted(by_stage.items())},
    }


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
    }


def _timings(anns: list[ImageAnnotation], prepare_s: float) -> dict[str, Any]:
    stages = sorted({s for a in anns for s in a.timings_s if s != "_total"})
    models = sorted({m for a in anns for m in a.model_calls})
    token_keys = sorted({k for a in anns for k in a.tokens})
    n = max(len(anns), 1)
    return {
        "per_image_s": _stats([a.timings_s.get("_total", 0.0) for a in anns]),
        "per_stage_s": {s: _stats([a.timings_s.get(s, 0.0) for a in anns]) for s in stages},
        "prepare_s": prepare_s,
        "calls_per_image": {m: sum(a.model_calls.get(m, 0) for a in anns) / n for m in models},
        "tokens_per_image": {k: sum(a.tokens.get(k, 0) for a in anns) / n for k in token_keys},
    }


SUMMARY_KEYS = {
    Task.DETECTION: ("mAP", "AP50", "precision@0.5", "recall@0.5", "f1@0.5"),
    Task.CLASSIFICATION: ("accuracy", "closed_accuracy", "macro_f1", "none_f1", "aupr"),
}


def _summarise(
    exp: ExperimentConfig, task: Task, subset: str, per_dataset: dict[str, dict]
) -> dict:
    keys = SUMMARY_KEYS[task]
    domains: dict[str, list[str]] = {}
    for name, m in per_dataset.items():
        domains.setdefault(m["domain"], []).append(name)

    def mean(names: list[str]) -> dict[str, float]:
        return {k: float(np.mean([per_dataset[n][k] for n in names])) for k in keys}

    return {
        "pipeline": exp.pipeline,
        "task": task.value,
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_group": Path(exp.dataset).stem,
        "setting": exp.setting.value,
        "subset": subset,
        "mean": mean(list(per_dataset)),
        "per_domain": {d: mean(ns) for d, ns in sorted(domains.items())},
        "per_dataset": {
            n: {
                **{k: m[k] for k in keys},
                "domain": m["domain"],
                "n_images": m["n_images"],
                "failure_rate": m["failures"]["rate"],
                "cost": m["cost"],
            }
            for n, m in per_dataset.items()
        },
    }


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_root(), capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def _log_to(path: Path) -> logging.Handler:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger = logging.getLogger("ovannot")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return handler
