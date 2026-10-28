"""Smoke test of every model-backed pipeline on 3 images of a dev dataset (cluster only).

Run inside scripts/slurm/run_experiment.sbatch --smoke, which starts the model servers. A
pipeline whose model servers do not all answer is skipped, with the reason.
"""

import httpx
import pytest

from ovannot.core.config import load_yaml
from ovannot.core.dataset import DATASETS, discover_datasets, restrict_to_setting
from ovannot.core.pipeline import PIPELINES, discover_pipelines
from ovannot.core.types import Setting, Task
from ovannot.models.base import ModelConfig
from ovannot.models.hub import ModelHub

pytestmark = pytest.mark.gpu
DATASETS_BY_TASK = {  # (loader, config, split, setting)
    Task.DETECTION: ("rf100vl", {"name": "into-the-vale"}, "valid", Setting.ZERO_SHOT),
    Task.CLASSIFICATION: ("elevater", {"name": "eurosat"}, "val", Setting.DESCRIBED),
}
MODEL_KEYS = ("model", "retriever", "ranker")  # config keys that name models
PIPELINE_IDS = [p for p in discover_pipelines() if PIPELINES.get(p).requires_models]


def server_up(cfg: dict) -> bool:
    fields = {k: v for k, v in cfg.items() if k in ModelConfig.model_fields}
    url = ModelConfig.model_validate(fields).resolved_endpoint().removesuffix("/v1")
    try:
        return httpx.get(f"{url}/health", timeout=5, trust_env=False).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.parametrize("pipeline_id", PIPELINE_IDS)
def test_pipeline_runs_on_three_images(pipeline_id, tmp_path):
    cls = PIPELINES.get(pipeline_id)
    cfg = load_yaml(cls.folder() / "config.yaml")
    hub = ModelHub.from_dir()
    down = [cfg[k] for k in MODEL_KEYS if k in cfg and not server_up(hub.config(cfg[k]))]
    if down:
        pytest.skip(f"servers not running: {down}")
    discover_datasets()
    loader_name, loader_cfg, split, setting = DATASETS_BY_TASK[cls.task]
    loader = DATASETS.get(loader_name)(loader_cfg)
    spec = restrict_to_setting(loader.spec(), setting)
    pipeline = cls(cfg, hub)
    pipeline.prepare(spec, tmp_path)
    for img in loader.images(split)[:3]:
        ann = pipeline.safe_annotate(img)
        assert not ann.failures, ann.failures
        assert ann.model_calls, "no model call recorded"
        if cls.task is Task.DETECTION:
            for d in ann.detections:
                assert d.class_name in spec.class_names
                x0, y0, x1, y1 = d.bbox_xyxy
                assert 0 <= x0 < x1 <= img.width and 0 <= y0 < y1 <= img.height
            print(pipeline_id, img.image_id, len(ann.detections), "detections", ann.timings_s)
        else:
            label = ann.label
            assert label is not None and (
                label.class_name in spec.class_names or label.class_name is None
            )
            assert 0.0 <= label.score <= 1.0
            print(pipeline_id, img.image_id, label.class_name, round(label.score, 3), ann.timings_s)
