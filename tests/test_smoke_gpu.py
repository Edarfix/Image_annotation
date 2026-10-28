"""Smoke test of every model-backed pipeline on 3 images of a dev dataset (cluster only).

Run inside scripts/slurm/run_experiment.sbatch --smoke, which starts the model servers. A
pipeline whose model server does not answer is skipped, with the reason.
"""

import httpx
import pytest

from ovannot.core.config import load_yaml
from ovannot.core.dataset import DATASETS, discover_datasets, restrict_to_setting
from ovannot.core.pipeline import PIPELINES, discover_pipelines
from ovannot.core.types import Setting
from ovannot.models.base import ModelConfig
from ovannot.models.hub import ModelHub
from ovannot.paths import repo_root

pytestmark = pytest.mark.gpu
DATASET = "into-the-vale"  # dev subset, 6 classes
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
    if not server_up(hub.config(cfg["model"])):
        pytest.skip(f"server of {cfg['model']} is not running")
    discover_datasets()
    loader = DATASETS.get("rf100vl")({"name": DATASET})
    spec = restrict_to_setting(loader.spec(), Setting.ZERO_SHOT)
    pipeline = cls(cfg, hub)
    pipeline.prepare(spec, tmp_path)
    for img in loader.images("valid")[:3]:
        ann = pipeline.safe_annotate(img)
        assert not ann.failures, ann.failures
        assert ann.model_calls, "no model call recorded"
        for d in ann.detections:
            assert d.class_name in spec.class_names
            x0, y0, x1, y1 = d.bbox_xyxy
            assert 0 <= x0 < x1 <= img.width and 0 <= y0 < y1 <= img.height
        print(pipeline_id, img.image_id, len(ann.detections), "detections", ann.timings_s)
    assert (repo_root() / "configs" / "models" / f"{cfg['model']}.yaml").exists()
