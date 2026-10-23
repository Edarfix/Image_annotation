"""Contract checks for every pipeline folder (the template included).

These automate the parts of the definition of "done" (AGENTS.md) that can be checked without
models: registration, valid config, prompts in files, complete README, up-to-date diagram, and
a run of ``annotate`` that returns a well-formed result.
"""

import re
from pathlib import Path

import pytest
from PIL import Image

from ovannot.core.config import load_yaml
from ovannot.core.pipeline import PIPELINES, discover_pipelines
from ovannot.core.types import ClassSpec, DatasetSpec, ImageAnnotation, ImageRecord, Setting
from ovannot.models.hub import ModelHub

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = [
    "Summary",
    "Diagram",
    "Models",
    "Prompts",
    "Step-by-step walkthrough",
    "Rationale",
    "Failure structure",
    "Cost",
    "Results",
    "Known limitations and next steps",
    "Changelog",
]
PIPELINE_IDS = discover_pipelines(include_templates=True)


def test_template_is_discovered_only_on_request():
    assert "_template" in PIPELINE_IDS
    assert "_template" not in discover_pipelines()


@pytest.mark.parametrize("pipeline_id", PIPELINE_IDS)
def test_folder_is_complete(pipeline_id, load_script):
    cls = PIPELINES.get(pipeline_id)
    folder = cls.folder()
    assert folder.name == pipeline_id

    cls.Config.model_validate(load_yaml(folder / "config.yaml"))

    prompts = list((folder / "prompts").glob("*"))
    assert prompts, "prompts live in prompts/, never in the code"

    readme = (folder / "README.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (\d+)\. (.+)$", readme, flags=re.M)
    assert [(int(n), title.strip()) for n, title in headings] == list(enumerate(SECTIONS, start=1))
    assert "<!-- BEGIN results" in readme and "<!-- END results -->" in readme

    render = load_script("scripts/render_diagrams.py")
    assert render.is_up_to_date(folder / "diagram.mmd"), "run scripts/render_diagrams.py"


@pytest.mark.parametrize("pipeline_id", PIPELINE_IDS)
def test_smoke_run_without_models(pipeline_id, tmp_path):
    """Pipelines that need no model run here; the others are smoke-tested on the cluster."""
    cls = PIPELINES.get(pipeline_id)
    if getattr(cls, "requires_models", True) and pipeline_id != "_template":
        pytest.skip("needs model servers (smoke-tested with -m gpu on the cluster)")
    image_path = tmp_path / "img.png"
    Image.new("RGB", (64, 48), "gray").save(image_path)
    spec = DatasetSpec(
        name="toy",
        task=cls.task,
        classes=[ClassSpec("crack"), ClassSpec("dent")],
        splits={"test": ["test/0"]},
        setting=Setting.ZERO_SHOT,
    )
    pipeline = cls(load_yaml(cls.folder() / "config.yaml"), ModelHub({}))
    pipeline.prepare(spec, tmp_path / "work")
    ann = pipeline.safe_annotate(ImageRecord("test/0", image_path, 64, 48))
    assert isinstance(ann, ImageAnnotation)
    assert ann.image_id == "test/0"
    assert not ann.failures
