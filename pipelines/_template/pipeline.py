"""Template pipeline: copy this folder to ``pipelines/<id>/`` and replace the TODOs.

Checklist (details in AGENTS.md, "Add a pipeline"):

1. Name the folder after the pipeline id (e.g. ``p05_my_idea``) and register that same id.
2. Set ``task``; declare hyper-parameters in the config class and their defaults in
   ``config.yaml``.
3. Put every prompt in ``prompts/`` and render it with ``self.prompt("<file>", ...)``.
4. Compose stages from :mod:`ovannot.stages`; wrap each one in ``with stage(ann, "<name>")``
   so that its time and failures are recorded.
5. Fill ``README.md`` (11 sections) and ``diagram.mmd``, then run
   ``scripts/render_diagrams.py``.

As shipped, this template runs end to end without any model: it renders its prompt and
returns no detections.
"""

from __future__ import annotations

from pathlib import Path

from ovannot.core.pipeline import Pipeline, PipelineConfig, register_pipeline, stage
from ovannot.core.types import DatasetSpec, ImageAnnotation, ImageRecord, Task


class TemplateConfig(PipelineConfig):
    score_threshold: float  # TODO: this pipeline's hyper-parameters (defaults in config.yaml)


@register_pipeline("_template")
class TemplatePipeline(Pipeline):
    task = Task.DETECTION
    Config = TemplateConfig
    requires_models = False  # TODO: delete this line once the pipeline calls models

    def prepare(self, ds: DatasetSpec, workdir: Path) -> None:
        super().prepare(ds, workdir)
        # TODO: build per-dataset artefacts once (e.g. concept cards) under `workdir`, and
        # reuse them when they already exist.

    def annotate(self, img: ImageRecord) -> ImageAnnotation:
        ann = ImageAnnotation(image_id=img.image_id)
        with stage(ann, "propose") as proposed:
            prompt = self.prompt("propose.jinja", class_names=self.ds.class_names)
            # TODO: call a model through self.hub, e.g. self.hub.get("<model name>"), parse
            # its answer into Detection objects and record the cost with ann.record_usage().
            del prompt
        if proposed.failed:
            return ann
        return ann
