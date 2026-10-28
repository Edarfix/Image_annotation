"""c00: SigLIP 2 zero-shot classification with prompt ensembling; "none" by threshold."""

from __future__ import annotations

from pathlib import Path

from ovannot.core.pipeline import Pipeline, PipelineConfig, register_pipeline, stage
from ovannot.core.types import DatasetSpec, ImageAnnotation, ImageRecord, Task
from ovannot.stages.classification import class_texts, siglip_decision, text_scores


class C00Config(PipelineConfig):
    model: str  # contrastive model, name in configs/models/
    description_lines: int  # description lines used as extra texts per class (0: names only)
    none_threshold: float  # "none" when the top class's sigmoid probability is below this


@register_pipeline("c00_siglip2_zeroshot")
class C00SigLIP2ZeroShot(Pipeline):
    task = Task.CLASSIFICATION
    Config = C00Config

    def prepare(self, ds: DatasetSpec, workdir: Path) -> None:
        super().prepare(ds, workdir)
        self.texts = class_texts(self.prompt, ds, self.cfg.description_lines)

    def annotate(self, img: ImageRecord) -> ImageAnnotation:
        ann = ImageAnnotation(image_id=img.image_id)
        with stage(ann, "score") as st:
            logits = text_scores(self.hub.get(self.cfg.model), img, self.texts)
            ann.record_usage(self.cfg.model)
        if not st.failed:
            ann.label = siglip_decision(
                logits, self.cfg.none_threshold, self.ds.allows_none, self.cfg.model
            )
        return ann
