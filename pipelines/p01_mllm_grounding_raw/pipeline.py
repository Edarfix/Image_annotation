"""p01: direct grounding by an MLLM, all classes in one prompt, JSON boxes (README.md)."""

from __future__ import annotations

from ovannot.core.pipeline import Pipeline, PipelineConfig, register_pipeline, stage
from ovannot.core.types import ImageAnnotation, ImageRecord, Task
from ovannot.stages.grounding import mllm_grounding, top_k


class P01Config(PipelineConfig):
    model: str  # name in configs/models/
    score_threshold: float  # operating threshold for precision / recall / F1 at IoU 0.5
    max_detections: int  # per image, RF100-VL's maxDets


@register_pipeline("p01_mllm_grounding_raw")
class P01MLLMGroundingRaw(Pipeline):
    task = Task.DETECTION
    Config = P01Config

    def annotate(self, img: ImageRecord) -> ImageAnnotation:
        ann = ImageAnnotation(image_id=img.image_id)
        classes = self.ds.classes
        prompt = self.prompt(
            "ground.jinja",
            class_names=[c.name for c in classes],
            descriptions={c.name: c.description for c in classes if c.description},
            instructions=self.ds.annotator_instructions or "",
        )
        with stage(ann, "ground") as st:
            detections = mllm_grounding(
                self.hub.get(self.cfg.model), img, prompt, self.ds.class_names, ann
            )
        if not st.failed:
            ann.detections = top_k(detections, self.cfg.max_detections)
        return ann
