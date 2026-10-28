"""p00: SAM 3 prompted with the raw name of each class; the floor (README.md)."""

from __future__ import annotations

from ovannot.core.pipeline import Pipeline, PipelineConfig, register_pipeline, stage
from ovannot.core.types import ImageAnnotation, ImageRecord, Task
from ovannot.stages.grounding import sam3_grounding, top_k


class P00Config(PipelineConfig):
    model: str  # name in configs/models/
    sam3_threshold: float  # SAM 3 keeps instances scored above this (low: mAP needs recall)
    score_threshold: float  # operating threshold for precision / recall / F1 at IoU 0.5
    max_detections: int  # per image, RF100-VL's maxDets


@register_pipeline("p00_sam3_raw")
class P00SAM3Raw(Pipeline):
    task = Task.DETECTION
    Config = P00Config

    def annotate(self, img: ImageRecord) -> ImageAnnotation:
        ann = ImageAnnotation(image_id=img.image_id)
        phrases = {
            n: [self.prompt("phrase.jinja", class_name=n).strip()] for n in self.ds.class_names
        }
        with stage(ann, "ground") as st:
            detections = sam3_grounding(
                self.hub.get(self.cfg.model), img, phrases, self.cfg.sam3_threshold, ann
            )
        if not st.failed:
            ann.detections = top_k(detections, self.cfg.max_detections)
        return ann
