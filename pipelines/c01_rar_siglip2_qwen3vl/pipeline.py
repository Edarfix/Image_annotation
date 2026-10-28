"""c01: retrieve-and-rank classification (RAR): SigLIP 2 shortlists, Qwen3-VL chooses (README)."""

from __future__ import annotations

from pathlib import Path

from ovannot.core.pipeline import Pipeline, PipelineConfig, register_pipeline, stage
from ovannot.core.types import ClassPrediction, DatasetSpec, ImageAnnotation, ImageRecord, Task
from ovannot.stages.classification import (
    class_texts,
    letters,
    multiple_choice,
    siglip_decision,
    text_scores,
)


class C01Config(PipelineConfig):
    retriever: str  # contrastive model, name in configs/models/
    ranker: str  # MLLM, name in configs/models/
    top_k: int  # candidates shown to the MLLM
    description_lines: int  # description lines per class (SigLIP texts)
    mllm_description_lines: int  # description lines shown to the MLLM per candidate
    none_threshold: float  # "none" also when the best class's MLLM probability is below this
    fallback_none_threshold: float  # SigLIP threshold if the MLLM stage fails (as in c00)


@register_pipeline("c01_rar_siglip2_qwen3vl")
class C01RARSigLIP2Qwen3VL(Pipeline):
    task = Task.CLASSIFICATION
    Config = C01Config

    def prepare(self, ds: DatasetSpec, workdir: Path) -> None:
        super().prepare(ds, workdir)
        self.texts = class_texts(self.prompt, ds, self.cfg.description_lines)
        self.short = {
            c.name: " ".join(c.description.splitlines()[: self.cfg.mllm_description_lines])
            for c in ds.classes
        }

    def annotate(self, img: ImageRecord) -> ImageAnnotation:
        ann = ImageAnnotation(image_id=img.image_id)
        with stage(ann, "retrieve") as st:
            logits = text_scores(self.hub.get(self.cfg.retriever), img, self.texts)
            ann.record_usage(self.cfg.retriever)
        if st.failed:
            return ann
        fallback = siglip_decision(
            logits, self.cfg.fallback_none_threshold, self.ds.allows_none, self.cfg.retriever
        )
        candidates = sorted(logits, key=logits.get, reverse=True)[: self.cfg.top_k]
        options = [*candidates, None] if self.ds.allows_none else candidates
        prompt = self.prompt(
            "choose.jinja",
            options=[
                {"letter": k, "name": o, "description": self.short.get(o, "") if o else ""}
                for k, o in zip(letters(len(options)), options, strict=True)
            ],
        )
        with stage(ann, "rank") as st:
            probs = multiple_choice(self.hub.get(self.cfg.ranker), img, prompt, options, ann)
        if st.failed:
            fallback.extras["fallback"] = "retriever"
            ann.label = fallback
            return ann
        known = {name: p for name, p in probs.items() if name is not None}
        best = max(known, key=known.get)
        p_none = probs.get(None, 0.0)
        is_none = self.ds.allows_none and (
            p_none > known[best] or known[best] < self.cfg.none_threshold
        )
        ann.label = ClassPrediction(
            class_name=None if is_none else best,
            score=p_none if is_none else known[best],
            class_scores={name: known.get(name, 0.0) for name in self.ds.class_names},
            provenance=[self.cfg.retriever, self.cfg.ranker],
            extras={
                "candidates": candidates,
                "p_none": p_none,
                "retriever_top": fallback.class_name,
            },
        )
        return ann
