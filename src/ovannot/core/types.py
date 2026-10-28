"""Shared data contract between datasets, pipelines, the runner and the evaluator.

Conventions relied upon everywhere:

- Boxes are ``(x_min, y_min, x_max, y_max)`` in absolute pixel coordinates of the original
  image file (no normalisation, no resizing). Only model wrappers convert to and from
  model-specific formats.
- Image ids are strings, unique within a dataset (across all its splits).
- Ground truth never appears in these types: a pipeline only sees a :class:`DatasetSpec`
  (restricted to the declared setting) and :class:`ImageRecord` objects. Ground truth lives in
  :mod:`ovannot.core.dataset` and is handed to the evaluator only.

Two tasks share the same types (decision D2):

- detection: :attr:`ImageAnnotation.detections` holds every instance found;
- classification: :attr:`ImageAnnotation.label` holds one :class:`ClassPrediction`, whose
  ``class_name`` is ``None`` when the image shows none of the classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

BBox = tuple[float, float, float, float]
"""``(x_min, y_min, x_max, y_max)`` in absolute pixels."""


class Task(StrEnum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"


class Setting(StrEnum):
    """What a run is allowed to know about the classes (decision D4).

    - ``zero_shot``: class names only.
    - ``described``: class names and their descriptions (task 1: "classes with descriptions";
      decision D17).
    - ``few_shot``: class names, descriptions, annotator instructions and labelled exemplars.

    Anything else (dataset name, domain...) would define a new setting.
    """

    ZERO_SHOT = "zero_shot"
    DESCRIBED = "described"
    FEW_SHOT = "few_shot"


class FailureKind(StrEnum):
    PARSE_ERROR = "parse_error"
    TIMEOUT = "timeout"
    OOM = "oom"
    EMPTY_OUTPUT = "empty_output"
    MODEL_ERROR = "model_error"
    INVALID_BOX = "invalid_box"
    OTHER = "other"


# --- what a pipeline may know about a dataset -------------------------------------------------


@dataclass(frozen=True)
class Exemplar:
    """A labelled example of a class, available in few-shot settings only.

    For detection it is one annotated box; for classification the whole image is the example
    and ``bbox_xyxy`` is ``None``. Exemplar images may contain other, unannotated classes (in
    RF100-VL they are only guaranteed to be exhaustively annotated for one class).
    """

    image_id: str
    path: Path
    bbox_xyxy: BBox | None = None


@dataclass
class ClassSpec:
    name: str
    description: str = ""  # empty in zero-shot
    exemplars: list[Exemplar] = field(default_factory=list)  # empty in zero-shot


@dataclass
class DatasetSpec:
    """Everything a pipeline is allowed to know about a dataset."""

    name: str
    task: Task
    classes: list[ClassSpec]
    splits: dict[str, list[str]]  # split name -> image ids
    annotator_instructions: str | None = None  # dataset-level instructions, if any
    domain: str = ""
    allows_none: bool = False  # classification: "none of the classes" is a valid answer
    setting: Setting | None = None  # set when the spec is restricted to a setting

    @property
    def class_names(self) -> list[str]:
        return [c.name for c in self.classes]


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: Path
    width: int
    height: int


# --- what a pipeline produces ------------------------------------------------------------------


@dataclass
class Detection:
    bbox_xyxy: BBox  # absolute pixels, always
    class_name: str
    score: float  # mandatory, comparable across images (mAP ranks by it)
    mask_rle: dict | None = None  # COCO RLE, in the original image size
    provenance: list[str] = field(default_factory=list)  # models that produced / validated it
    extras: dict[str, Any] = field(default_factory=dict)  # e.g. attribute scores, rationale


@dataclass
class ClassPrediction:
    """The image-level decision of a classification pipeline.

    ``class_name`` is ``None`` when the pipeline decides that the image shows none of the
    classes (only meaningful if the dataset allows it). ``score`` is the confidence in that
    decision and must be comparable across images. ``class_scores`` optionally holds one score
    per class; it lets the evaluator sweep the "none" threshold without re-running models.
    """

    class_name: str | None
    score: float
    class_scores: dict[str, float] = field(default_factory=dict)
    provenance: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Failure:
    stage: str
    kind: FailureKind
    message: str


@dataclass
class ImageAnnotation:
    """Output of :meth:`Pipeline.annotate` for one image, with its cost and failures."""

    image_id: str
    detections: list[Detection] = field(default_factory=list)  # detection task
    label: ClassPrediction | None = None  # classification task
    timings_s: dict[str, float] = field(default_factory=dict)  # per stage
    model_calls: dict[str, int] = field(default_factory=dict)  # per model name
    tokens: dict[str, int] = field(default_factory=dict)  # "<model>/prompt", "<model>/completion"
    failures: list[Failure] = field(default_factory=list)

    def record_usage(
        self, model: str, *, calls: int = 1, prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> None:
        """Add the cost of model calls made for this image."""
        self.model_calls[model] = self.model_calls.get(model, 0) + calls
        for kind, n in (("prompt", prompt_tokens), ("completion", completion_tokens)):
            if n:
                key = f"{model}/{kind}"
                self.tokens[key] = self.tokens.get(key, 0) + n

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ImageAnnotation:
        raw_label = d.get("label")
        label = None
        if raw_label is not None:
            label = ClassPrediction(
                class_name=raw_label["class_name"],
                score=float(raw_label["score"]),
                class_scores={k: float(v) for k, v in raw_label.get("class_scores", {}).items()},
                provenance=list(raw_label.get("provenance", [])),
                extras=dict(raw_label.get("extras", {})),
            )
        return cls(
            image_id=d["image_id"],
            detections=[
                Detection(
                    bbox_xyxy=_bbox(x["bbox_xyxy"]),
                    class_name=x["class_name"],
                    score=float(x["score"]),
                    mask_rle=x.get("mask_rle"),
                    provenance=list(x.get("provenance", [])),
                    extras=dict(x.get("extras", {})),
                )
                for x in d.get("detections", [])
            ],
            label=label,
            timings_s={k: float(v) for k, v in d.get("timings_s", {}).items()},
            model_calls={k: int(v) for k, v in d.get("model_calls", {}).items()},
            tokens={k: int(v) for k, v in d.get("tokens", {}).items()},
            failures=[
                Failure(stage=f["stage"], kind=FailureKind(f["kind"]), message=f["message"])
                for f in d.get("failures", [])
            ],
        )


def _bbox(values: Any) -> BBox:
    x0, y0, x1, y1 = (float(v) for v in values)
    return (x0, y0, x1, y1)


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses, enums, paths, tuples and numpy values into plain JSON values."""
    if isinstance(value, np.generic | np.ndarray):  # model outputs: np.float32 scores...
        return value.tolist()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    return value
