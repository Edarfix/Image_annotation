"""Dataset plugins: what a dataset provides, and what pipelines are allowed to see.

A dataset plugin is a folder ``datasets/<name>/`` whose ``loader.py`` defines a subclass of
:class:`DatasetLoader` decorated with ``@register_dataset("<name>")``. A loader exposes three
things, kept strictly apart:

- :meth:`DatasetLoader.spec`: everything a pipeline may know (classes, descriptions,
  instructions, exemplars). The runner passes it through :func:`restrict_to_setting` before
  handing it to the pipeline.
- :meth:`DatasetLoader.images`: the images of a split, to annotate.
- :meth:`DatasetLoader.ground_truth`: the labels of a split, given to the evaluator only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

from ovannot.core.registry import Registry, discover_plugins, make_register
from ovannot.core.types import ClassSpec, DatasetSpec, ImageRecord, Setting, Task
from ovannot.paths import repo_root


@dataclass
class DetectionGroundTruth:
    coco: dict[str, Any]  # COCO dict restricted to the split; image ids map to ImageRecords
    image_ids: dict[str, int]  # ImageRecord.image_id -> COCO image id


@dataclass
class ClassificationGroundTruth:
    labels: dict[str, str | None]  # image_id -> class name, None means "none of the classes"


GroundTruth = DetectionGroundTruth | ClassificationGroundTruth


class DatasetLoader(ABC):
    id: ClassVar[str] = ""  # set by @register_dataset
    task: ClassVar[Task]

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    @abstractmethod
    def spec(self) -> DatasetSpec:
        """The full spec, with everything any setting may use."""

    @abstractmethod
    def images(self, split: str) -> list[ImageRecord]:
        """The images of ``split``, in a deterministic order."""

    @abstractmethod
    def ground_truth(self, split: str) -> GroundTruth:
        """The labels of ``split``. Only the evaluator may call this."""


DATASETS: Registry[DatasetLoader] = Registry("dataset")


def _check_loader(cls: type) -> None:
    if not (isinstance(cls, type) and issubclass(cls, DatasetLoader)):
        raise TypeError(f"{cls!r} must subclass DatasetLoader")
    if not isinstance(getattr(cls, "task", None), Task):
        raise TypeError(f"{cls.__name__} must set the class attribute 'task' to a Task")


register_dataset = make_register(DATASETS, _check_loader)


def discover_datasets(root: Path | None = None) -> list[str]:
    """Import every ``datasets/<name>/loader.py`` so their loaders are registered."""
    return discover_plugins(
        root or repo_root() / "datasets", "loader.py", "ovannot_plugins.datasets", DATASETS
    )


def restrict_to_setting(spec: DatasetSpec, setting: Setting) -> DatasetSpec:
    """Return a copy of ``spec`` containing only what ``setting`` allows (decision D4).

    ``zero_shot`` keeps the class names only; ``few_shot`` adds the descriptions, annotator
    instructions and exemplars. The domain is never passed: any other input defines a separate
    setting. ``spec.name`` is kept because it identifies the dataset (cache keys, concept-card
    folders); pipelines must never put it in a prompt. The result shares no mutable state
    with ``spec``.
    """
    spec = deepcopy(spec)
    if setting is Setting.ZERO_SHOT:
        return replace(
            spec,
            classes=[ClassSpec(name=c.name) for c in spec.classes],
            annotator_instructions=None,
            domain="",
            setting=setting,
        )
    if setting is Setting.FEW_SHOT:
        return replace(spec, domain="", setting=setting)
    raise ValueError(f"unknown setting {setting!r}")
