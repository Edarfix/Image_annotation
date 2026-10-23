"""The Pipeline interface, its registry, and the ``stage`` helper used inside pipelines.

A pipeline plugin is a folder ``pipelines/<id>/`` whose ``pipeline.py`` defines a subclass of
:class:`Pipeline` decorated with ``@register_pipeline("<id>")`` (see ``pipelines/_template``).
"""

from __future__ import annotations

import inspect
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from ovannot.core.errors import classify_exception
from ovannot.core.registry import Registry, discover_plugins, make_register
from ovannot.core.types import DatasetSpec, Failure, ImageAnnotation, ImageRecord, Task
from ovannot.models.hub import ModelHub
from ovannot.paths import repo_root
from ovannot.utils.prompts import render_prompt


class PipelineConfig(BaseModel):
    """Base class of pipeline hyper-parameters. Unknown keys are errors (typos fail fast)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Pipeline(ABC):
    """Base class of every annotation pipeline.

    Subclasses set ``task`` and, if they have hyper-parameters, ``Config`` (a
    :class:`PipelineConfig` subclass whose defaults are in the pipeline's ``config.yaml``).
    """

    id: ClassVar[str] = ""  # set by @register_pipeline
    task: ClassVar[Task]
    Config: ClassVar[type[PipelineConfig]] = PipelineConfig
    requires_models: ClassVar[bool] = True  # False: smoke-tested locally, without model servers

    def __init__(self, cfg: Mapping[str, Any], hub: ModelHub) -> None:
        self.cfg = self.Config.model_validate(dict(cfg))
        self.hub = hub
        self.ds: DatasetSpec | None = None
        self.workdir: Path | None = None

    def prepare(self, ds: DatasetSpec, workdir: Path) -> None:
        """Called once per dataset before any :meth:`annotate` call.

        Override to build per-dataset artefacts (e.g. concept cards) under ``workdir``; reuse
        them when they already exist, so that re-runs are cheap. Call ``super().prepare``.
        """
        self.ds = ds
        self.workdir = workdir

    @abstractmethod
    def annotate(self, img: ImageRecord) -> ImageAnnotation:
        """Annotate one image. Must not raise: record errors as failures (see :func:`stage`)."""

    def safe_annotate(self, img: ImageRecord) -> ImageAnnotation:
        """Runner entry point: :meth:`annotate`, with any escaped exception turned into a
        failure of the pseudo-stage ``annotate``."""
        try:
            return self.annotate(img)
        except Exception as exc:
            ann = ImageAnnotation(image_id=img.image_id)
            ann.failures.append(
                Failure("annotate", classify_exception(exc), f"{type(exc).__name__}: {exc}")
            )
            return ann

    @classmethod
    def folder(cls) -> Path:
        """The plugin folder of this pipeline (where config.yaml and prompts/ live)."""
        return Path(inspect.getfile(cls)).resolve().parent

    def prompt(self, name: str, **variables: object) -> str:
        """Render ``prompts/<name>`` from the pipeline folder."""
        return render_prompt(self.folder() / "prompts" / name, **variables)


PIPELINES: Registry[Pipeline] = Registry("pipeline")


def _check_pipeline(cls: type) -> None:
    if not (isinstance(cls, type) and issubclass(cls, Pipeline)):
        raise TypeError(f"{cls!r} must subclass Pipeline")
    if not isinstance(getattr(cls, "task", None), Task):
        raise TypeError(f"{cls.__name__} must set the class attribute 'task' to a Task")


register_pipeline = make_register(PIPELINES, _check_pipeline)


def discover_pipelines(root: Path | None = None, include_templates: bool = False) -> list[str]:
    """Import every ``pipelines/<id>/pipeline.py`` so their pipelines are registered."""
    return discover_plugins(
        root or repo_root() / "pipelines",
        "pipeline.py",
        "ovannot_plugins.pipelines",
        PIPELINES,
        include_templates=include_templates,
    )


@dataclass
class StageStatus:
    failed: bool = False


@contextmanager
def stage(ann: ImageAnnotation, name: str) -> Iterator[StageStatus]:
    """Time a stage of :meth:`Pipeline.annotate` and turn its exceptions into failures.

    The time spent is added to ``ann.timings_s[name]``. If the body raises, the exception is
    recorded in ``ann.failures`` (kind from :func:`classify_exception`) and swallowed; check
    ``status.failed`` to decide on a fallback::

        with stage(ann, "propose") as st:
            candidates = propose(...)
        if st.failed:
            return ann
    """
    status = StageStatus()
    start = time.perf_counter()
    try:
        yield status
    except Exception as exc:
        status.failed = True
        ann.failures.append(Failure(name, classify_exception(exc), f"{type(exc).__name__}: {exc}"))
    finally:
        ann.timings_s[name] = ann.timings_s.get(name, 0.0) + time.perf_counter() - start
