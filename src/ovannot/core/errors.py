"""Exceptions raised inside stages, and their mapping to failure kinds.

Stages raise these exceptions; :func:`ovannot.core.pipeline.stage` catches them and records a
:class:`~ovannot.core.types.Failure` with the matching kind, so pipelines never crash the run.
"""

from __future__ import annotations

import json

from ovannot.core.types import FailureKind


class OvannotError(Exception):
    """Base class of the errors raised on purpose by stages and model wrappers."""

    kind = FailureKind.OTHER


class ParseError(OvannotError):
    """A model answered, but its output could not be parsed (bad JSON, wrong schema)."""

    kind = FailureKind.PARSE_ERROR


class EmptyOutputError(OvannotError):
    """A model answered with nothing usable (no text, no candidates when some were needed)."""

    kind = FailureKind.EMPTY_OUTPUT


class ModelError(OvannotError):
    """The model server or library failed (HTTP error, crash, refused request)."""

    kind = FailureKind.MODEL_ERROR


class InvalidBoxError(OvannotError):
    """A box is degenerate, not finite, or outside the image beyond tolerance."""

    kind = FailureKind.INVALID_BOX


def classify_exception(exc: BaseException) -> FailureKind:
    """Map any exception to the failure kind recorded in ``failures.jsonl``."""
    if isinstance(exc, OvannotError):
        return exc.kind
    if isinstance(exc, TimeoutError):  # also covers asyncio.TimeoutError (Python >= 3.11)
        return FailureKind.TIMEOUT
    if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
        return FailureKind.OOM  # torch.cuda.OutOfMemoryError says "CUDA out of memory"
    if isinstance(exc, json.JSONDecodeError):
        return FailureKind.PARSE_ERROR
    # pydantic is a core dependency, but import lazily to keep this module cheap
    from pydantic import ValidationError

    if isinstance(exc, ValidationError):
        return FailureKind.PARSE_ERROR
    return FailureKind.OTHER
