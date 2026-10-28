import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ovannot.core.dataset import restrict_to_setting
from ovannot.core.errors import ParseError
from ovannot.core.pipeline import Pipeline, PipelineConfig, register_pipeline, stage
from ovannot.core.types import (
    ClassSpec,
    DatasetSpec,
    Exemplar,
    FailureKind,
    ImageAnnotation,
    ImageRecord,
    Setting,
    Task,
)
from ovannot.models.hub import ModelHub

IMG = ImageRecord(image_id="test/0", path=Path("/nonexistent.jpg"), width=10, height=10)


class EchoConfig(PipelineConfig):
    threshold: float


@register_pipeline("test_echo")
class Echo(Pipeline):
    task = Task.DETECTION
    Config = EchoConfig

    def annotate(self, img):
        raise RuntimeError("boom")


def test_config_is_validated_and_unknown_keys_rejected():
    assert Echo({"threshold": 0.5}, ModelHub({})).cfg.threshold == 0.5
    with pytest.raises(ValidationError):
        Echo({"threshold": 0.5, "treshold": 0.6}, ModelHub({}))


def test_safe_annotate_turns_escaped_exceptions_into_failures():
    ann = Echo({"threshold": 0.5}, ModelHub({})).safe_annotate(IMG)
    assert ann.image_id == "test/0"
    assert [(f.stage, f.kind) for f in ann.failures] == [("annotate", FailureKind.OTHER)]


def test_register_rejects_classes_without_task():
    with pytest.raises(TypeError, match="task"):

        @register_pipeline("test_no_task")
        class NoTask(Pipeline):
            def annotate(self, img):
                return ImageAnnotation(img.image_id)


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (ParseError("bad json"), FailureKind.PARSE_ERROR),
        (json.JSONDecodeError("x", "doc", 0), FailureKind.PARSE_ERROR),
        (TimeoutError(), FailureKind.TIMEOUT),
        (RuntimeError("CUDA out of memory. Tried to allocate"), FailureKind.OOM),
        (ValueError("other"), FailureKind.OTHER),
    ],
)
def test_stage_records_failure_kind_and_time(exc, kind):
    ann = ImageAnnotation("i")
    with stage(ann, "verify") as st:
        raise exc
    assert st.failed
    assert [(f.stage, f.kind) for f in ann.failures] == [("verify", kind)]
    assert ann.timings_s["verify"] >= 0


def test_stage_accumulates_time_over_repeated_calls():
    ann = ImageAnnotation("i")
    for _ in range(3):
        with stage(ann, "verify") as st:
            pass
    assert not st.failed and not ann.failures
    assert set(ann.timings_s) == {"verify"}


def test_zero_shot_spec_keeps_only_class_names():
    spec = DatasetSpec(
        name="ds",
        task=Task.DETECTION,
        classes=[
            ClassSpec(
                "crack",
                "a thin dark line",
                [Exemplar("train/1", Path("a.jpg"), (0.0, 0.0, 1.0, 1.0))],
            )
        ],
        splits={"test": ["test/0"]},
        annotator_instructions="Label every crack.",
        domain="Industrial",
    )
    zero = restrict_to_setting(spec, Setting.ZERO_SHOT)
    assert zero.classes == [ClassSpec("crack")]
    assert zero.annotator_instructions is None and zero.domain == ""
    assert zero.setting is Setting.ZERO_SHOT
    few = restrict_to_setting(spec, Setting.FEW_SHOT)
    assert few.classes == spec.classes and few.annotator_instructions == "Label every crack."
    assert few.domain == ""  # any other input defines a new setting (D4)
    assert few.classes is not spec.classes  # pipelines cannot alter the loader's spec
    assert spec.setting is None and spec.domain == "Industrial"  # the original is untouched
