import json
from pathlib import Path

from ovannot.core.types import (
    ClassPrediction,
    Detection,
    Exemplar,
    Failure,
    FailureKind,
    ImageAnnotation,
    to_jsonable,
)


def test_image_annotation_round_trips_through_json():
    ann = ImageAnnotation(
        image_id="test/3",
        detections=[
            Detection(
                bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
                class_name="crack",
                score=0.8,
                mask_rle={"size": [10, 10], "counts": "abc"},
                provenance=["sam3"],
                extras={"yes_prob": 0.9},
            )
        ],
        label=ClassPrediction(class_name=None, score=0.7, class_scores={"crack": 0.3}),
        timings_s={"propose": 1.5},
        failures=[Failure("verify", FailureKind.TIMEOUT, "slow")],
    )
    ann.record_usage("qwen", prompt_tokens=100, completion_tokens=5)
    ann.record_usage("qwen", prompt_tokens=50)

    restored = ImageAnnotation.from_dict(json.loads(json.dumps(ann.to_dict())))

    assert restored == ann
    assert restored.model_calls == {"qwen": 2}
    assert restored.tokens == {"qwen/prompt": 150, "qwen/completion": 5}
    assert isinstance(restored.detections[0].bbox_xyxy, tuple)
    assert restored.failures[0].kind is FailureKind.TIMEOUT


def test_to_jsonable_converts_paths_enums_and_tuples():
    ex = Exemplar(image_id="train/1", path=Path("/data/x.jpg"), bbox_xyxy=(0, 0, 1, 1))
    assert to_jsonable(ex) == {
        "image_id": "train/1",
        "path": "/data/x.jpg",
        "bbox_xyxy": [0, 0, 1, 1],
    }
    assert to_jsonable({"k": FailureKind.OOM}) == {"k": "oom"}
