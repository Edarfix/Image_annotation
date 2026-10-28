"""p00 and p01 with fake models, then the runner end to end on a synthetic dataset."""

from pathlib import Path

import pytest
import yaml
from PIL import Image

from ovannot.core.config import load_yaml
from ovannot.core.dataset import (
    DatasetLoader,
    DetectionGroundTruth,
    register_dataset,
)
from ovannot.core.pipeline import PIPELINES, discover_pipelines
from ovannot.core.types import (
    ClassSpec,
    DatasetSpec,
    FailureKind,
    ImageRecord,
    Setting,
    Task,
)
from ovannot.models.hub import ModelHub, register_model_kind
from ovannot.models.mllm_client import ChatResult, OpenAIMLLM
from ovannot.models.sam3 import Grounding
from ovannot.runner.run import run_experiment
from ovannot.utils.io import read_json

discover_pipelines()
ANSWERS: dict[str, str] = {}


@register_model_kind("test_fake_mllm")
class FakeMLLM:
    """Answers from ANSWERS (keyed by image file name); converts boxes like the real wrapper."""

    def __init__(self, name, cfg):
        self.name = name
        self.cfg = type("Cfg", (), {"coord_format": "qwen3vl_rel1000"})()
        self.prompts = []

    def chat(self, messages, images=(), **_):
        self.prompts.append(messages[-1]["content"])
        return ChatResult(ANSWERS[Path(images[0]).name], prompt_tokens=5, completion_tokens=3)

    boxes_from_text = OpenAIMLLM.boxes_from_text


@register_model_kind("test_fake_sam3")
class FakeSAM3:
    def __init__(self, name, cfg):
        self.name = name

    def detect(self, image, phrases, threshold):
        return [
            Grounding(p, [(10.0, 10.0, 30.0, 30.0), (0, 0, 0, 5)], [0.9, 0.8])  # 2nd is invalid
            if p == "dent"
            else Grounding(p, [], [])
            for p in phrases
        ]


def hub():
    return ModelHub(
        {"qwen3_vl_30b": {"kind": "test_fake_mllm"}, "sam3": {"kind": "test_fake_sam3"}}
    )


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "img.png"
    Image.new("RGB", (200, 100), "gray").save(path)
    return ImageRecord("test/0", path, 200, 100)


SPEC = DatasetSpec(
    "toy",
    Task.DETECTION,
    [ClassSpec("dent"), ClassSpec("crack")],
    {"test": ["test/0"]},
    setting=Setting.ZERO_SHOT,
)


def build(pipeline_id, h, tmp_path):
    cls = PIPELINES.get(pipeline_id)
    p = cls(load_yaml(cls.folder() / "config.yaml"), h)
    p.prepare(SPEC, tmp_path)
    return p


def test_p01_keeps_known_labels_and_converts_boxes(image, tmp_path):
    ANSWERS["img.png"] = (
        '[{"bbox_2d": [100, 100, 500, 900], "label": "Dent"},'
        ' {"bbox_2d": [0, 0, 10, 10], "label": "scratch"},'
        ' {"bbox_2d": [5, 5, 5, 9], "label": "crack"}]'  # zero width: dropped
    )
    h = hub()
    ann = build("p01_mllm_grounding_raw", h, tmp_path).safe_annotate(image)
    assert not ann.failures
    assert [(d.class_name, d.bbox_xyxy, d.score) for d in ann.detections] == [
        ("dent", (20.0, 10.0, 100.0, 90.0), 1.0)
    ]
    assert ann.model_calls == {"qwen3_vl_30b": 1}
    assert ann.tokens == {"qwen3_vl_30b/prompt": 5, "qwen3_vl_30b/completion": 3}
    prompt = h.get("qwen3_vl_30b").prompts[0]
    assert "'dent, crack'" in prompt and "toy" not in prompt  # never the dataset name


def test_p01_records_parse_failures(image, tmp_path):
    ANSWERS["img.png"] = "Sorry, I see nothing."
    ann = build("p01_mllm_grounding_raw", hub(), tmp_path).safe_annotate(image)
    assert [(f.stage, f.kind) for f in ann.failures] == [("ground", FailureKind.PARSE_ERROR)]
    assert ann.detections == []


def test_p00_keeps_valid_sam3_boxes_with_their_scores(image, tmp_path):
    ann = build("p00_sam3_raw", hub(), tmp_path).safe_annotate(image)
    assert not ann.failures
    assert [(d.class_name, d.bbox_xyxy, d.score) for d in ann.detections] == [
        ("dent", (10.0, 10.0, 30.0, 30.0), 0.9)
    ]
    assert ann.detections[0].extras == {"phrase": "dent"}


# --- runner -------------------------------------------------------------------------------

BOXES = {0: (10.0, 10.0, 50.0, 40.0), 1: (60.0, 20.0, 90.0, 80.0), 2: (5.0, 5.0, 20.0, 20.0)}


@register_dataset("test_toy")
class ToyLoader(DatasetLoader):
    task = Task.DETECTION

    def spec(self):
        return DatasetSpec(
            self.cfg["name"],
            Task.DETECTION,
            [ClassSpec("dent", "a hollow")],
            {"valid": [f"valid/{i}" for i in BOXES]},
            domain="Industrial",
        )

    def images(self, split):
        root = Path(self.cfg["root"])
        return [ImageRecord(f"valid/{i}", root / f"{i}.png", 100, 100) for i in BOXES]

    def ground_truth(self, split):
        anns = [
            {
                "id": i + 1,
                "image_id": i,
                "category_id": 0,
                "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                "area": 1.0,
                "iscrowd": 0,
            }
            for i, b in BOXES.items()
        ]
        coco = {
            "images": [
                {"id": i, "file_name": f"{i}.png", "width": 100, "height": 100} for i in BOXES
            ],
            "annotations": anns,
            "categories": [{"id": 0, "name": "dent"}],
        }
        return DetectionGroundTruth(coco, {f"valid/{i}": i for i in BOXES})


def test_runner_end_to_end_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("OVANNOT_CACHE", str(tmp_path / "cache"))
    for i, (x0, y0, x1, y1) in BOXES.items():
        Image.new("RGB", (100, 100)).save(tmp_path / f"{i}.png")
        rel = [round(v * 10) for v in (x0, y0, x1, y1)]  # 100 px wide: pixel * 10 = rel 1000
        ANSWERS[f"{i}.png"] = f'[{{"bbox_2d": {rel}, "label": "dent"}}]'
    (tmp_path / "group.yaml").write_text(
        yaml.safe_dump(
            {
                "loader": "test_toy",
                "datasets": ["toy-a"],
                "splits": {"dev": "valid"},
                "loader_cfg": {"root": str(tmp_path)},
            }
        )
    )
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        yaml.safe_dump(
            {
                "pipeline": "p01_mllm_grounding_raw",
                "setting": "few_shot",
                "dataset": str(tmp_path / "group.yaml"),
                "models": {"qwen3_vl_30b": {"kind": "test_fake_mllm"}},
            }
        )
    )

    run_dir = run_experiment(exp, runs_root=tmp_path / "runs", limit=2, workers=2)
    m = read_json(run_dir / "metrics.json")
    assert m["mean"]["mAP"] == pytest.approx(1.0)
    assert m["per_domain"] == {"Industrial": m["mean"]}
    assert m["per_dataset"]["toy-a"]["n_images"] == 2
    resolved = load_yaml(run_dir / "config.resolved.yaml")
    assert resolved["experiment"]["setting"] == "few_shot"
    assert "qwen3_vl_30b" in resolved["models_used"]
    timings = read_json(run_dir / "toy-a" / "timings.json")
    assert timings["calls_per_image"] == {"qwen3_vl_30b": 1.0}
    assert len((run_dir / "toy-a" / "annotations.jsonl").read_text().splitlines()) == 2

    # same command again: resumed, nothing re-annotated
    ANSWERS.clear()  # any new model call would now fail
    assert run_experiment(exp, runs_root=tmp_path / "runs", limit=2, workers=2) == run_dir
    assert read_json(run_dir / "toy-a" / "metrics.json")["failures"]["rate"] == 0.0
