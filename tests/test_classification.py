"""Task 1: metrics, ELEVATER loader, classification stages and pipelines with fake models."""

import json
import math

import numpy as np
import pytest
import yaml
from PIL import Image

from ovannot.core.config import load_yaml
from ovannot.core.dataset import (
    DATASETS,
    ClassificationGroundTruth,
    discover_datasets,
    restrict_to_setting,
)
from ovannot.core.pipeline import PIPELINES, discover_pipelines
from ovannot.core.types import (
    ClassPrediction,
    ClassSpec,
    DatasetSpec,
    FailureKind,
    ImageAnnotation,
    ImageRecord,
    Setting,
    Task,
)
from ovannot.eval.classification import evaluate_classification, open_set_curve
from ovannot.models.hub import ModelHub, register_model_kind
from ovannot.models.mllm_client import ChatResult
from ovannot.runner.run import run_experiment
from ovannot.utils.io import read_json

discover_pipelines()
discover_datasets()


# --- metrics ------------------------------------------------------------------------------


def pred(image_id, name, scores):
    return ImageAnnotation(
        image_id,
        label=ClassPrediction(name, max(scores.values()) if name else 0.9, class_scores=scores),
    )


def test_classification_metrics_with_none():
    gt = ClassificationGroundTruth({"a": "cat", "b": "dog", "c": None, "d": None})
    anns = [
        pred("a", "cat", {"cat": 0.9, "dog": 0.1}),  # right
        pred("b", "cat", {"cat": 0.6, "dog": 0.4}),  # wrong class
        pred("c", None, {"cat": 0.2, "dog": 0.1}),  # right "none"
        pred("d", "dog", {"cat": 0.1, "dog": 0.95}),  # open-set error, very confident
    ]
    m, predictions = evaluate_classification(gt, anns, ["cat", "dog"])
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["closed_accuracy"] == pytest.approx(0.5)
    assert (m["none_precision"], m["none_recall"]) == (1.0, 0.5)
    assert m["n_open_set"] == 2
    # positives: the TP "a" (0.9); negatives: "c" (0.2) and "d" (0.95) -> d ranks first
    assert m["aupr"] == pytest.approx(0.5)
    assert predictions[2]["class_name"] is None


def test_open_set_curve():
    perfect = open_set_curve(np.array([0.9, 0.8]), np.array([0.1, 0.2]))
    assert perfect == {"aupr": 1.0, "p_at_95r": 1.0, "r_at_95p": 1.0}
    bad = open_set_curve(np.array([0.1]), np.array([0.9]))
    assert bad["aupr"] == pytest.approx(0.5) and bad["r_at_95p"] is None
    tied = open_set_curve(np.array([0.5]), np.array([0.5]))  # ties count together
    assert tied["aupr"] == pytest.approx(0.5)


# --- ELEVATER loader ------------------------------------------------------------------------


@pytest.fixture
def elevater_root(tmp_path):
    root = tmp_path / "eurosat"
    for split in ("val", "test"):
        (root / split / "images").mkdir(parents=True)
        rows = []
        for i in range(20):
            name = f"img{i}.png"
            Image.new("RGB", (8 + i, 8)).save(root / split / "images" / name)
            rows.append({"file": name, "label": i % 10})
        (root / split / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def test_elevater_open_set_split(elevater_root, load_script):
    loader = DATASETS.get("elevater")({"name": "eurosat", "root": str(elevater_root)})
    spec = loader.spec()
    assert spec.allows_none and len(spec.classes) == 8  # 20% of 10 classes withheld
    assert spec.classes[0].name == "annual crop land"
    assert spec.classes[0].description.splitlines()[0] == "arable land"  # duplicates removed
    gt = loader.ground_truth("val")
    withheld = {loader.classes[i]["name"] for i in loader.withheld}
    assert sum(v is None for v in gt.labels.values()) == 4  # 2 classes x 2 images
    assert not withheld & set(spec.class_names)
    images = loader.images("test")
    assert (images[3].image_id, images[3].width) == ("test/3", 11)
    again = DATASETS.get("elevater")({"name": "eurosat", "root": str(elevater_root)})
    assert again.withheld == loader.withheld  # seeded
    sampled = DATASETS.get("elevater")(
        {"name": "eurosat", "root": str(elevater_root), "max_images": 5}
    )
    assert len(sampled.images("val")) == 5
    closed = DATASETS.get("elevater")(
        {"name": "eurosat", "root": str(elevater_root), "open_set": False}
    )
    assert len(closed.spec().classes) == 10 and not closed.spec().allows_none
    described = restrict_to_setting(spec, Setting.DESCRIBED)
    assert described.classes[0].description and not described.classes[0].exemplars


# --- pipelines with fake models -------------------------------------------------------------

CLASSES = ["forest", "river", "highway"]
SIGLIP_LOGITS = {"forest": 2.0, "river": 0.5, "highway": -1.0}
MLLM = {"text": "B", "top": [("B", -0.1), ("A", -2.5), (" D", -3.0), ("hello", -1.0)]}


@register_model_kind("test_fake_siglip")
class FakeSigLIP:
    def __init__(self, name, cfg):
        self.name = name

    def logits(self, image, texts):
        # the logit of a text is its class's logit (each text starts with "a photo of a <class>")
        return np.array(
            [next((v for k, v in SIGLIP_LOGITS.items() if k in t), -5.0) for t in texts]
        )


@register_model_kind("test_fake_ranker")
class FakeRanker:
    def __init__(self, name, cfg):
        self.name = name
        self.prompts = []

    def chat(self, messages, images=(), logprobs=False, **gen):
        assert logprobs and gen == {"max_tokens": 1}
        self.prompts.append(messages[-1]["content"])
        top = [{"token": t, "logprob": lp} for t, lp in MLLM["top"]]
        return ChatResult(MLLM["text"], 100, 1, logprobs=[{"token": "B", "top_logprobs": top}])


def hub():
    return ModelHub(
        {
            "siglip2_so400m": {"kind": "test_fake_siglip"},
            "qwen3_vl_30b": {"kind": "test_fake_ranker"},
        }
    )


SPEC = DatasetSpec(
    "toy",
    Task.CLASSIFICATION,
    [ClassSpec(n, f"{n} seen from a satellite\nsecond line") for n in CLASSES],
    {"val": ["val/0"]},
    allows_none=True,
    setting=Setting.DESCRIBED,
)


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "img.png"
    Image.new("RGB", (16, 16)).save(path)
    return ImageRecord("val/0", path, 16, 16)


def build(pipeline_id, h, tmp_path, **overrides):
    cls = PIPELINES.get(pipeline_id)
    p = cls({**load_yaml(cls.folder() / "config.yaml"), **overrides}, h)
    p.prepare(SPEC, tmp_path)
    return p


def test_c00_takes_the_top_siglip_class(image, tmp_path):
    ann = build("c00_siglip2_zeroshot", hub(), tmp_path).safe_annotate(image)
    assert ann.label.class_name == "forest"
    assert ann.label.score == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert set(ann.label.class_scores) == set(CLASSES)
    strict = build("c00_siglip2_zeroshot", hub(), tmp_path, none_threshold=0.95)
    assert strict.safe_annotate(image).label.class_name is None


def test_c01_ranks_the_shortlist_with_letter_probabilities(image, tmp_path):
    h = hub()
    ann = build("c01_rar_siglip2_qwen3vl", h, tmp_path, top_k=3).safe_annotate(image)
    assert not ann.failures
    # options: A forest, B river, C highway, D none; B wins
    pa, pb, pd = math.exp(-2.5), math.exp(-0.1), math.exp(-3.0)
    total = pa + pb + pd
    assert ann.label.class_name == "river"
    assert ann.label.score == pytest.approx(pb / total)
    assert ann.label.class_scores["forest"] == pytest.approx(pa / total)
    assert ann.label.class_scores["highway"] == 0.0
    assert ann.label.extras["p_none"] == pytest.approx(pd / total)
    prompt = h.get("qwen3_vl_30b").prompts[0]
    assert "A. forest (forest seen from a satellite second line)" in prompt
    assert "D. None of the above" in prompt and "toy" not in prompt


def test_c01_falls_back_to_the_retriever_when_the_ranker_fails(image, tmp_path):
    MLLM.update(text="I think it is a forest", top=[("I", -0.1)])
    try:
        ann = build("c01_rar_siglip2_qwen3vl", hub(), tmp_path).safe_annotate(image)
    finally:
        MLLM.update(text="B", top=[("B", -0.1), ("A", -2.5), (" D", -3.0), ("hello", -1.0)])
    assert [(f.stage, f.kind) for f in ann.failures] == [("rank", FailureKind.PARSE_ERROR)]
    assert ann.label.class_name == "forest" and ann.label.extras["fallback"] == "retriever"


def test_runner_classification_end_to_end(elevater_root, tmp_path, monkeypatch):
    monkeypatch.setenv("OVANNOT_CACHE", str(tmp_path / "cache"))
    loader = DATASETS.get("elevater")({"name": "eurosat", "root": str(elevater_root)})
    first_known = loader.spec().class_names[0]
    SIGLIP_LOGITS.clear()
    SIGLIP_LOGITS.update({first_known: 3.0})  # every image: the first known class
    try:
        group = tmp_path / "group.yaml"
        group.write_text(
            yaml.safe_dump(
                {
                    "loader": "elevater",
                    "datasets": ["eurosat"],
                    "splits": {"dev": "val"},
                    "loader_cfg": {"root": str(elevater_root)},
                    "split_loader_cfg": {"val": {"max_images": 10}},
                }
            )
        )
        exp = tmp_path / "exp.yaml"
        exp.write_text(
            yaml.safe_dump(
                {
                    "pipeline": "c00_siglip2_zeroshot",
                    "setting": "described",
                    "dataset": str(group),
                    "models": {"siglip2_so400m": {"kind": "test_fake_siglip"}},
                }
            )
        )
        monkeypatch.setattr(
            "ovannot.models.hub.ModelHub.from_dir",
            classmethod(lambda cls, path=None: hub()),
        )
        run_dir = run_experiment(exp, runs_root=tmp_path / "runs", workers=2)
    finally:
        SIGLIP_LOGITS.clear()
        SIGLIP_LOGITS.update({"forest": 2.0, "river": 0.5, "highway": -1.0})
    m = read_json(run_dir / "metrics.json")
    assert m["task"] == "classification" and m["per_dataset"]["eurosat"]["n_images"] == 10
    d = read_json(run_dir / "eurosat" / "metrics.json")
    assert d["closed_accuracy"] == pytest.approx(
        d["per_class"][first_known]["n"] / (10 - d["n_open_set"])
    )
    assert d["none_recall"] == 0.0  # c00 never says "none" with a 3.0 logit
