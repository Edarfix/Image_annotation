import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "key",
    "arxiv_id",
    "version",
    "title",
    "acronym",
    "short_title",
    "year",
    "venue",
    "group",
    "role",
}


@pytest.fixture(scope="module")
def dp():
    spec = importlib.util.spec_from_file_location(
        "download_papers", ROOT / "papers/download_papers.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["download_papers"] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules["download_papers"]


def test_papers_yaml_is_well_formed(dp):
    papers = yaml.safe_load((ROOT / "papers/papers.yaml").read_text())["papers"]
    for p in papers:
        assert set(p) >= REQUIRED, f"{p.get('key')}: missing {REQUIRED - set(p)}"
        assert p["group"] in dp.GROUP_ORDER
        assert p["version"].startswith("v")
    for field in ("key", "arxiv_id"):
        values = [p[field] for p in papers]
        assert len(values) == len(set(values)), f"duplicate {field}"
    assert len({dp.pdf_path(p).name for p in papers}) == len(papers)


def test_normalise_handles_ligatures_accents_and_punctuation(dp):
    assert dp.normalise("Efﬁcient Sünderhauf: “Thinking” 3-D") == "efficientsunderhaufthinking3d"


def test_title_matching(dp):
    page = "arXiv:2511.16719v2\nSAM 3: Segment Any-\nthing with Concepts\nNicolas Carion et al."
    assert dp.title_matches("SAM 3: Segment Anything with Concepts", page)
    assert dp.title_matches(
        "SAM 3: Segment Anything with Concepts", page.replace("Concepts", "Concepfs")
    )
    assert not dp.title_matches("Grounding DINO: Marrying DINO with Grounded Pre-Training", page)
    assert not dp.title_matches("", page)
