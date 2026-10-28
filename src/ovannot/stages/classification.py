"""Classification stages (task 1): image-text scoring and MLLM multiple choice.

- :func:`text_scores`: one score per class from a contrastive model (SigLIP), averaging the
  logits of several texts per class (the class name in templates, and the name with each line
  of its description: prompt ensembling as in CLIP and CuPL, and ELEVATER's knowledge prompts).
- :func:`multiple_choice`: the MLLM sees the image and lettered options (class names with
  their descriptions, and optionally "none of these"), answers one letter, and the
  probability of each option is read from the log-probabilities of its first answer token.
"""

from __future__ import annotations

import math
import string
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from ovannot.core.errors import ParseError
from ovannot.core.types import ClassPrediction, DatasetSpec, ImageAnnotation, ImageRecord


def text_scores(siglip, img: ImageRecord, texts: Mapping[str, Sequence[str]]) -> dict[str, float]:
    """Mean SigLIP logit of each class over its texts (``texts``: class name -> texts)."""
    flat = [t for ts in texts.values() for t in ts]
    logits = siglip.logits(img.path, flat)
    out, i = {}, 0
    for name, ts in texts.items():
        out[name] = float(np.mean(logits[i : i + len(ts)]))
        i += len(ts)
    return out


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def letters(n: int) -> list[str]:
    if n > len(string.ascii_uppercase):
        raise ValueError(f"at most 26 options, got {n}")
    return list(string.ascii_uppercase[:n])


def multiple_choice(
    mllm,
    img: ImageRecord,
    prompt: str,
    options: Sequence[str | None],
    ann: ImageAnnotation,
) -> dict[str | None, float]:
    """Probability of each option (a class name, or ``None`` for "none of these").

    ``prompt`` must present the options with the letters A, B, C... in the order of
    ``options`` and ask for the letter only. The probabilities are the softmax of the
    first-token log-probabilities of the letters (variants such as " A" are merged); a letter
    outside the top log-probabilities gets 0. Falls back to the first letter of the answer text
    (probability 1) if no option letter is among the top tokens.
    """
    result = mllm.chat(
        [{"role": "user", "content": prompt}], images=[img.path], logprobs=True, max_tokens=1
    )
    ann.record_usage(
        mllm.name, prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens
    )
    keys = letters(len(options))
    mass = dict.fromkeys(keys, 0.0)
    first = (result.logprobs or [{}])[0]
    for cand in first.get("top_logprobs", []) or []:
        token = cand["token"].strip().strip(".():").upper()
        if token in mass:
            mass[token] += math.exp(cand["logprob"])
    total = sum(mass.values())
    if total == 0:
        token = result.text.strip().strip(".():").upper()[:1]
        if token not in mass:
            raise ParseError(f"no option letter in the answer {result.text[:80]!r}")
        mass[token], total = 1.0, 1.0
    return {opt: mass[k] / total for k, opt in zip(keys, options, strict=True)}


def class_texts(render: Callable[..., str], ds: DatasetSpec, n_lines: int) -> dict[str, list[str]]:
    """Texts of every class from the pipeline's ``siglip_texts.jinja`` (``render`` is
    ``pipeline.prompt``): one text per non-empty rendered line, using at most ``n_lines`` lines
    of the class description."""
    out = {}
    for c in ds.classes:
        lines = [x for x in c.description.splitlines() if x.strip()][:n_lines]
        rendered = render("siglip_texts.jinja", class_name=c.name, description_lines=lines)
        out[c.name] = [t.strip() for t in rendered.splitlines() if t.strip()]
    return out


def siglip_decision(
    logits: Mapping[str, float], none_threshold: float, allows_none: bool, model: str
) -> ClassPrediction:
    """Top class by SigLIP probability; "none" if allowed and that probability is below
    ``none_threshold``. ``class_scores`` holds every class's sigmoid probability."""
    probs = {k: sigmoid(v) for k, v in logits.items()}
    top = max(probs, key=probs.get)
    is_none = allows_none and probs[top] < none_threshold
    return ClassPrediction(
        class_name=None if is_none else top,
        score=1 - probs[top] if is_none else probs[top],
        class_scores=probs,
        provenance=[model],
    )
