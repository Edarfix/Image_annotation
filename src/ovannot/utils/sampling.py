"""Seeded subsets that stay stable when the sample size changes."""

from __future__ import annotations

import random


def seeded_subset(key: str, n: int, k: int | None) -> list[int]:
    """Sorted indices of the first ``k`` items of a permutation of ``range(n)`` seeded by
    ``key`` (all of them if ``k`` is None or >= n). A smaller ``k`` gives a subset of a larger
    one, so a dev sample of 200 is included in an extraction of 500."""
    order = list(range(n))
    random.Random(key).shuffle(order)
    return sorted(order if k is None else order[:k])
