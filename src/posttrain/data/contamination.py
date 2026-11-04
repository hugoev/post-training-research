"""N-gram overlap checks between training and evaluation problems."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from posttrain.data.problems import Problem


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def problem_text(problem: Problem) -> str:
    return f"{problem.description}\n{problem.reference_solution or ''}"


@dataclass
class OverlapMatch:
    eval_id: str
    train_id: str | None
    overlap: float  # fraction of the eval problem's n-grams found in that train problem


def find_overlaps(
    train: dict[str, str], evaluation: dict[str, str], n: int = 10
) -> list[OverlapMatch]:
    """For each eval text, the training text sharing the largest fraction of its n-grams."""
    index: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for train_id, text in train.items():
        for gram in _ngrams(_tokens(text), n):
            index[gram].add(train_id)

    matches = []
    for eval_id, text in evaluation.items():
        grams = _ngrams(_tokens(text), n)
        counts: dict[str, int] = defaultdict(int)
        for gram in grams:
            for train_id in index.get(gram, ()):
                counts[train_id] += 1
        if not grams or not counts:
            matches.append(OverlapMatch(eval_id, None, 0.0))
            continue
        best = max(counts, key=counts.__getitem__)
        matches.append(OverlapMatch(eval_id, best, counts[best] / len(grams)))
    return matches
