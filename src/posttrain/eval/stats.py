"""Uncertainty estimates and paired significance tests over per-problem scores."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Estimate:
    mean: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "ci_low": self.ci_low, "ci_high": self.ci_high}


def bootstrap_ci(
    values, num_resamples: int = 10_000, confidence: float = 0.95, seed: int = 0
) -> Estimate:
    """Percentile bootstrap over problems (the unit of sampling in a benchmark)."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("No values")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(num_resamples, x.size))].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return Estimate(float(x.mean()), float(low), float(high))


@dataclass
class PairedComparison:
    mean_a: float
    mean_b: float
    diff: Estimate  # b - a
    p_value: float  # two-sided paired sign-flip permutation test
    num_problems: int

    def to_dict(self) -> dict:
        return {
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "diff": self.diff.to_dict(),
            "p_value": self.p_value,
            "num_problems": self.num_problems,
        }


def paired_comparison(
    a, b, num_resamples: int = 10_000, confidence: float = 0.95, seed: int = 0
) -> PairedComparison:
    """Compare two models evaluated on the same problems, aligned by index."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired comparison requires scores on the same problems")
    d = b - a
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(num_resamples, d.size))
    null = np.abs((signs * d).mean(axis=1))
    observed = abs(d.mean())
    # Add-one smoothing keeps the Monte Carlo p-value strictly positive.
    p_value = (np.sum(null >= observed - 1e-12) + 1) / (num_resamples + 1)
    return PairedComparison(
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        diff=bootstrap_ci(d, num_resamples, confidence, seed),
        p_value=float(min(1.0, p_value)),
        num_problems=int(d.size),
    )


def mcnemar_exact(a, b) -> dict[str, float]:
    """Exact McNemar test for paired binary outcomes (e.g. greedy pass/fail)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        k = min(only_a, only_b)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return {"only_a": only_a, "only_b": only_b, "p_value": p}
