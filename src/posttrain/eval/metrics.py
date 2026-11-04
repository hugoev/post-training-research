"""Correctness metrics aggregated over problems."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from posttrain.eval.stats import bootstrap_ci
from posttrain.execution import ExecutionResult, Status


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021): 1 - C(n-c, k) / C(n, k)."""
    if not 0 <= c <= n:
        raise ValueError(f"Need 0 <= c <= n, got n={n}, c={c}")
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} samples per problem, got {n}")
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def per_problem_scores(results: list[ExecutionResult], ks: list[int]) -> dict[str, float]:
    n = len(results)
    c = sum(r.passed for r in results)
    scores = {f"pass@{k}": pass_at_k(n, c, k) for k in ks if k <= n}
    scores["test_pass_rate"] = float(np.mean([r.pass_rate for r in results]))
    scores["compile_rate"] = float(np.mean([r.compiled for r in results]))
    # Executed without syntax errors, load-time exceptions, crashes or timeouts.
    scores["execution_rate"] = float(
        np.mean([r.status in (Status.PASSED, Status.FAILED) for r in results])
    )
    return scores


def aggregate(
    results_by_problem: dict[str, list[ExecutionResult]], ks: list[int], seed: int = 0
) -> dict[str, Any]:
    per_problem = {tid: per_problem_scores(rs, ks) for tid, rs in results_by_problem.items()}
    metric_names = sorted({name for scores in per_problem.values() for name in scores})
    summary: dict[str, Any] = {"num_problems": len(per_problem)}
    for name in metric_names:
        values = [s[name] for s in per_problem.values() if name in s]
        summary[name] = bootstrap_ci(values, seed=seed).to_dict()
    statuses = Counter(r.status.value for rs in results_by_problem.values() for r in rs)
    total = sum(statuses.values())
    summary["status_distribution"] = {k: v / total for k, v in sorted(statuses.items())}
    return summary
