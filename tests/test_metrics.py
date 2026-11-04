import itertools
import math

import numpy as np
import pytest

from posttrain.eval.metrics import aggregate, pass_at_k
from posttrain.eval.stats import bootstrap_ci, mcnemar_exact, paired_comparison
from posttrain.execution import ExecutionResult, Status


def brute_force_pass_at_k(n, c, k):
    samples = [True] * c + [False] * (n - c)
    subsets = list(itertools.combinations(samples, k))
    return sum(any(s) for s in subsets) / len(subsets)


@pytest.mark.parametrize(
    "n,c,k", [(5, 0, 1), (5, 5, 3), (5, 1, 1), (10, 3, 5), (8, 2, 8), (6, 4, 2)]
)
def test_pass_at_k_matches_brute_force(n, c, k):
    assert math.isclose(pass_at_k(n, c, k), brute_force_pass_at_k(n, c, k), abs_tol=1e-12)


def test_pass_at_k_rejects_k_larger_than_n():
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 5)


def _result(status, passed, total=2):
    return ExecutionResult(Status(status), passed, total)


def test_aggregate():
    results = {
        "a": [_result("passed", 2), _result("failed", 1)],
        "b": [_result("syntax_error", 0), _result("timeout", 0)],
    }
    summary = aggregate(results, ks=[1, 2])
    assert summary["num_problems"] == 2
    assert summary["pass@1"]["mean"] == pytest.approx(0.25)
    assert summary["pass@2"]["mean"] == pytest.approx(0.5)
    assert summary["test_pass_rate"]["mean"] == pytest.approx((0.75 + 0) / 2)
    assert summary["compile_rate"]["mean"] == pytest.approx(0.75)
    assert summary["execution_rate"]["mean"] == pytest.approx(0.5)
    assert summary["status_distribution"]["passed"] == pytest.approx(0.25)


def test_bootstrap_ci_brackets_mean():
    x = np.random.default_rng(1).random(200)
    est = bootstrap_ci(x)
    assert est.ci_low < est.mean < est.ci_high


def test_paired_comparison_detects_clear_difference():
    a = np.zeros(100)
    b = np.r_[np.ones(40), np.zeros(60)]
    result = paired_comparison(a, b)
    assert result.diff.mean == pytest.approx(0.4)
    assert result.diff.ci_low > 0
    assert result.p_value < 0.001


def test_paired_comparison_identical_runs():
    a = np.r_[np.ones(30), np.zeros(70)]
    assert paired_comparison(a, a).p_value == 1.0


def test_mcnemar_exact():
    a = [True] * 10 + [False] * 10
    b = [True] * 10 + [True] * 8 + [False] * 2
    result = mcnemar_exact(a, b)
    assert result["only_a"] == 0 and result["only_b"] == 8
    assert result["p_value"] == pytest.approx(2 / 2**8)
