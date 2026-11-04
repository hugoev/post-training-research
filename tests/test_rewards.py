import pytest

from posttrain.execution import Executor
from posttrain.rewards import ExecutionReward, format_reward

TESTS = ["assert sq(2) == 4", "assert sq(3) == 9", "assert sq(-1) == 1"]


def _completion(code):
    return [{"role": "assistant", "content": f"Sure!\n```python\n{code}\n```"}]


@pytest.fixture(scope="module")
def executor():
    return Executor(timeout_s=10, per_test_timeout_s=1.0, memory_mb=512)


def _call(reward, completions):
    n = len(completions)
    return reward(
        prompts=[None] * n,
        completions=completions,
        tests=[TESTS] * n,
        setup_code=[""] * n,
        entry_point=["sq"] * n,
    )


def test_binary_and_fraction_rewards(executor):
    completions = [
        _completion("def sq(x): return x * x"),
        _completion("def sq(x): return 4 if x == 2 else 0"),  # hardcodes the visible test
        _completion("def sq(x) return"),
    ]
    assert _call(ExecutionReward(executor, "binary"), completions) == [1.0, 0.0, 0.0]
    fraction = _call(ExecutionReward(executor, "fraction"), completions)
    assert fraction == pytest.approx([1.0, 1 / 3, 0.0])


def test_wrong_self_written_assert_does_not_mask_correct_solution(executor):
    completion = [_completion("def sq(x): return x * x\n\nassert sq(2) == 5")]
    assert _call(ExecutionReward(executor, "binary"), completion) == [1.0]
    assert _call(ExecutionReward(executor, "binary", clean=False), completion) == [0.0]


def test_reward_tracks_statuses(executor):
    reward = ExecutionReward(executor, "binary")
    _call(reward, [_completion("def sq(x): return x * x"), _completion("import sys; sys.exit(0)")])
    assert reward.status_counts["passed"] == 1
    assert reward.__name__ == "execution_binary"


def test_format_reward():
    completions = [
        _completion("def sq(x): return x * x"),
        [{"role": "assistant", "content": "def sq(x): return x * x"}],
        _completion("def sq(x) return"),
        "```python\nx = 1\n```",
    ]
    assert format_reward(None, completions) == [1.0, 0.0, 0.0, 1.0]
