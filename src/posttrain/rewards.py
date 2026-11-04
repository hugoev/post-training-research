"""Verifiable rewards for GRPO, in the signature TRL's GRPOTrainer expects.

TRL calls each reward function as `f(prompts=..., completions=..., **dataset_columns)`
and logs it under `f.__name__`.
"""

from __future__ import annotations

import ast
from collections import Counter

from posttrain.execution import ExecutionJob, Executor, extract_code, sanitize


def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"]


class ExecutionReward:
    """Rewards correctness on the problem's full (mostly hidden) test suite.

    mode="binary": 1.0 only when every test passes.
    mode="fraction": share of tests passed. Denser, but a policy can earn partial
    credit by special-casing the test visible in the prompt.
    """

    def __init__(self, executor: Executor, mode: str = "binary", clean: bool = True):
        if mode not in ("binary", "fraction"):
            raise ValueError(f"Unknown correctness mode {mode!r}")
        self.executor = executor
        self.mode = mode
        self.clean = clean
        self.__name__ = f"execution_{mode}"
        self.status_counts: Counter[str] = Counter()

    def _program(self, completion, entry_point: str) -> str:
        code = extract_code(completion_text(completion), entry_point or None).code
        return sanitize(code) if self.clean else code

    def __call__(
        self, prompts, completions, tests, setup_code, entry_point, test_prelude=None, **kwargs
    ):
        preludes = test_prelude if test_prelude is not None else [""] * len(completions)
        jobs = [
            ExecutionJob(self._program(c, ep), list(t), setup, prelude)
            for c, t, setup, ep, prelude in zip(
                completions, tests, setup_code, entry_point, preludes, strict=True
            )
        ]
        results = self.executor.run_many(jobs)
        self.status_counts.update(r.status.value for r in results)
        if self.mode == "binary":
            return [float(r.passed) for r in results]
        return [r.pass_rate for r in results]


def format_reward(prompts, completions, **kwargs) -> list[float]:
    """1.0 for a fenced code block that parses as Python, else 0.0."""
    rewards = []
    for c in completions:
        extracted = extract_code(completion_text(c))
        try:
            ast.parse(extracted.code)
            ok = extracted.fenced and bool(extracted.code.strip())
        except (SyntaxError, ValueError):
            ok = False
        rewards.append(float(ok))
    return rewards
