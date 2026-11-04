"""Convert problems and scored samples into SFT, DPO and GRPO datasets."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from posttrain.data.problems import Problem
from posttrain.data.prompts import build_messages, format_solution


def sft_records(problems: list[Problem], num_prompt_tests: int) -> list[dict[str, Any]]:
    """Conversational prompt/completion pairs; TRL computes loss on the completion only."""
    records = []
    for p in problems:
        if not p.reference_solution:
            continue
        records.append(
            {
                "prompt": build_messages(p, num_prompt_tests),
                "completion": [
                    {"role": "assistant", "content": format_solution(p.reference_solution)}
                ],
            }
        )
    return records


def grpo_records(problems: list[Problem], num_prompt_tests: int) -> list[dict[str, Any]]:
    """Prompts plus the hidden test suite each reward call executes against."""
    return [
        {
            "prompt": build_messages(p, num_prompt_tests),
            "task_id": p.task_id,
            "tests": list(p.tests),
            "setup_code": p.setup_code,
            "test_prelude": p.test_prelude,
            "entry_point": p.entry_point or "",
        }
        for p in problems
    ]


def preference_pairs(
    samples: list[dict[str, Any]], max_pairs_per_problem: int = 4, seed: int = 0
) -> list[dict[str, Any]]:
    """Pair execution-verified passing samples (chosen) with failing ones (rejected).

    `samples` are rows of an eval run's samples.jsonl. Problems where every sample
    passes or every sample fails yield no pairs.
    """
    rng = random.Random(seed)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_task[s["task_id"]].append(s)

    pairs = []
    for task_id, group in by_task.items():
        good = _dedupe([s for s in group if s["result"]["status"] == "passed"])
        bad = _dedupe([s for s in group if s["result"]["status"] != "passed"])
        if not good or not bad:
            continue
        rng.shuffle(good)
        # Prefer the clearest failures (lowest test pass rate) as rejected responses.
        bad.sort(key=lambda s: (s["result"]["tests_passed"] / s["result"]["tests_total"]))
        for chosen, rejected in list(zip(good, bad, strict=False))[:max_pairs_per_problem]:
            pairs.append(
                {
                    "task_id": task_id,
                    "prompt": chosen["messages"],
                    "chosen": [{"role": "assistant", "content": chosen["completion"]}],
                    "rejected": [{"role": "assistant", "content": rejected["completion"]}],
                }
            )
    return pairs


def pass_rates(samples: list[dict[str, Any]]) -> dict[str, float]:
    by_task: dict[str, list[bool]] = defaultdict(list)
    for s in samples:
        by_task[s["task_id"]].append(s["result"]["status"] == "passed")
    return {task: sum(v) / len(v) for task, v in by_task.items()}


def learnable_tasks(
    samples: list[dict[str, Any]], min_rate: float = 0.0, max_rate: float = 1.0
) -> list[str]:
    """Tasks whose sampled pass rate is strictly between min_rate and max_rate.

    In GRPO, a group where every rollout gets the same reward has zero advantage and
    contributes no gradient, so always-solved and never-solved prompts are wasted compute.
    """
    return sorted(t for t, r in pass_rates(samples).items() if min_rate < r < max_rate)


def _dedupe(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen, unique = set(), []
    for s in samples:
        key = s["code"].strip()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique
