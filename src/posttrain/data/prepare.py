"""Build the local KodCode training/held-out splits: verify, decontaminate, split."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from posttrain.data.contamination import find_overlaps, problem_text
from posttrain.data.problems import (
    _load_hf,
    kodcode_row_to_problem,
    load_problems,
    parse_dataset_spec,
    processed_dir,
    save_problems,
)
from posttrain.execution import ExecutionJob, Executor
from posttrain.runs import write_json

DEFAULT_EVAL_SETS = ("mbpp:test", "mbppplus:test", "humaneval:test", "humanevalplus:test")


def prepare_kodcode(
    executor: Executor,
    heldout_size: int = 500,
    validation_size: int = 200,
    seed: int = 0,
    eval_sets: tuple[str, ...] = DEFAULT_EVAL_SETS,
    ngram: int = 10,
    threshold: float = 0.5,
) -> dict[str, Any]:
    rows = _load_hf("KodCode/KodCode-Light-RL-10K", None, ["train"])
    converted = [kodcode_row_to_problem(r) for r in rows]
    supported = [p for p in converted if p is not None]

    results = executor.run_many(
        [ExecutionJob(p.reference_solution, list(p.tests), p.setup_code, p.test_prelude)
         for p in supported]
    )
    verified = [p for p, r in zip(supported, results, strict=True) if r.passed]
    failed_reference = Counter(r.status.value for r in results if not r.passed)

    eval_texts = {
        p.task_id: problem_text(p)
        for spec in eval_sets
        for p in load_problems(*parse_dataset_spec(spec))
    }
    train_texts = {p.task_id: problem_text(p) for p in verified}
    # Check both directions: a long KodCode problem can embed a short benchmark problem.
    contaminated = {
        m.train_id
        for m in find_overlaps(train_texts, eval_texts, ngram)
        if m.train_id and m.overlap >= threshold
    }
    contaminated |= {
        m.eval_id
        for m in find_overlaps(eval_texts, train_texts, ngram)
        if m.overlap >= threshold
    }
    clean = [p for p in verified if p.task_id not in contaminated]

    rng = random.Random(seed)
    shuffled = sorted(clean, key=lambda p: p.task_id)
    rng.shuffle(shuffled)
    # heldout: final test set for reporting. validation: loss curves and model selection.
    heldout = shuffled[:heldout_size]
    validation = shuffled[heldout_size : heldout_size + validation_size]
    train = shuffled[heldout_size + validation_size :]

    out = processed_dir() / "kodcode"
    save_problems(train, out / "train.jsonl")
    save_problems(validation, out / "validation.jsonl")
    save_problems(heldout, out / "heldout.jsonl")
    report = {
        "source": "KodCode/KodCode-Light-RL-10K",
        "rows": len(rows),
        "unsupported_tests": len(rows) - len(supported),
        "reference_failed": dict(failed_reference),
        "verified": len(verified),
        "contaminated": sorted(contaminated),
        "decontamination": {"eval_sets": list(eval_sets), "ngram": ngram, "threshold": threshold},
        "train": len(train),
        "validation": len(validation),
        "heldout": len(heldout),
        "difficulty": {
            name: dict(Counter(p.difficulty for p in split))
            for name, split in [("train", train), ("validation", validation), ("heldout", heldout)]
        },
        "seed": seed,
    }
    write_json(out / "report.json", report)
    return report
