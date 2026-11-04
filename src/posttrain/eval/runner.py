"""End-to-end evaluation: generate, extract, execute, score, and compare runs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from posttrain.config import EvalConfig, to_dict
from posttrain.data import build_messages, load_problems
from posttrain.data.problems import Problem
from posttrain.eval.metrics import aggregate, pass_at_k
from posttrain.eval.stats import mcnemar_exact, paired_comparison
from posttrain.execution import ExecutionJob, ExecutionResult, Executor, extract_code, sanitize
from posttrain.runs import create_run_dir, read_jsonl, set_seed, write_json, write_jsonl


def score_completions(
    problems: list[Problem],
    conversations: list[list[dict]],
    completions: list[list[str]],
    executor: Executor,
    clean: bool = True,
    truncated: list[list[bool]] | None = None,
) -> list[dict[str, Any]]:
    jobs, rows = [], []
    truncated = truncated or [[False] * len(s) for s in completions]
    for problem, messages, samples, cut in zip(
        problems, conversations, completions, truncated, strict=True
    ):
        for index, completion in enumerate(samples):
            extracted = extract_code(completion, problem.entry_point)
            code = sanitize(extracted.code) if clean else extracted.code
            jobs.append(
                ExecutionJob(code, list(problem.tests), problem.setup_code, problem.test_prelude)
            )
            rows.append(
                {
                    "task_id": problem.task_id,
                    "sample_index": index,
                    "messages": messages,
                    "completion": completion,
                    "code": code,
                    "fenced": extracted.fenced,
                    "sanitized": code != extracted.code,
                    "truncated": cut[index],
                }
            )
    for row, result in zip(rows, executor.run_many(jobs), strict=True):
        row["result"] = result.to_dict()
    return rows


def summarize(rows: list[dict[str, Any]], ks: list[int], seed: int = 0) -> dict[str, Any]:
    by_problem: dict[str, list[ExecutionResult]] = defaultdict(list)
    for row in rows:
        by_problem[row["task_id"]].append(ExecutionResult.from_dict(row["result"]))
    summary = aggregate(by_problem, ks, seed)
    summary["fenced_rate"] = float(np.mean([row["fenced"] for row in rows]))
    summary["sanitized_rate"] = float(np.mean([row.get("sanitized", False) for row in rows]))
    return summary


def run_eval(cfg: EvalConfig) -> Path:
    set_seed(cfg.seed)
    run_dir = create_run_dir(cfg.output_dir, "eval", cfg.run_name, to_dict(cfg))
    problems = load_problems(
        cfg.data.dataset, cfg.data.split, cfg.data.limit, cfg.data.task_ids_file
    )
    conversations = [build_messages(p, cfg.data.num_prompt_tests) for p in problems]
    print(f"[eval] {len(problems)} problems from {cfg.data.dataset}:{cfg.data.split}")

    from posttrain.eval.generate import make_generator

    generator = make_generator(cfg.model, cfg.generation)
    generation = generator.generate(conversations)
    print(f"[eval] generated {generation.num_generated_tokens} tokens "
          f"in {generation.wall_time_s:.1f}s")

    rows = score_completions(
        problems,
        conversations,
        generation.texts,
        Executor.from_config(cfg.execution),
        clean=cfg.execution.sanitize,
        truncated=generation.truncated,
    )
    write_jsonl(run_dir / "samples.jsonl", rows)
    metrics = {
        "dataset": f"{cfg.data.dataset}:{cfg.data.split}",
        "model": cfg.model.name_or_path,
        "adapter": cfg.model.adapter_path,
        "correctness": summarize(rows, cfg.ks, cfg.seed),
        "generation": generation.stats(),
    }
    write_json(run_dir / "metrics.json", metrics)
    _print_summary(metrics)
    print(f"[eval] wrote {run_dir}")
    return run_dir


def _print_summary(metrics: dict[str, Any]) -> None:
    for name, value in metrics["correctness"].items():
        if isinstance(value, dict) and "mean" in value:
            print(f"  {name:>16}: {value['mean']:.4f}  "
                  f"[{value['ci_low']:.4f}, {value['ci_high']:.4f}]")
    print(f"  {'statuses':>16}: {json.dumps(metrics['correctness']['status_distribution'])}")
    gen = metrics["generation"]
    print(f"  {'throughput':>16}: {gen['generated_tokens_per_s']:.1f} tok/s")
    print(f"  {'truncation_rate':>16}: {gen['truncation_rate']:.4f}")


def report_table(run_dirs: list[Path], metric: str = "pass@1") -> str:
    """Markdown table: one row per model, one column per dataset, `mean [95% CI]` cells."""
    import yaml

    cells: dict[str, dict[str, str]] = defaultdict(dict)
    datasets: list[str] = []
    for run_dir in sorted(run_dirs):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        model = cfg["model"]["adapter_path"] or cfg["model"]["name_or_path"]
        dataset = metrics["dataset"]
        if dataset not in datasets:
            datasets.append(dataset)
        value = metrics["correctness"].get(metric)
        if value:
            cells[model][dataset] = (
                f"{100 * value['mean']:.1f} [{100 * value['ci_low']:.1f}, "
                f"{100 * value['ci_high']:.1f}]"
            )
    lines = [
        f"| model | {' | '.join(datasets)} |",
        f"|---|{'---|' * len(datasets)}",
    ]
    for model, row in cells.items():
        lines.append(f"| {model} | " + " | ".join(row.get(d, "") for d in datasets) + " |")
    return "\n".join(lines)


def compare_runs(run_a: Path, run_b: Path, k: int = 1) -> dict[str, Any]:
    """Paired comparison of two eval runs on their shared problems (b relative to a)."""
    scores_a, binary_a = _per_problem(Path(run_a), k)
    scores_b, binary_b = _per_problem(Path(run_b), k)
    shared = sorted(set(scores_a) & set(scores_b))
    if not shared:
        raise ValueError("The runs share no problems")
    result: dict[str, Any] = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "metric": f"pass@{k}",
        "only_in_a": len(set(scores_a) - set(shared)),
        "only_in_b": len(set(scores_b) - set(shared)),
        "paired": paired_comparison(
            [scores_a[t] for t in shared], [scores_b[t] for t in shared]
        ).to_dict(),
    }
    if all(t in binary_a and t in binary_b for t in shared):
        result["mcnemar"] = mcnemar_exact(
            [binary_a[t] for t in shared], [binary_b[t] for t in shared]
        )
    return result


def _per_problem(run_dir: Path, k: int) -> tuple[dict[str, float], dict[str, bool]]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in read_jsonl(run_dir / "samples.jsonl"):
        groups[row["task_id"]].append(row["result"]["status"] == "passed")
    scores = {t: pass_at_k(len(v), sum(v), k) for t, v in groups.items()}
    binary = {t: v[0] for t, v in groups.items() if len(v) == 1}
    return scores, binary
