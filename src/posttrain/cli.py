"""Command-line entry point: `posttrain <command> ...`.

Config-driven commands take a YAML path followed by dotted overrides, e.g.
    posttrain eval configs/eval/mbpp_test.yaml generation.num_samples=10 ks=[1,5,10]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _eval(args):
    from posttrain.config import EvalConfig, load_config
    from posttrain.eval.runner import run_eval

    run_eval(load_config(EvalConfig, args.config, args.overrides))


def _train(stage):
    def run(args):
        from posttrain.config import TrainConfig, load_config

        cfg = load_config(TrainConfig, args.config, args.overrides)
        if stage == "sft":
            from posttrain.train.sft import run_sft as fn
        elif stage == "dpo":
            from posttrain.train.dpo import run_dpo as fn
        else:
            from posttrain.train.grpo import run_grpo as fn
        fn(cfg)

    return run


def _compare(args):
    from posttrain.eval.runner import compare_runs

    result = compare_runs(Path(args.run_a), Path(args.run_b), args.k)
    print(json.dumps(result, indent=2))


def _report(args):
    from posttrain.eval.runner import report_table

    print(report_table([Path(p) for p in args.runs], args.metric))


def _merge(args):
    from posttrain.train.merge import merge_adapter

    merge_adapter(args.base, args.adapter, args.output, args.dtype)


def _build_prefs(args):
    from posttrain.data.builders import preference_pairs
    from posttrain.runs import read_jsonl, write_jsonl

    samples = read_jsonl(Path(args.samples))
    pairs = preference_pairs(samples, args.max_pairs_per_problem, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, pairs)
    tasks = {s["task_id"] for s in samples}
    print(f"[prefs] {len(pairs)} pairs covering {len({p['task_id'] for p in pairs})}"
          f"/{len(tasks)} problems -> {out}")


def _prepare_data(args):
    from posttrain.data.prepare import prepare_kodcode
    from posttrain.execution import Executor

    executor = Executor(timeout_s=30, per_test_timeout_s=10, memory_mb=2048)
    report = prepare_kodcode(
        executor, args.heldout_size, args.validation_size, args.seed, tuple(args.eval_sets)
    )
    summary = {k: v for k, v in report.items() if k != "contaminated"}
    summary["contaminated"] = len(report["contaminated"])
    print(json.dumps(summary, indent=2))


def _filter_difficulty(args):
    import numpy as np

    from posttrain.data.builders import learnable_tasks, pass_rates
    from posttrain.runs import read_jsonl

    samples = read_jsonl(Path(args.samples))
    rates = pass_rates(samples)
    keep = learnable_tasks(samples, args.min_rate, args.max_rate)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(keep) + "\n")
    hist, edges = np.histogram(list(rates.values()), bins=[0, 1e-9, 0.25, 0.5, 0.75, 1 - 1e-9, 1])
    labels = ["0", "(0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1)", "1"]
    print("pass-rate histogram:", dict(zip(labels, hist.tolist(), strict=True)))
    print(f"[filter] kept {len(keep)}/{len(rates)} tasks with "
          f"{args.min_rate} < pass rate < {args.max_rate} -> {out}")


def _contamination(args):
    from posttrain.data import load_problems, parse_dataset_spec
    from posttrain.data.contamination import find_overlaps, problem_text

    def texts(specs):
        return {
            p.task_id: problem_text(p)
            for spec in specs
            for p in load_problems(*parse_dataset_spec(spec))
        }

    train = texts(args.train)
    evaluation = texts(args.eval)
    matches = find_overlaps(train, evaluation, args.ngram)
    flagged = sorted((m for m in matches if m.overlap >= args.threshold),
                     key=lambda m: -m.overlap)
    report = {
        "train": args.train,
        "eval": args.eval,
        "ngram": args.ngram,
        "threshold": args.threshold,
        "num_eval": len(matches),
        "num_flagged": len(flagged),
        "flagged": [m.__dict__ for m in flagged],
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "flagged"}, indent=2))
    for m in flagged[:20]:
        print(f"  {m.eval_id} ~ {m.train_id}: {m.overlap:.2f}")


def _default_eval_sets():
    from posttrain.data.prepare import DEFAULT_EVAL_SETS

    return DEFAULT_EVAL_SETS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="posttrain", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn, help_text in [
        ("eval", _eval, "generate, execute and score a model"),
        ("sft", _train("sft"), "supervised fine-tuning"),
        ("dpo", _train("dpo"), "direct preference optimization"),
        ("grpo", _train("grpo"), "GRPO with execution rewards"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config")
        p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. trainer.max_steps=10")
        p.set_defaults(fn=fn)

    p = sub.add_parser("compare", help="paired statistical comparison of two eval runs")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--k", type=int, default=1)
    p.set_defaults(fn=_compare)

    p = sub.add_parser("report", help="markdown table of eval runs (model x dataset)")
    p.add_argument("runs", nargs="+", help="eval run directories")
    p.add_argument("--metric", default="pass@1")
    p.set_defaults(fn=_report)

    p = sub.add_parser("merge", help="merge a LoRA adapter into base weights")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dtype", default="bfloat16")
    p.set_defaults(fn=_merge)

    p = sub.add_parser("build-prefs", help="build DPO pairs from a sampled eval run")
    p.add_argument("--samples", required=True, help="samples.jsonl from `posttrain eval`")
    p.add_argument("--output", required=True)
    p.add_argument("--max-pairs-per-problem", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=_build_prefs)

    p = sub.add_parser("prepare-data", help="verify, decontaminate and split KodCode")
    p.add_argument("dataset", choices=["kodcode"])
    p.add_argument("--heldout-size", type=int, default=500)
    p.add_argument("--validation-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-sets", nargs="+", default=list(_default_eval_sets()))
    p.set_defaults(fn=_prepare_data)

    p = sub.add_parser("filter-difficulty",
                       help="keep tasks the policy sometimes but not always solves (for GRPO)")
    p.add_argument("--samples", required=True, help="samples.jsonl from a multi-sample eval")
    p.add_argument("--output", required=True)
    p.add_argument("--min-rate", type=float, default=0.0)
    p.add_argument("--max-rate", type=float, default=1.0)
    p.set_defaults(fn=_filter_difficulty)

    p = sub.add_parser("contamination", help="n-gram overlap between train and eval problems")
    p.add_argument("--train", nargs="+", required=True, help="dataset:split, e.g. mbpp:train")
    p.add_argument("--eval", nargs="+", required=True)
    p.add_argument("--ngram", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output")
    p.set_defaults(fn=_contamination)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
