from posttrain.config import EvalConfig, TrainConfig, load_config
from posttrain.data.builders import preference_pairs, sft_records
from posttrain.data.contamination import find_overlaps
from posttrain.data.problems import (
    humaneval_row_to_problem,
    kodcode_row_to_problem,
    mbpp_row_to_problem,
    mbppplus_row_to_problem,
)
from posttrain.data.prompts import build_messages
from posttrain.execution import extract_code

MBPP_ROW = {
    "task_id": 7,
    "text": "Write a function to find the shared elements from the given two lists.",
    "code": "def similar_elements(a, b):\n    return tuple(set(a) & set(b))",
    "test_list": [
        "assert set(similar_elements((3, 4, 5, 6), (5, 7, 4, 10))) == set((4, 5))",
        "assert set(similar_elements((1, 2, 3, 4), (5, 4, 3, 7))) == set((3, 4))",
        "assert set(similar_elements((11, 12, 14, 13), (17, 15, 14, 13))) == set((13, 14))",
    ],
    "test_setup_code": "",
}

HUMANEVAL_ROW = {
    "task_id": "HumanEval/0",
    "prompt": "from typing import List\n\n\ndef has_close(xs: List[float], t: float) -> bool:\n"
    '    """Any two numbers closer than t."""\n',
    "canonical_solution": "    return any(abs(a - b) < t for i, a in enumerate(xs) "
    "for b in xs[i + 1:])\n",
    "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9], 0.3) is False\n"
    "    assert candidate([1.0, 2.8, 3.0], 0.3) is True\n",
    "entry_point": "has_close",
}


def test_mbpp_conversion():
    p = mbpp_row_to_problem(MBPP_ROW)
    assert p.task_id == "mbpp/7"
    assert p.entry_point == "similar_elements"
    assert len(p.tests) == 3


def test_mbpp_fixture_runs_after_solution():
    from posttrain.execution import Executor

    row = {
        "task_id": 1,
        "text": "Return the value stored in a node.",
        "code": "class Node:\n    def __init__(self, v):\n        self.v = v\n\n"
        "def value(n):\n    return n.v",
        "test_list": ["assert value(root) == 3"],
        "test_setup_code": "root = Node(3)",
    }
    p = mbpp_row_to_problem(row)
    assert p.entry_point == "value"
    assert Executor(timeout_s=10).run(p.reference_solution, list(p.tests), p.setup_code).passed


KODCODE_ROW = {
    "question_id": "Leetcode_1_I",
    "question": "Return the number of distinct elements in a list.",
    "solution": "def count_distinct(nums):\n    return len(set(nums))",
    "test": (
        "import pytest\nfrom solution import count_distinct\n\n"
        "def test_empty():\n    assert count_distinct([]) == 0\n\n"
        "def test_dupes():\n    assert count_distinct([1, 1, 2]) == 2\n\n"
        "def test_type_error():\n    with pytest.raises(TypeError):\n"
        "        count_distinct(None)\n"
    ),
    "test_info": "[{'function_declaration': 'def count_distinct(nums):', "
    "'function_name': 'count_distinct'}]",
    "gpt_difficulty": "easy",
}


def test_kodcode_conversion_runs_pytest_style_tests():
    from posttrain.execution import Executor, Status

    p = kodcode_row_to_problem(KODCODE_ROW)
    assert p.tests == ("test_empty()", "test_dupes()", "test_type_error()")
    prompt = build_messages(p)[1]["content"]
    assert "def count_distinct(nums):" in prompt and "test_dupes" not in prompt
    executor = Executor(timeout_s=10)
    run = lambda code: executor.run(code, list(p.tests), p.setup_code, p.test_prelude)  # noqa: E731
    assert run(p.reference_solution).passed
    partial = run("def count_distinct(nums):\n    return len(nums or [])")
    assert partial.test_results == [True, False, False]
    missing = run("def other(nums):\n    return 0")
    assert missing.status is Status.RUNTIME_ERROR and "ImportError" in missing.error


def test_kodcode_rejects_fixture_based_tests():
    row = {**KODCODE_ROW, "test": "def test_out(capsys):\n    assert True\n"}
    assert kodcode_row_to_problem(row) is None


def test_mbppplus_shows_original_asserts_but_scores_full_script():
    row = {
        "task_id": 2,
        "prompt": "Add two numbers.",
        "code": "def add(a, b):\n    return a + b",
        "test_list": ["assert add(1, 2) == 3"],
        "test_imports": [],
        "test": "for a, b in [(1, 2), (5, 5)]:\n    assert add(a, b) == a + b\n",
    }
    p = mbppplus_row_to_problem(row)
    prompt = build_messages(p, num_prompt_tests=1)[1]["content"]
    assert "assert add(1, 2) == 3" in prompt and "for a, b" not in prompt
    assert p.entry_point == "add" and len(p.tests) == 1


def test_learnable_tasks_drops_always_and_never_solved():
    from posttrain.data.builders import learnable_tasks

    samples = [
        _sample("always", 0, "passed", 3, "a"), _sample("always", 1, "passed", 3, "b"),
        _sample("never", 0, "failed", 0, "c"), _sample("never", 1, "timeout", 0, "d"),
        _sample("mixed", 0, "passed", 3, "e"), _sample("mixed", 1, "failed", 1, "f"),
    ]
    assert learnable_tasks(samples) == ["mixed"]


def test_stop_tokens_include_end_of_turn():
    from posttrain.eval.generate import stop_token_ids

    class Tok:
        eos_token_id = 1

        def get_vocab(self):
            return {"<|endoftext|>": 1, "<|im_end|>": 2, "a": 3}

    class Gen:
        eos_token_id = [1, 4]

    assert stop_token_ids(Tok(), Gen()) == [1, 2, 4]


def test_prompt_hides_all_but_requested_tests():
    p = mbpp_row_to_problem(MBPP_ROW)
    prompt = build_messages(p, num_prompt_tests=1)[1]["content"]
    assert p.tests[0] in prompt
    assert p.tests[1] not in prompt and p.tests[2] not in prompt


def test_humaneval_conversion_and_reference_passes():
    from posttrain.execution import Executor

    p = humaneval_row_to_problem(HUMANEVAL_ROW)
    assert p.setup_code == "from typing import List"
    assert p.starter_code in build_messages(p)[1]["content"]
    result = Executor(timeout_s=10).run(p.reference_solution, list(p.tests), p.setup_code)
    assert result.passed


def test_sft_records_use_reference_solution():
    records = sft_records([mbpp_row_to_problem(MBPP_ROW)], num_prompt_tests=1)
    assert records[0]["completion"][0]["role"] == "assistant"
    assert "def similar_elements" in records[0]["completion"][0]["content"]


def _sample(task, idx, status, passed, code):
    return {
        "task_id": task,
        "sample_index": idx,
        "messages": [{"role": "user", "content": task}],
        "completion": f"```python\n{code}\n```",
        "code": code,
        "result": {"status": status, "tests_passed": passed, "tests_total": 3},
    }


def test_preference_pairs():
    samples = [
        _sample("t1", 0, "passed", 3, "good"),
        _sample("t1", 1, "failed", 2, "close"),
        _sample("t1", 2, "syntax_error", 0, "broken"),
        _sample("t1", 3, "passed", 3, "good"),  # duplicate of sample 0
        _sample("t2", 0, "passed", 3, "all pass"),  # no failing sample -> no pair
    ]
    pairs = preference_pairs(samples)
    assert len(pairs) == 1
    assert "good" in pairs[0]["chosen"][0]["content"]
    assert "broken" in pairs[0]["rejected"][0]["content"]


def test_contamination_flags_near_duplicates():
    train = {"a": "write a function that returns the sum of squares of a list of integers"}
    evaluation = {
        "dup": "Write a function that returns the sum of squares of a list of integers!",
        "new": "parse an ISO date string into a datetime object and return the weekday",
    }
    by_id = {m.eval_id: m for m in find_overlaps(train, evaluation, n=5)}
    assert by_id["dup"].overlap == 1.0 and by_id["dup"].train_id == "a"
    assert by_id["new"].overlap == 0.0


def test_extract_prefers_block_defining_entry_point():
    text = (
        "Here:\n```python\ndef helper():\n    return 1\n\n\n\n```\n"
        "```python\ndef solve(x):\n    return x\n```\nUsage:\n```python\nprint(solve(1))\n```"
    )
    assert extract_code(text, "solve").code.startswith("def solve")


def test_extract_handles_truncated_and_unfenced_output():
    truncated = extract_code("```python\ndef f():\n    return 1")
    assert truncated.code.startswith("def f") and not truncated.fenced
    raw = extract_code("def f():\n    return 1")
    assert raw.code.startswith("def f") and not raw.fenced


def test_sanitize_drops_self_tests_but_keeps_definitions():
    from posttrain.execution import sanitize

    code = (
        "import functools\n\n"
        "LIMIT = 10\n\n"
        "@functools.cache\n"
        "def fib(n):\n"
        "    # comment kept\n"
        "    return n if n < 2 else fib(n - 1) + fib(n - 2)\n\n"
        "assert fib(10) == 56\n"
        "print(fib(5))\n"
        "if __name__ == '__main__':\n"
        "    print('demo')\n"
    )
    cleaned = sanitize(code)
    assert "@functools.cache" in cleaned and "# comment kept" in cleaned
    assert "LIMIT = 10" in cleaned and "import functools" in cleaned
    assert "assert" not in cleaned and "print" not in cleaned
    assert sanitize("def f(:\n  pass") == "def f(:\n  pass"


def test_config_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "seed: 1\ngeneration:\n  num_samples: 1\n  temperature: 0.0\n"
    )
    (tmp_path / "child.yaml").write_text(
        "inherit: base.yaml\nrun_name: child\ngeneration:\n  num_samples: 8\n"
    )
    cfg = load_config(EvalConfig, tmp_path / "child.yaml", ["generation.temperature=1.0"])
    assert (cfg.run_name, cfg.seed) == ("child", 1)
    assert cfg.generation.num_samples == 8 and cfg.generation.temperature == 1.0


def test_report_table(tmp_path):
    import json

    from posttrain.eval.runner import report_table

    runs = [("m1", "a:test", 0.5), ("m1", "b:test", 0.25), ("m2", "a:test", 1.0)]
    for model, dataset, mean in runs:
        run = tmp_path / f"{model}-{dataset.replace(':', '_')}"
        run.mkdir()
        (run / "config.yaml").write_text(f"model:\n  name_or_path: {model}\n  adapter_path: null\n")
        (run / "metrics.json").write_text(json.dumps({
            "dataset": dataset,
            "correctness": {"pass@1": {"mean": mean, "ci_low": mean, "ci_high": mean}},
        }))
    table = report_table(sorted(tmp_path.iterdir()))
    assert "| model | a:test | b:test |" in table
    assert "| m1 | 50.0 [50.0, 50.0] | 25.0 [25.0, 25.0] |" in table
    assert "| m2 | 100.0 [100.0, 100.0] |  |" in table


def test_config_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("run_name: x\ngeneration:\n  num_samples: 1\n")
    cfg = load_config(EvalConfig, path, ["generation.num_samples=10", "ks=[1,5]"])
    assert cfg.generation.num_samples == 10 and cfg.ks == [1, 5]
    train = load_config(
        TrainConfig,
        None,
        [
            "trainer.learning_rate=1e-5",
            "lora.r=8",
            "trainer.save_strategy=no",
            "lora.enabled=false",
        ],
    )
    assert train.trainer["learning_rate"] == 1e-5 and train.lora.r == 8
    assert train.trainer["save_strategy"] == "no" and train.lora.enabled is False
