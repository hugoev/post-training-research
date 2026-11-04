"""A common schema for executable coding problems, plus loaders for public benchmarks."""

from __future__ import annotations

import ast
import builtins
import dataclasses
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Problem:
    task_id: str
    source: str
    description: str
    # Snippets executed one by one after the solution; each is one scored test.
    tests: tuple[str, ...]
    entry_point: str | None = None
    # Runs before the solution in the same namespace (e.g. imports from a starter prompt).
    setup_code: str = ""
    # Runs after the solution, before the tests (e.g. a pytest-style test module).
    test_prelude: str = ""
    # Asserts that may be shown in the prompt (a subset, controlled by num_prompt_tests).
    examples: tuple[str, ...] = ()
    # Signature + docstring the model must complete (HumanEval style).
    starter_code: str = ""
    # Function declarations the solution must provide, shown in the prompt.
    interface: str = ""
    reference_solution: str | None = None
    difficulty: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Problem:
        return cls(**{**d, "tests": tuple(d["tests"]), "examples": tuple(d.get("examples", ()))})


def mbpp_row_to_problem(row: dict) -> Problem:
    raw_tests = row["test_list"]
    # MBPP's test setup builds fixtures from classes the solution defines, so it must run
    # after the solution: attach it to each test rather than to setup_code.
    fixture = (row.get("test_setup_code") or "").strip()
    tests = tuple(f"{fixture}\n{t}" if fixture else t for t in raw_tests)
    return Problem(
        task_id=f"mbpp/{row['task_id']}",
        source="mbpp",
        description=row["text"],
        tests=tests,
        examples=tests,
        entry_point=_called_function(raw_tests[0]) if raw_tests else None,
        reference_solution=row["code"],
    )


def mbppplus_row_to_problem(row: dict) -> Problem:
    examples = tuple(row["test_list"])
    return Problem(
        task_id=f"mbppplus/{row['task_id']}",
        source="mbppplus",
        description=row["prompt"],
        # EvalPlus ships one script looping over hundreds of inputs: scored as a single test.
        tests=(row["test"],),
        examples=examples,
        setup_code="\n".join(row.get("test_imports") or []),
        entry_point=_called_function(examples[0]) if examples else None,
        reference_solution=row["code"],
    )


def humaneval_row_to_problem(row: dict, source: str = "humaneval") -> Problem:
    entry_point = row["entry_point"]
    return Problem(
        task_id=row["task_id"] if source == "humaneval" else f"{source}/{row['task_id']}",
        source=source,
        description=row["prompt"],
        # HumanEval ships one `check(candidate)` function, so it is scored as a single test.
        tests=(f"{row['test']}\ncheck({entry_point})",),
        entry_point=entry_point,
        setup_code=_import_lines(row["prompt"]),
        starter_code=row["prompt"],
        reference_solution=row["prompt"] + row["canonical_solution"],
    )


def kodcode_row_to_problem(row: dict) -> Problem | None:
    """Convert a KodCode row; None when its pytest file needs fixtures, marks or classes."""
    tests = _plain_pytest_functions(row["test"])
    if not tests:
        return None
    info = row["test_info"] or []
    if isinstance(info, str):
        try:
            info = ast.literal_eval(info)
        except (ValueError, SyntaxError):
            return None
    declarations = [
        i["function_declaration"].strip() for i in info if i.get("function_declaration")
    ]
    names = [i["function_name"] for i in info if i.get("function_name")]
    if not declarations:
        return None
    return Problem(
        task_id=f"kodcode/{row['question_id']}",
        source="kodcode",
        description=row["question"],
        tests=tuple(f"{name}()" for name in tests),
        test_prelude=row["test"],
        interface="\n".join(declarations),
        entry_point=names[0] if names else None,
        reference_solution=row["solution"],
        difficulty=row.get("gpt_difficulty"),
    )


def _plain_pytest_functions(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    if any(isinstance(n, ast.ClassDef) and n.name.startswith("Test") for n in tree.body):
        return []
    names = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            args = node.args
            if node.decorator_list or args.args or args.vararg or args.kwonlyargs or args.kwarg:
                return []
            names.append(node.name)
    return names


_BUILTIN_NAMES = frozenset(dir(builtins))


def _called_function(test: str) -> str | None:
    """The first non-builtin function called in an assert, e.g. `foo` in `assert foo(1) == 2`."""
    try:
        tree = ast.parse(test)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in _BUILTIN_NAMES:
                return node.func.id
    return None


def _import_lines(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    imports = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
    return "\n".join(ast.unparse(n) for n in imports)


def _load_hf(path: str, name: str | None, splits: Iterable[str]) -> list[dict]:
    from datasets import load_dataset

    rows: list[dict] = []
    for split in splits:
        rows.extend(load_dataset(path, name, split=split))
    return rows


def processed_dir() -> Path:
    return Path(os.environ.get("POSTTRAIN_DATA_DIR", "data/processed"))


def save_problems(problems: list[Problem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in problems:
            f.write(json.dumps(p.to_dict()) + "\n")


def _load_processed(name: str, splits: list[str]) -> list[Problem]:
    problems = []
    for split in splits:
        path = processed_dir() / name / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `posttrain prepare-data {name}` first")
        with path.open() as f:
            problems.extend(Problem.from_dict(json.loads(line)) for line in f if line.strip())
    return problems


LOADERS: dict[str, Callable[[list[str]], list[Problem]]] = {
    "mbpp": lambda s: [
        mbpp_row_to_problem(r) for r in _load_hf("google-research-datasets/mbpp", "full", s)
    ],
    "mbppplus": lambda s: [
        mbppplus_row_to_problem(r) for r in _load_hf("evalplus/mbppplus", None, s)
    ],
    "humaneval": lambda s: [
        humaneval_row_to_problem(r) for r in _load_hf("openai/openai_humaneval", None, s)
    ],
    "humanevalplus": lambda s: [
        humaneval_row_to_problem(r, "humanevalplus")
        for r in _load_hf("evalplus/humanevalplus", None, s)
    ],
    # Built locally by `posttrain prepare-data kodcode` (verified, decontaminated, split).
    "kodcode": lambda s: _load_processed("kodcode", s),
}


# Problems whose own reference solution cannot pass their tests.
KNOWN_BROKEN = {
    "humanevalplus/HumanEval/32": "test runs `_poly(*candidate(*inp), inp)`, unpacking a float",
}


def load_problems(
    dataset: str, split: str, limit: int | None = None, task_ids_file: str | None = None
) -> list[Problem]:
    if dataset not in LOADERS:
        raise ValueError(f"Unknown dataset {dataset!r}; available: {sorted(LOADERS)}")
    problems = [p for p in LOADERS[dataset](split.split("+")) if p.task_id not in KNOWN_BROKEN]
    if task_ids_file:
        keep = set(Path(task_ids_file).read_text().split())
        problems = [p for p in problems if p.task_id in keep]
    return problems[:limit] if limit else problems


def parse_dataset_spec(spec: str) -> tuple[str, str]:
    """Parse "mbpp:test" into ("mbpp", "test")."""
    dataset, sep, split = spec.partition(":")
    if not sep:
        raise ValueError(f"Expected dataset:split, got {spec!r}")
    return dataset, split
