"""Prompt construction shared by evaluation and every training stage."""

from __future__ import annotations

from posttrain.data.problems import Problem

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Reply with a single ```python code block "
    "containing a complete, self-contained solution."
)


def user_prompt(problem: Problem, num_prompt_tests: int = 1) -> str:
    if problem.starter_code:
        return (
            "Complete the following Python function.\n\n"
            f"```python\n{problem.starter_code.rstrip()}\n```\n\n"
            "Return the full function definition, including the signature."
        )
    parts = [problem.description.strip()]
    if problem.interface:
        parts.append(
            "Your solution must define:\n```python\n" + problem.interface.strip() + "\n```"
        )
    shown = problem.examples[:num_prompt_tests]
    if shown:
        parts.append(
            "Your code should satisfy these tests:\n```python\n" + "\n".join(shown) + "\n```"
        )
    return "\n\n".join(parts)


def build_messages(problem: Problem, num_prompt_tests: int = 1) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(problem, num_prompt_tests)},
    ]


def format_solution(code: str) -> str:
    return f"```python\n{code.strip()}\n```"
