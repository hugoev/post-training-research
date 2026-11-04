"""Extract the program from a model completion."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

_FENCE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_OPEN_FENCE = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\n(.*)$", re.DOTALL | re.IGNORECASE)


@dataclass
class ExtractedCode:
    code: str
    fenced: bool


def extract_code(text: str, entry_point: str | None = None) -> ExtractedCode:
    """Pick the fenced block defining `entry_point`, else the longest block.

    Falls back to an unterminated fence (truncated generation) and finally to the raw text.
    """
    blocks = [b.strip("\n") for b in _FENCE.findall(text)]
    if blocks:
        if entry_point:
            pattern = re.compile(rf"^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", re.M)
            defining = [b for b in blocks if pattern.search(b)]
            if defining:
                return ExtractedCode(max(defining, key=len), True)
        return ExtractedCode(max(blocks, key=len), True)
    match = _OPEN_FENCE.search(text)
    if match:
        return ExtractedCode(match.group(1).strip("\n"), False)
    return ExtractedCode(text.strip("\n"), False)


_DEFINITIONS = (
    ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    ast.Assign, ast.AnnAssign,
)


def sanitize(code: str) -> str:
    """Keep only top-level definitions, dropping the model's own asserts, prints and demos.

    Without this, a correct function followed by a wrong self-written assert fails at load
    time and is scored on the model's tests rather than the benchmark's. Code that does not
    parse is returned unchanged so syntax errors are still counted.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code
    kept = [node for node in tree.body if isinstance(node, _DEFINITIONS)]
    if len(kept) == len(tree.body):
        return code
    lines = code.splitlines()
    segments = []
    for node in kept:
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        segments.append("\n".join(lines[start - 1 : node.end_lineno]))
    return "\n\n".join(segments)
