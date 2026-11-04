"""Typed experiment configs loaded from YAML with dotted command-line overrides.

Example:
    cfg = load_config(EvalConfig, "configs/eval/mbpp_test.yaml",
                      ["generation.num_samples=10", "generation.temperature=0.8"])
"""

from __future__ import annotations

import dataclasses
import re
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


class _Loader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 scalars: `1e-5` is a float and only true/false are booleans.

    YAML 1.1 would read `save_strategy: no` as False and `1e-5` as a string.
    """


_BOOL = "tag:yaml.org,2002:bool"
_Loader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in resolvers if tag != _BOOL]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver(
    _BOOL, re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+$"),
    list("-+0123456789."),
)


def _yaml_load(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)  # noqa: S506 - _Loader derives from SafeLoader


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    revision: str | None = None
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    attn_implementation: str | None = None
    # LoRA adapter applied on top of the base weights (evaluation only).
    adapter_path: str | None = None


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | str = "all-linear"
    # Embedding rows to train alongside LoRA (with tied weights, also the output rows).
    # Base models often leave chat-template tokens like <|im_end|> untrained, so the model
    # cannot learn to emit them through LoRA on linear layers alone.
    trainable_tokens: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    dataset: str = "mbpp"
    # Splits can be concatenated with "+", e.g. "train+validation".
    split: str = "train"
    eval_split: str | None = None
    limit: int | None = None
    # Whitespace-separated task ids to keep (e.g. from `posttrain filter-difficulty`).
    task_ids_file: str | None = None
    # Number of test asserts shown in the prompt; the rest stay hidden from the model.
    num_prompt_tests: int = 1


@dataclass
class GenerationConfig:
    backend: str = "hf"  # "hf" or "vllm"
    num_samples: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    batch_size: int = 16
    seed: int = 0
    gpu_memory_utilization: float = 0.85
    max_model_len: int | None = 4096


@dataclass
class ExecutionConfig:
    timeout_s: float = 20.0
    per_test_timeout_s: float = 5.0
    memory_mb: int = 2048
    max_workers: int = 8
    # Strip top-level code other than definitions (the model's own asserts, prints, demos).
    sanitize: bool = True


@dataclass
class RewardConfig:
    # "binary": 1 only if every test passes. "fraction": share of tests passed.
    correctness: str = "binary"


@dataclass
class EvalConfig:
    run_name: str = "eval"
    output_dir: str = "runs"
    seed: int = 0
    ks: list[int] = field(default_factory=lambda: [1])
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=lambda: DataConfig(split="test"))
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)


@dataclass
class TrainConfig:
    run_name: str = "train"
    output_dir: str = "runs"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    # DPO only: JSONL of {prompt, chosen, rejected} produced by `posttrain build-prefs`.
    preference_file: str | None = None
    # Passed verbatim to the TRL config class (SFTConfig / DPOConfig / GRPOConfig).
    trainer: dict[str, Any] = field(default_factory=dict)


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"Override must look like key.subkey=value, got {item!r}")
        *parents, leaf = key.split(".")
        node = cfg
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override {key!r}: {part!r} is not a mapping")
        node[leaf] = _yaml_load(raw)
    return cfg


def from_dict(cls: type[T], data: dict[str, Any]) -> T:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        tp = hints[name]
        if dataclasses.is_dataclass(tp) and isinstance(value, dict):
            value = from_dict(tp, value)
        kwargs[name] = value
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a config, resolving `inherit: other.yaml` (relative to this file) recursively."""
    raw = _yaml_load(path.read_text()) or {}
    parent = raw.pop("inherit", None)
    if parent is None:
        return raw
    return _deep_merge(_read_yaml(path.parent / parent), raw)


def load_config(cls: type[T], path: str | Path | None, overrides: list[str] = ()) -> T:
    raw: dict[str, Any] = {}
    if path is not None:
        raw = _read_yaml(Path(path))
    return from_dict(cls, apply_overrides(raw, list(overrides)))


def to_dict(cfg: Any) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
