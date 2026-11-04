"""GRPO with rewards from executing generated programs against hidden tests."""

from __future__ import annotations

import os
import socket
from pathlib import Path

from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from posttrain.config import TrainConfig
from posttrain.data import load_problems
from posttrain.data.builders import grpo_records
from posttrain.execution import Executor
from posttrain.rewards import ExecutionReward, format_reward
from posttrain.runs import write_json
from posttrain.train.common import (
    ResourceMonitor,
    finish_run,
    load_model_and_tokenizer,
    peft_config,
    start_run,
)


class ExecutionStatusLogger(TrainerCallback):
    """Logs how rollouts fail (syntax, timeout, crash...) to surface reward hacking early."""

    def __init__(self, reward: ExecutionReward):
        self.reward = reward

    def on_log(self, args, state, control, logs=None, **kwargs):
        counts = self.reward.status_counts
        total = sum(counts.values())
        if logs is None or not total:
            return
        fractions = {f"rollout_status/{s}": c / total for s, c in counts.items()}
        logs.update(fractions)
        # Trainer appends to log_history before callbacks run, so patch the stored entry too.
        if state.log_history:
            state.log_history[-1].update(fractions)
        counts.clear()


def _ensure_master_port() -> None:
    """Colocated vLLM opens a torch.distributed store. TRL prefers port 29500, and its free-port
    check (bind with SO_REUSEADDR, no listen) can pass while the port is taken, e.g. by a Windows
    process under WSL, so let the OS pick a port unless one was set explicitly."""
    if os.environ.get("MASTER_PORT"):
        return
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])


def run_grpo(cfg: TrainConfig) -> Path:
    if cfg.trainer.get("use_vllm") and cfg.trainer.get("vllm_mode", "colocate") == "colocate":
        _ensure_master_port()
    run_dir = start_run(cfg, "grpo")
    data = cfg.data
    problems = load_problems(data.dataset, data.split, data.limit, data.task_ids_file)
    train = Dataset.from_list(grpo_records(problems, data.num_prompt_tests))
    print(f"[grpo] {len(train)} prompts, correctness reward = {cfg.reward.correctness}")

    correctness = ExecutionReward(
        Executor.from_config(cfg.execution), cfg.reward.correctness, cfg.execution.sanitize
    )
    model, tokenizer = load_model_and_tokenizer(cfg.model, padding_side="left")
    monitor = ResourceMonitor()
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness, format_reward],
        args=GRPOConfig(output_dir=str(run_dir / "checkpoints"), seed=cfg.seed, **cfg.trainer),
        train_dataset=train,
        processing_class=tokenizer,
        peft_config=peft_config(cfg.lora),
        callbacks=[monitor, ExecutionStatusLogger(correctness)],
    )
    trainer.train()
    finish_run(trainer, run_dir, monitor)
    if trainer.is_world_process_zero():
        write_json(run_dir / "reward_config.json", {"correctness": cfg.reward.correctness})
    return run_dir
