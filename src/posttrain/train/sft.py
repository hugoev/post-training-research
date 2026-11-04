"""Supervised fine-tuning on reference solutions (loss on the assistant completion only)."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from posttrain.config import TrainConfig
from posttrain.data import load_problems
from posttrain.data.builders import sft_records
from posttrain.train.common import (
    ResourceMonitor,
    TokenRows,
    finish_run,
    load_model_and_tokenizer,
    peft_config,
    start_run,
)


def run_sft(cfg: TrainConfig) -> Path:
    run_dir = start_run(cfg, "sft")
    data = cfg.data
    train = Dataset.from_list(
        sft_records(
            load_problems(data.dataset, data.split, data.limit, data.task_ids_file),
            data.num_prompt_tests,
        )
    )
    evaluation = None
    if data.eval_split:
        evaluation = Dataset.from_list(
            sft_records(load_problems(data.dataset, data.eval_split), data.num_prompt_tests)
        )
    print(f"[sft] {len(train)} train / {len(evaluation) if evaluation else 0} eval examples")

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    monitor = ResourceMonitor()
    args = SFTConfig(output_dir=str(run_dir / "checkpoints"), seed=cfg.seed, **cfg.trainer)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=evaluation,
        processing_class=tokenizer,
        peft_config=peft_config(cfg.lora),
        callbacks=[monitor],
    )
    token_rows = None
    if cfg.lora.enabled and cfg.lora.trainable_tokens:
        if args.weight_decay:
            raise ValueError("trainable_tokens needs weight_decay=0: decay would shrink all rows")
        token_rows = TokenRows(trainer.model, tokenizer, cfg.lora.trainable_tokens)
        trainer.add_callback(token_rows)
    trainer.train()
    finish_run(trainer, run_dir, monitor)
    if token_rows is not None and trainer.is_world_process_zero():
        token_rows.save(run_dir / "final")
    return run_dir
