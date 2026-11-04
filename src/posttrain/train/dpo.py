"""Direct Preference Optimization on execution-verified preference pairs.

With LoRA, the frozen reference policy is the starting model with the adapter disabled,
so start from a merged SFT checkpoint to make the KL term anchor to the SFT policy.
"""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from posttrain.config import TrainConfig
from posttrain.runs import read_jsonl
from posttrain.train.common import (
    ResourceMonitor,
    finish_run,
    load_model_and_tokenizer,
    peft_config,
    start_run,
)


def run_dpo(cfg: TrainConfig) -> Path:
    if not cfg.preference_file:
        raise ValueError("DPO needs preference_file (create one with `posttrain build-prefs`)")
    rows = read_jsonl(Path(cfg.preference_file))
    train = Dataset.from_list(
        [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]
    )
    run_dir = start_run(cfg, "dpo")
    print(f"[dpo] {len(train)} preference pairs from {cfg.preference_file}")

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    monitor = ResourceMonitor()
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=DPOConfig(output_dir=str(run_dir / "checkpoints"), seed=cfg.seed, **cfg.trainer),
        train_dataset=train,
        processing_class=tokenizer,
        peft_config=peft_config(cfg.lora),
        callbacks=[monitor],
    )
    trainer.train()
    finish_run(trainer, run_dir, monitor)
    return run_dir
