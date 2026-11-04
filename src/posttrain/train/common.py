"""Model, tokenizer and LoRA construction shared by all training stages and HF inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback

from posttrain.config import LoraConfig, ModelConfig, TrainConfig, to_dict
from posttrain.runs import create_run_dir, set_seed, write_json

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"Unknown dtype {name!r}; use one of {sorted(_DTYPES)}")
    return _DTYPES[name]


def load_tokenizer(cfg: ModelConfig, padding_side: str = "right"):
    tok = AutoTokenizer.from_pretrained(
        cfg.name_or_path, revision=cfg.revision, padding_side=padding_side
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg: ModelConfig):
    dtype = resolve_dtype(cfg.dtype)
    kwargs: dict[str, Any] = {"revision": cfg.revision, "dtype": dtype}
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    if torch.cuda.is_available():
        from accelerate import PartialState

        kwargs["device_map"] = {"": PartialState().local_process_index}
    if cfg.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=dtype,
        )
    return AutoModelForCausalLM.from_pretrained(cfg.name_or_path, **kwargs)


def align_stop_tokens(model, tokenizer) -> None:
    """Make generation (including TRL's GRPO rollouts) stop at the end-of-turn token."""
    from posttrain.eval.generate import stop_token_ids

    model.generation_config.eos_token_id = stop_token_ids(tokenizer, model.generation_config)


def load_model_and_tokenizer(cfg: ModelConfig, padding_side: str = "right"):
    tokenizer = load_tokenizer(cfg, padding_side)
    model = load_model(cfg)
    align_stop_tokens(model, tokenizer)
    return model, tokenizer


def peft_config(cfg: LoraConfig):
    if not cfg.enabled:
        return None
    from peft import LoraConfig as PeftLoraConfig

    return PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=cfg.target_modules,
        task_type="CAUSAL_LM",
    )


TOKEN_ROWS_FILE = "token_rows.pt"


class TokenRows(TrainerCallback):
    """Trains selected rows of the tied embedding matrix in place, alongside LoRA.

    PEFT's ``trainable_token_indices`` rebuilds the full embedding matrix on every forward,
    which overflows a 12 GB GPU; masking the gradient of the shared weight instead costs one
    full-size gradient plus optimizer state. Rows are saved next to each adapter checkpoint
    and written back by ``merge_adapter``. Must be created after the trainer wraps the model
    with PEFT (which freezes base weights) and before ``train()`` builds the optimizer.
    """

    def __init__(self, model, tokenizer, tokens: list[str]):
        vocab = tokenizer.get_vocab()
        missing = [t for t in tokens if t not in vocab]
        if missing:
            raise ValueError(f"trainable_tokens not in the vocabulary: {missing}")
        self.token_ids = [vocab[t] for t in tokens]
        self.weight = model.get_input_embeddings().weight
        output = model.get_output_embeddings()
        if output is None or output.weight is not self.weight:
            raise ValueError("trainable_tokens requires tied input/output embeddings")
        mask = torch.zeros(self.weight.shape[0], 1, dtype=self.weight.dtype)
        mask[self.token_ids] = 1
        mask = mask.to(self.weight.device)
        self.weight.requires_grad_(True)
        self.weight.register_hook(lambda grad: grad * mask)

    def save(self, directory: Path | str) -> None:
        rows = self.weight.detach()[self.token_ids].cpu()
        torch.save({"token_ids": self.token_ids, "rows": rows}, Path(directory) / TOKEN_ROWS_FILE)

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.save(Path(args.output_dir) / f"checkpoint-{state.global_step}")


def start_run(cfg: TrainConfig, stage: str) -> Path:
    set_seed(cfg.seed)
    from accelerate import PartialState

    state = PartialState()
    run_dir = None
    if state.is_main_process:
        run_dir = create_run_dir(cfg.output_dir, stage, cfg.run_name, to_dict(cfg))
    # Every rank must write checkpoints to the directory the main rank created.
    run_dir = _broadcast_path(run_dir, state)
    return run_dir


def _broadcast_path(path: Path | None, state) -> Path:
    if state.num_processes == 1:
        return path
    import torch.distributed as dist

    holder = [str(path) if path else None]
    dist.broadcast_object_list(holder, src=0)
    return Path(holder[0])


class ResourceMonitor(TrainerCallback):
    """Records wall time, peak GPU memory and throughput for the systems comparisons."""

    def __init__(self):
        self.start = None
        self.summary: dict[str, Any] = {}

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_end(self, args, state, control, **kwargs):
        logged_tokens = [e["num_tokens"] for e in state.log_history if "num_tokens" in e]
        self.summary = {
            "train_wall_time_s": time.perf_counter() - self.start,
            "global_steps": state.global_step,
            "num_tokens": logged_tokens[-1] if logged_tokens else None,
        }
        if torch.cuda.is_available():
            self.summary["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
            self.summary["gpu"] = torch.cuda.get_device_name()


def finish_run(trainer, run_dir: Path, monitor: ResourceMonitor) -> None:
    trainer.save_model(str(run_dir / "final"))
    if trainer.is_world_process_zero():
        trainer.processing_class.save_pretrained(str(run_dir / "final"))
        write_json(
            run_dir / "train_metrics.json",
            {"resources": monitor.summary, "log_history": trainer.state.log_history},
        )
        print(f"[train] wrote {run_dir}")
