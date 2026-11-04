"""Generation backends with throughput accounting. Heavy imports are deferred."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from posttrain.config import GenerationConfig, ModelConfig


@dataclass
class GenerationOutput:
    texts: list[list[str]]  # [prompt][sample]
    # [prompt][sample]: True when generation hit max_new_tokens without a stop token.
    truncated: list[list[bool]] = field(default_factory=list)
    num_prompt_tokens: int = 0
    num_generated_tokens: int = 0
    wall_time_s: float = 0.0
    extra: dict = field(default_factory=dict)

    def stats(self) -> dict:
        num_sequences = sum(len(t) for t in self.texts)
        num_truncated = sum(sum(t) for t in self.truncated)
        return {
            "wall_time_s": self.wall_time_s,
            "num_sequences": num_sequences,
            "truncation_rate": num_truncated / max(num_sequences, 1),
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_generated_tokens": self.num_generated_tokens,
            "generated_tokens_per_s": self.num_generated_tokens / max(self.wall_time_s, 1e-9),
            "sequences_per_s": num_sequences / max(self.wall_time_s, 1e-9),
            **self.extra,
        }


_END_OF_TURN_TOKENS = ("<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>")


def stop_token_ids(tokenizer, generation_config=None) -> list[int]:
    """EOS plus the chat template's end-of-turn token.

    A base model's EOS is e.g. <|endoftext|>, but after chat-format SFT it ends turns with
    <|im_end|>; without stopping there, it keeps generating past its answer.
    """
    ids = {tokenizer.eos_token_id}
    configured = getattr(generation_config, "eos_token_id", None)
    if isinstance(configured, int):
        ids.add(configured)
    elif configured:
        ids.update(configured)
    vocab = tokenizer.get_vocab()
    ids.update(vocab[t] for t in _END_OF_TURN_TOKENS if t in vocab)
    return sorted(i for i in ids if i is not None)


class HFGenerator:
    def __init__(self, model_cfg: ModelConfig, gen_cfg: GenerationConfig):
        import torch

        from posttrain.train.common import load_model, load_tokenizer

        self.torch = torch
        self.cfg = gen_cfg
        self.tokenizer = load_tokenizer(model_cfg, padding_side="left")
        self.model = load_model(model_cfg)
        if model_cfg.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, model_cfg.adapter_path)
        self.model.eval()
        self.stop_ids = stop_token_ids(self.tokenizer, self.model.generation_config)

    def generate(self, conversations: list[list[dict]]) -> GenerationOutput:
        torch, tok, cfg = self.torch, self.tokenizer, self.cfg
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        prompts = [
            tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations
        ]
        # Each prompt is repeated num_samples times; sort by length to minimise padding.
        expanded = [(i, p) for i, p in enumerate(prompts) for _ in range(cfg.num_samples)]
        order = sorted(range(len(expanded)), key=lambda j: len(expanded[j][1]))
        texts: list[list[str]] = [[] for _ in prompts]
        truncated: list[list[bool]] = [[] for _ in prompts]
        prompt_tokens = generated_tokens = 0
        start = time.perf_counter()
        for b in range(0, len(order), cfg.batch_size):
            batch = [expanded[j] for j in order[b : b + cfg.batch_size]]
            enc = tok(
                [p for _, p in batch], return_tensors="pt", padding=True, add_special_tokens=False
            ).to(self.model.device)
            sampling = (
                {"do_sample": True, "temperature": cfg.temperature, "top_p": cfg.top_p}
                if cfg.temperature > 0
                else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
            )
            with torch.inference_mode():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=cfg.max_new_tokens,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=self.stop_ids,
                    **sampling,
                )
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            prompt_tokens += int(enc["attention_mask"].sum())
            for (i, _), seq in zip(batch, new_tokens, strict=True):
                ids = seq.tolist()
                stops = [k for k, t in enumerate(ids) if t in self.stop_ids]
                if stops:
                    ids = ids[: stops[0] + 1]
                generated_tokens += len(ids)
                texts[i].append(tok.decode(ids, skip_special_tokens=True))
                truncated[i].append(not stops)
        wall = time.perf_counter() - start
        extra = {"backend": "hf"}
        if torch.cuda.is_available():
            extra["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
        return GenerationOutput(texts, truncated, prompt_tokens, generated_tokens, wall, extra)


class VLLMGenerator:
    def __init__(self, model_cfg: ModelConfig, gen_cfg: GenerationConfig):
        from vllm import LLM

        self.cfg = gen_cfg
        self.adapter_path = model_cfg.adapter_path
        self.llm = LLM(
            model=model_cfg.name_or_path,
            revision=model_cfg.revision,
            dtype=model_cfg.dtype,
            seed=gen_cfg.seed,
            gpu_memory_utilization=gen_cfg.gpu_memory_utilization,
            max_model_len=gen_cfg.max_model_len,
            enable_lora=self.adapter_path is not None,
            max_lora_rank=64,
        )
        self.stop_ids = stop_token_ids(self.llm.get_tokenizer())

    def generate(self, conversations: list[list[dict]]) -> GenerationOutput:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        cfg = self.cfg
        params = SamplingParams(
            n=cfg.num_samples,
            temperature=cfg.temperature,
            top_p=cfg.top_p if cfg.temperature > 0 else 1.0,
            max_tokens=cfg.max_new_tokens,
            seed=cfg.seed,
            stop_token_ids=self.stop_ids,
        )
        lora = LoRARequest("adapter", 1, self.adapter_path) if self.adapter_path else None
        start = time.perf_counter()
        outputs = self.llm.chat(conversations, params, lora_request=lora, use_tqdm=True)
        wall = time.perf_counter() - start
        texts = [[o.text for o in out.outputs] for out in outputs]
        truncated = [[o.finish_reason == "length" for o in out.outputs] for out in outputs]
        prompt_tokens = sum(len(out.prompt_token_ids) for out in outputs)
        generated = sum(len(o.token_ids) for out in outputs for o in out.outputs)
        return GenerationOutput(
            texts, truncated, prompt_tokens, generated, wall, {"backend": "vllm"}
        )


def make_generator(model_cfg: ModelConfig, gen_cfg: GenerationConfig):
    if gen_cfg.backend == "hf":
        return HFGenerator(model_cfg, gen_cfg)
    if gen_cfg.backend == "vllm":
        return VLLMGenerator(model_cfg, gen_cfg)
    raise ValueError(f"Unknown generation backend {gen_cfg.backend!r}")
