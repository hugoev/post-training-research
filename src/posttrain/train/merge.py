"""Merge a LoRA adapter into full-precision base weights for the next training stage or vLLM."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from posttrain.train.common import TOKEN_ROWS_FILE, align_stop_tokens, resolve_dtype


def merge_adapter(base: str, adapter: str, output: str, dtype: str = "bfloat16") -> Path:
    # Merge into unquantized weights: merging into a 4-bit base would bake in rounding error.
    model = AutoModelForCausalLM.from_pretrained(base, dtype=resolve_dtype(dtype))
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    rows_file = Path(adapter) / TOKEN_ROWS_FILE
    if rows_file.exists():
        saved = torch.load(rows_file)
        embeddings = merged.get_input_embeddings().weight
        with torch.no_grad():
            embeddings[saved["token_ids"]] = saved["rows"].to(embeddings)
        print(f"[merge] restored embedding rows for token ids {saved['token_ids']}")
    # The adapter's tokenizer carries any EOS change made during SFT; the base's
    # generation config does not, so re-derive the stop tokens from it.
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    align_stop_tokens(merged, tokenizer)
    out = Path(output)
    merged.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"[merge] wrote {out}")
    return out
