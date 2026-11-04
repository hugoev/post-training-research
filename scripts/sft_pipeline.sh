#!/usr/bin/env bash
# SFT -> merge -> evaluation suite -> training-set rollouts (inputs for DPO and GRPO).
#   scripts/sft_pipeline.sh [sft config] [extra sft overrides...]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

config=${1:-configs/sft/kodcode_lora.yaml}
shift || true
base=$(uv run python -c "
from posttrain.config import TrainConfig, load_config
print(load_config(TrainConfig, '$config').model.name_or_path)" 2>/dev/null)

uv run posttrain sft "$config" "$@"
run=$(ls -td runs/sft/*/ | head -1)
run=${run%/}
uv run posttrain merge --base "$base" --adapter "$run/final" --output "$run/merged"
scripts/eval_suite.sh "sft-$(basename "$run")" "$run/merged"
uv run posttrain eval configs/eval/rollouts_kodcode_train.yaml \
    run_name="rollouts-$(basename "$run")" model.name_or_path="$run/merged"
echo "[pipeline] done: $run"
