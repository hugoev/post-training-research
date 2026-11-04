#!/usr/bin/env bash
# After sft_pipeline.sh: DPO and GRPO from the merged SFT model, each merged and evaluated.
#   scripts/post_sft.sh <sft run dir> [rollouts eval run dir]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

sft=${1%/}
name=$(basename "$sft")
rollouts=${2:-$(ls -td runs/eval/rollouts-"$name"-*/ | head -1)}
rollouts=${rollouts%/}
prefs=data/prefs/$name.jsonl
tasks=data/tasks/grpo_learnable.txt
mkdir -p data/prefs data/tasks

uv run posttrain build-prefs --samples "$rollouts/samples.jsonl" --output "$prefs"
uv run posttrain filter-difficulty --samples "$rollouts/samples.jsonl" --output "$tasks"

# GRPO (colocated vLLM + training on one 12 GB GPU) is the likeliest stage to run out of
# memory, so smoke-test it before spending hours on DPO.
uv run posttrain grpo configs/grpo/kodcode_lora.yaml model.name_or_path="$sft/merged" \
    run_name=grpo-smoke output_dir=runs/smoke trainer.max_steps=3 trainer.save_strategy=no

train_merge_eval() {
    local stage=$1 run
    shift
    uv run posttrain "$stage" "configs/$stage/kodcode_lora.yaml" model.name_or_path="$sft/merged" "$@"
    run=$(ls -td runs/"$stage"/*/ | head -1)
    run=${run%/}
    uv run posttrain merge --base "$sft/merged" --adapter "$run/final" --output "$run/merged"
    scripts/eval_suite.sh "$stage-$(basename "$run")" "$run/merged"
}

train_merge_eval dpo preference_file="$prefs"
train_merge_eval grpo
echo "[post-sft] done"
