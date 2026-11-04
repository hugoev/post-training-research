#!/usr/bin/env bash
# Evaluate one model on the full suite with identical settings.
#   scripts/eval_suite.sh <run-name-prefix> <model path or hub id> [extra overrides...]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

prefix=$1
model=$2
shift 2
for suite in kodcode_heldout mbppplus humanevalplus; do
    uv run posttrain eval "configs/eval/${suite}.yaml" \
        run_name="${prefix}-${suite}" model.name_or_path="${model}" "$@"
done
