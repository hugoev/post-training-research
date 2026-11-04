# Post-Training for Verifiable Code Generation

An end-to-end system for studying how supervised fine-tuning (SFT), Direct Preference
Optimization (DPO) and Group Relative Policy Optimization (GRPO) change the coding ability of
open-weight language models, measured by executing generated programs against test cases.

**Research question:** can post-training improve a model's ability to solve programming problems
with verifiable correctness, and how do different optimization methods affect generalization,
reliability and computational cost?

## Results

Qwen2.5-Coder-1.5B on one RTX 3060 (12 GB); pass@1 (%) with greedy decoding, single seed.
Full tables, paired tests, training cost and caveats: [`results/results.md`](results/results.md).

| Model | KodCode held-out | MBPP+ | HumanEval+ |
|---|---|---|---|
| Base | 46.2 | 31.5 | 53.4 |
| SFT | 78.6 | 67.5 | 65.6 |
| SFT + DPO | 79.6 | 67.5 | 68.1 |
| SFT + GRPO | **80.0** | **68.5** | **72.4** |
| Reference: Qwen2.5-Coder-1.5B-Instruct | 64.4 | 61.4 | 68.7 |

- SFT on verified solutions gives most of the gain and makes nearly all output compile.
- GRPO with execution rewards adds a smaller gain over SFT. It is significant only on the
  out-of-distribution HumanEval+ (+6.7 points, p = 0.001; 11 problems gained, none lost).
- Offline DPO on the same execution signal changed little (no significant difference).
- Training took 80 min (SFT), 62 min (DPO) and 133 min (GRPO) on the single 12 GB GPU.

## Experimental design

The policy starts from the **base** `Qwen2.5-Coder-1.5B`, so every gain is attributable to our
own post-training. `Qwen2.5-Coder-1.5B-Instruct` (Qwen's industrial post-training of the same
base) is evaluated as an external reference point, not as a starting point.

| Config | Starting point | Training data |
|---|---|---|
| Base | `Qwen2.5-Coder-1.5B` | none |
| SFT | Base | verified KodCode solutions (8,835 problems) |
| SFT + DPO | merged SFT | pairs of the SFT model's own samples, labelled by execution |
| SFT + GRPO | merged SFT | KodCode prompts the SFT model sometimes solves; reward from hidden tests |
| Reference | `Qwen2.5-Coder-1.5B-Instruct` | (Qwen's own post-training) |

### Data

Training uses [KodCode-Light-RL-10K](https://huggingface.co/datasets/KodCode/KodCode-Light-RL-10K):
10,000 problems with pytest test files and difficulty labels. `posttrain prepare-data kodcode`:

1. drops the 1.5% of problems whose tests need pytest fixtures, marks or test classes;
2. keeps only problems whose reference solution passes in our sandbox (9,537 of 9,851);
3. removes problems with ≥50% 10-gram overlap, in either direction, with any evaluation set (2);
4. splits deterministically into train (8,835), validation (200) and held-out (500).

### Evaluation suites

| Suite | Problems | Role |
|---|---|---|
| KodCode held-out | 500 | in-distribution generalization to unseen problems |
| MBPP+ (EvalPlus) | 378 | out of distribution, with hundreds of extra inputs per problem |
| HumanEval+ (EvalPlus) | 163 | out of distribution, function completion from a docstring |
| MBPP / HumanEval | 500 / 164 | original versions, for comparison with published numbers |

Validation is used for checkpoint and hyperparameter selection; held-out numbers are only
reported. EvalPlus test suites catch solutions that pass the original 3 asserts by accident.
HumanEval+/32 is excluded because its own test cannot pass (`KNOWN_BROKEN` in
`data/problems.py`). Every reference solution of every suite has been run through the sandbox.

### Metrics

- **pass@k** uses the unbiased estimator from Chen et al. (2021), with 95% bootstrap confidence
  intervals over problems.
- **Test-case pass rate, compile rate and execution rate** are reported along with a breakdown
  of failure types (assertion failures, syntax errors, runtime errors, timeouts, crashes).
- **Paired comparisons** (`posttrain compare`) report the per-problem difference with a bootstrap
  confidence interval and a sign-flip permutation p-value, plus an exact McNemar test for greedy
  pass/fail.
- **Systems metrics**: generation throughput (tokens/s), peak GPU memory, training wall time and
  tokens seen.

### Reward design and reward hacking

Tests are hidden from the model; prompts show only the required function signature (KodCode),
the docstring (HumanEval) or one example assert (MBPP). The sandbox
(`src/posttrain/execution/`) counts a test as passed only when an unforgeable, per-run
nonce-tagged event is emitted after that test completes. Exiting early, printing fake results,
infinite loops, memory blowups and destructive filesystem calls all fail. GRPO logs the
distribution of rollout outcomes (`rollout_status/*`) at every step.

Before execution, generated code is reduced to its top-level definitions (imports, functions,
classes, assignments). Otherwise a correct function followed by a wrong self-written `assert`
fails at import time and is scored on the model's own tests. This applies identically to
evaluation and rewards (`execution.sanitize`).

**Sparse GRPO signal.** A GRPO group in which every rollout gets the same reward has zero
advantage and no gradient. `posttrain filter-difficulty` keeps only training problems the starting
policy solves in some but not all of 8 samples.

The sandbox isolates processes and limits resources; it is not a security boundary against
deliberately adversarial code. Run it inside a container for that.

## Setup

```bash
sudo apt install build-essential     # Triton (used by TRL's loss kernels) JIT-compiles C code
uv sync --extra train --extra vllm   # PyTorch, Transformers, TRL, PEFT, vLLM
uv run pytest                        # sandbox, metrics, data and reward tests (CPU only)
source scripts/env.sh                # no-op unless gcc / nvcc are missing (see below)
```

`scripts/env.sh` applies workarounds only when the toolchain is missing: Zig's bundled C
compiler for Triton, the WSL `libcuda` path, and vLLM's PyTorch sampler instead of FlashInfer's
(which needs `nvcc`).

## Pipeline

The whole experiment runs as two scripts:

```bash
scripts/sft_pipeline.sh                    # SFT -> merge -> eval suite -> training-set rollouts
scripts/post_sft.sh runs/sft/<run>         # DPO and GRPO from the merged SFT model, each evaluated
```

The individual steps:

```bash
source scripts/env.sh

# 0. Data and contamination audit
uv run posttrain prepare-data kodcode
uv run posttrain contamination --train kodcode:train --eval mbppplus:test humanevalplus:test

# 1. Baselines (base model and the Instruct reference)
scripts/eval_suite.sh base Qwen/Qwen2.5-Coder-1.5B
scripts/eval_suite.sh instruct Qwen/Qwen2.5-Coder-1.5B-Instruct

# 2. SFT -> merge -> evaluate
uv run posttrain sft configs/sft/kodcode_lora.yaml
uv run posttrain merge --base Qwen/Qwen2.5-Coder-1.5B \
    --adapter runs/sft/<run>/final --output runs/sft/<run>/merged
scripts/eval_suite.sh sft runs/sft/<run>/merged

# 3. Sample the SFT model on training problems (feeds both DPO and GRPO)
uv run posttrain eval configs/eval/rollouts_kodcode_train.yaml model.name_or_path=runs/sft/<run>/merged

# 4. DPO
uv run posttrain build-prefs --samples runs/eval/<rollouts>/samples.jsonl \
    --output data/prefs/sft_kodcode.jsonl
uv run posttrain dpo configs/dpo/kodcode_lora.yaml \
    model.name_or_path=runs/sft/<run>/merged preference_file=data/prefs/sft_kodcode.jsonl

# 5. GRPO on problems with 0 < pass rate < 1
uv run posttrain filter-difficulty --samples runs/eval/<rollouts>/samples.jsonl \
    --output data/tasks/grpo_learnable.txt
uv run posttrain grpo configs/grpo/kodcode_lora.yaml model.name_or_path=runs/sft/<run>/merged

# 6. Results
uv run posttrain report runs/eval/*heldout* runs/eval/*plus*
uv run posttrain compare runs/eval/<base-run> runs/eval/<sft-run>
```

Any config value can be overridden from the command line with dotted keys
(`trainer.max_steps=10`, `generation.backend=hf`, `data.limit=20`). Configs can `inherit:` a
parent file. Each run directory stores the resolved config, package versions, git commit, GPU,
metrics and per-sample outputs.

### Multi-GPU

```bash
uv run accelerate launch --config_file configs/accelerate/ddp.yaml -m posttrain.cli sft <config>
uv run accelerate launch --config_file configs/accelerate/fsdp.yaml -m posttrain.cli sft <config>
```

## Layout

```
configs/            eval / sft / dpo / grpo experiment configs, accelerate launch configs
scripts/            env.sh (toolchain workarounds), eval_suite.sh, sft_pipeline.sh, post_sft.sh
results/            reported results
src/posttrain/
  config.py         typed configs, inheritance, dotted overrides
  runs.py           run directories, environment capture, seeding
  data/             problem schema, loaders, prompts, KodCode preparation, contamination, builders
  execution/        sandboxed subprocess executor, code extraction and sanitization
  eval/             generation backends (HF, vLLM), metrics, statistics, eval runner, reports
  rewards.py        GRPO reward functions
  train/            SFT, DPO, GRPO, adapter merging
tests/              CPU-only unit tests
```

## Roadmap

- [x] Executable evaluation with pass@k, confidence intervals and paired tests
- [x] SFT / DPO / GRPO pipelines with LoRA and QLoRA
- [x] Verified, decontaminated KodCode training data; EvalPlus evaluation
- [x] Difficulty filtering for GRPO
- [x] Baselines, SFT, DPO and GRPO results (single seed)
- [ ] Sampled pass@k; multiple seeds
- [ ] Reward-design ablations: binary vs fractional rewards, KL coefficient, loss type
- [ ] LoRA vs QLoRA (quality, memory, step time); HF vs vLLM rollout throughput in GRPO
- [ ] Ray orchestration for sweeps across seeds and hyperparameters
- [ ] PyTorch Profiler / Nsight traces of training steps; custom Triton kernels
- [ ] Containerised sandbox (Docker or gVisor)
- [ ] Technical report
