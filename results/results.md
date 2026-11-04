# Results: Qwen2.5-Coder-1.5B, single RTX 3060 (12 GB)

All numbers: greedy decoding, one sample per problem, single training seed. Brackets are 95%
bootstrap confidence intervals over problems.

## pass@1 (%)

| Model | KodCode held-out (500) | MBPP+ (378) | HumanEval+ (163) |
|---|---|---|---|
| Base `Qwen2.5-Coder-1.5B` | 46.2 [41.8, 50.6] | 31.5 [27.0, 36.2] | 53.4 [46.0, 61.3] |
| SFT | 78.6 [75.0, 82.2] | 67.5 [62.7, 72.2] | 65.6 [58.3, 73.0] |
| SFT + DPO | 79.6 [76.0, 83.0] | 67.5 [62.7, 72.2] | 68.1 [60.7, 75.5] |
| SFT + GRPO | **80.0** [76.4, 83.4] | **68.5** [63.8, 73.3] | **72.4** [65.6, 79.1] |
| Reference: `Qwen2.5-Coder-1.5B-Instruct` | 64.4 [60.2, 68.6] | 61.4 [56.3, 66.4] | 68.7 [61.3, 75.5] |

## Paired comparisons against SFT (same problems)

Difference in pass@1 points with a bootstrap 95% CI, a sign-flip permutation p-value, and the
McNemar discordant counts (problems solved only by SFT vs only by the other model).

| Comparison | Benchmark | Difference | p | Only SFT / only other |
|---|---|---|---|---|
| DPO vs SFT | KodCode held-out | +1.0 [−0.8, +2.8] | 0.40 | 9 / 14 |
| DPO vs SFT | MBPP+ | 0.0 [−1.9, +1.9] | 1.00 | 6 / 6 |
| DPO vs SFT | HumanEval+ | +2.5 [0.0, +5.5] | 0.22 | 1 / 5 |
| GRPO vs SFT | KodCode held-out | +1.4 [−1.2, +4.0] | 0.36 | 19 / 26 |
| GRPO vs SFT | MBPP+ | +1.1 [−1.9, +4.2] | 0.61 | 15 / 19 |
| GRPO vs SFT | HumanEval+ | **+6.7** [+3.1, +11.0] | **0.001** | 0 / 11 |

## Reliability

| Model | Compile rate (KodCode / MBPP+ / HumanEval+) |
|---|---|
| Base | 72.2 / 76.7 / 89.0 |
| SFT, SFT + DPO, SFT + GRPO | 99.7–100 on all suites |
| Instruct | 99.4–99.6 |

After SFT, nearly all failures are wrong logic (failed tests), not unparseable code. No trained
model hits the 768-token generation cap on more than 0.3% of problems. Mean generation length
is 78–144 tokens per suite.

## Training cost

| Stage | Steps | Wall time | Peak GPU memory | Data |
|---|---|---|---|---|
| SFT (LoRA r=32 + 2 embedding rows) | 277 | 80 min | 9.6 GB | 8,835 verified solutions, 1 epoch |
| DPO (LoRA r=32) | 101 | 62 min | 7.0 GB | 3,230 execution-labelled pairs, 1 epoch |
| GRPO (LoRA r=32, colocated vLLM) | 300 | 133 min | 11.0 GB | 1,200 prompts × 8 rollouts |

Preference pairs and GRPO prompts both come from 8 samples of the SFT model on 3,000 training
problems (pass@1 72.6%, pass@8 93.4%). 1,601 problems were solved in some but not all samples;
those feed both DPO and GRPO.

## GRPO training dynamics

| Steps | Rollout pass rate | Groups with zero reward variance | Mean completion length | Entropy |
|---|---|---|---|---|
| 1–50 | 65.4% | 26% | 149 | 0.254 |
| 51–100 | 63.8% | 29% | 152 | 0.232 |
| 101–150 | 70.4% | 32% | 140 | 0.212 |
| 151–200 | 73.2% | 38% | 143 | 0.190 |
| 201–234 | 71.3% | 46% | 133 | 0.182 |

The SFT model's pass rate on these prompts is 61.2%. Nearly every prompt is seen once, so the
rise in rollout pass rate reflects improvement on new problems. Groups in which all 8 rollouts
get the same reward produce no gradient; their growing share shows the filtered set running out
of signal as the policy improves.

## Findings

1. **SFT on verified solutions gives most of the gain** (+12 to +36 points) and fixes output
   reliability. It beats the Instruct model on KodCode and MBPP+, but not on HumanEval+, whose
   docstring-completion format is least like the training data.
2. **GRPO with execution rewards adds a smaller gain, concentrated out of distribution.** The
   only significant improvement over SFT is HumanEval+ (+6.7, p = 0.001, 11 problems gained,
   none lost; it holds under a Bonferroni correction for three benchmarks). GRPO then also
   exceeds the Instruct model there.
3. **No reward hacking was observed.** Completions got slightly shorter, timeouts fell to zero,
   and format compliance stayed at 100%.
4. **Offline DPO on the same execution signal barely moved.** Final reward margin was about
   0.08 and pair accuracy about 66%. This reflects conservative settings (β = 0.1, lr 1e-5,
   one epoch) and fixed offline data; it is not compute-matched with GRPO, which saw about
   9,600 fresh on-policy samples.
5. **GRPO's signal decays and its entropy falls** as training progresses, so longer runs need
   harder problems or periodic re-filtering. Whether lower entropy reduces pass@k is untested.

## Caveats

- One seed per method, one hyperparameter setting each, greedy single-sample evaluation.
- HumanEval+ has 163 problems; the significant result rests on 11 discordant problems.
- No sampled pass@k yet, so this does not show whether GRPO adds capability or only sharpens
  the top answer.

## Implementation note: untrained chat tokens

The base model's `<|im_start|>` and `<|im_end|>` embeddings are untrained: their norm is 0.395
against about 0.67 for ordinary tokens. With embeddings tied, LoRA on linear layers cannot teach
the model to emit `<|im_end|>`, so the first SFT run never stopped generating while still
scoring well on pass@1. The SFT config therefore also trains those two embedding rows
(`lora.trainable_tokens`), and every evaluation now reports a truncation rate.
