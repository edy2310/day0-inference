# Metrics and gate logic

This document defines every metric day0 reports, the estimator it uses, the intervals, and the gate
that turns them into a status. The code is the reference: `day0/quality/metrics.py`,
`day0/stats.py`, `day0/gate.py`, `day0/perf/metrics.py`.

## Units and unit of analysis

- Every KL is `KL(P_ref || Q_cand)` in **nats** (natural logarithm).
- The statistical unit is the **prompt** (suite item). Per-position values are averaged within a
  prompt first, so positions are never treated as independent samples (no pseudo-replication).
- Logprobs stored in artifacts are rounded to 6 decimals. Candidate metrics are computed from the
  server's floats, so an identical candidate can show KL around 1e-13 instead of exactly 0.

## Two paths

- **Prefill (teacher-forced).** The reference output is appended to the prompt and sent to
  `/v1/completions` with `prompt_logprobs=k` and `max_tokens=1`. Position `m + j`, where `m` is the
  prompt length, holds the distribution that predicts output token `j`. This path compares full
  distributions at identical contexts. It does not exercise speculative decoding, and vLLM does not
  use the prefix cache for prompt-logprob requests.
- **Generation (free-running greedy).** Both sides generate greedily with `logprobs=k`. This path
  exercises the real decode loop and measures what users see (flips, exact match). Distributions
  are compared only up to the first divergence.

## Top-k KL estimator (`topk_kl`)

The estimator sees only the top-k of each side, so it merges everything else into a TAIL bucket.

```text
topk_kl(p_lp, q_lp, eps=1e-10):
  U = keys(p_lp) ∪ keys(q_lp)
  for each side s in {p, q}:
      known_s   = {t: exp(lp) for t, lp in s}
      tail_s    = max(0, 1 - sum(known_s))
      floor_s   = min(known_s)
      missing_s = [t for t in U if t not in known_s]
      imputed_s = min(floor_s, tail_s / len(missing_s)) if missing_s else 0
      dist_s[t] = known_s[t] or imputed_s, for t in U
      dist_s[TAIL] = max(eps, tail_s - imputed_s * len(missing_s))
      clip every value to >= eps, then renormalize
  return sum over U ∪ {TAIL} of p * log(p / q), clamped to >= 0
```

**Why this is a lower bound (merge step).** Merging outcomes into a TAIL bucket is a
coarse-graining. By the data-processing inequality it cannot increase KL. With exact probabilities
and no imputation, the estimate is therefore a lower bound of the true KL. Imputation for tokens
missing from one side can push the estimate in either direction. The error is measured against the
exact full-vocabulary KL by `scripts/validate_kl.py` (see `docs/kl_validation.md`).

The unit tests check the identical case (zero), the full-vocabulary case (equal to the exact KL),
disjoint sets (finite and positive), non-negativity over 1000 seeded random pairs, and the merge
bound over 300 seeded random pairs with the same token set on both sides.

## Per-prompt metrics

| Metric | Definition |
|---|---|
| `prefill_kl_mean` | Mean over output positions of `topk_kl(ref_prefill_topk[j], cand_prefill_topk[j])`. |
| `prefill_kl_p99_position` | 99th percentile over positions of the same per-position KL (tail diagnostic). |
| `prefill_top1_agreement` | Fraction of positions where the argmax matches. Ties go to the lowest token id. |
| `prefill_nll_delta_mean` | Mean of `ref_lp[j] - cand_lp[j]` over output tokens. Positive means the candidate finds the reference continuation less likely. |
| `exact_match` | Output token ids equal. If the server returned no ids, output text equal. |
| `first_divergence` | First output position where the outputs differ. A prefix diverges at the end of the shorter output. |
| `first_divergence_frac` | `first_divergence / max(len(ref_output), 1)`. |
| `gen_kl_prefix_mean` | Mean of `topk_kl` over generation positions before the first divergence (all positions if identical). Position 0 always has an identical context. `None` if the prefix is empty or logprobs are missing. |
| `flip` | `ref_correct != cand_correct`. `None` when either side has no gold label. |
| `flip_direction` | `correct_to_wrong`, `wrong_to_correct`, or `None`. |

**Answer correctness.** Number items take the first match of `answer is N` and otherwise the last
number in the text, compared numerically with `Decimal`. Exact items compare after stripping
whitespace and trailing punctuation and case-folding.

## Aggregates and intervals

Per configuration, per slice, and overall (`day0/gate.py::aggregate`):

| Metric | Unit | Interval |
|---|---|---|
| `flip_rate` | gold items | Wilson |
| `flip_rate_c2w`, `flip_rate_w2c` | gold items | Wilson |
| `exact_match_rate` | all items | Wilson |
| `accuracy_ref`, `accuracy_cand` | gold items | Wilson (reported, not gated) |
| `prefill_kl_mean`, `prefill_top1_agreement`, `prefill_nll_delta_mean`, `gen_kl_prefix_mean` | per-prompt values | percentile bootstrap of the mean |

**Wilson score interval** for `k` successes in `n` trials at confidence `c`, with `z` the two-sided
normal quantile:
`center = (p + z²/2n) / (1 + z²/n)` and
`half = z·sqrt(p(1−p)/n + z²/4n²) / (1 + z²/n)`.

**Bootstrap.** Resample the per-prompt values with replacement (`budget.bootstrap_resamples` times),
take the mean of each resample, and use the `(1 − c)/2` and `(1 + c)/2` percentiles. Seeds come
from `derive_seed(run.seed, "bootstrap", config_name, metric, slice)`. "Paired" in the sense of the
specification means resampling items, shared across the metrics of one item. It does not mean
pairing positions.

**Confidence.** With `budget.confidence = c`, the intervals are two-sided at level `c`. The gate
uses one bound, so each decision is a one-sided test at level `(1 + c) / 2`. With the default
`c = 0.95`, that is 97.5%.

## Gate (`day0/gate.py`)

Statuses are evaluated in this order. The first that applies is reported.

1. **`ERROR`.** The candidate raised during start, quality, or perf. The message is stored.
2. **`INSUFFICIENT_DATA`.** A covered slice has fewer items than `min_prompts_per_slice` for a gated
   metric, the candidate has no quality records, or the target-concurrency summary is missing or
   invalid.
3. **`FAIL_QUALITY`.** In any covered slice:
   - the upper bound of `flip_rate` is above `flip_rate_max`, or
   - the upper bound of `prefill_kl_mean` is above `kl_mean_max`, or
   - the lower bound of `prefill_top1_agreement` is below `top1_agreement_min`.
   Every failing (slice, metric, bound, threshold) is recorded.
4. **`FAIL_SLO`.** The p99 TTFT and the p99 TPOT at the target concurrency do not meet the SLO.
5. **`PASS`.**

The reference has quality `PASS` by definition, and its SLO is evaluated normally.

**Coverage per metric (decision D-17).** A slice is judged for a metric only on the items that have
data for that metric. A slice with zero such items is "not covered" for that metric. The flip rate
uses the gold items of the slice. Prefill metrics use the items whose prompts were short enough for
the teacher-forced path. Prompts above `prefill_max_prompt_tokens` are measured only on the
generation path.

**Small samples are conservative.** Zero flips in 60 gold items already fail at a 5% threshold,
because the Wilson upper bound is about 6%. Zero flips in 100 items passes. This is intended.

## Recommendation (§14.2)

Eligible: the reference if its SLO is met, and every candidate with status `PASS`. The highest
target throughput (`output_tok_per_s`) wins. Ties go to the lower p99 TTFT, then the lower cost. If
nothing is eligible, the recommendation is `None` with the reason
`no configuration meets the SLO at target concurrency`.

## Pareto front (§14.4)

Points are configurations with valid perf and quality data. The reference is at
`(x = 1, y = 0)`. A candidate is at `x = target throughput / reference target throughput`, with
`y` equal to its flip rate. Dominance uses the **upper Wilson bound** of the flip rate, so a
configuration with few gold items cannot look better than it is. A point is on the front when no
other point has `x ≥` and `y ≤` with at least one strict inequality. The chart plots the point
estimate with the Wilson interval as error bars.

## Noise floor and determinism (§12.5)

- **Noise floor (reference).** The first `num_prompts` items, in `item_id` order, are generated at
  two batch concurrencies. The report shows exact-match rate, prefix KL, and the flip rate with its
  Wilson interval, which is the reference's self-disagreement.
- **Determinism (candidate).** The same check at `concurrency_low` and `concurrency_high`. The
  identical-output rate is the exact-match rate.
- **Reference self-consistency.** The mean over prompts of the KL between the reference's decode
  and prefill distributions at the same positions. It is context for the prefill numbers.

## Performance metrics (§11.3)

All times are in milliseconds, measured with `time.perf_counter_ns()`.

- `TTFT = first content chunk - send`. `E2E = end - send`.
- `TPOT = (end - first chunk) / (output_tokens - 1)`. Undefined for one-token outputs.
- `ITL`: for consecutive content chunks, the gap divided by the tokens in the later chunk, repeated
  once per token. Chunk token counts come from `continuous_usage_stats`. Without it each chunk
  counts as one token and the level is flagged `chunk_token_counts_exact = false`.
- Per level: p50, p95, p99, and mean of TTFT, TPOT, ITL, and E2E (`numpy.percentile`, linear).
- `output_tok_per_s = sum(output_tokens) / window`, with `window` from the first measured request
  start to the last measured request end.
- `goodput_rps`: requests with TTFT ≤ `ttft_p99_ms` and TPOT ≤ `tpot_p99_ms`, divided by the window.
  A request with an undefined TPOT counts as meeting the TPOT bound (decision D-20).
- `cost_per_1M = gpu_hour_usd × num_gpus / (output_tok_per_s × 3600) × 1e6`. The unit test uses
  $3.00/h, 1 GPU, 2,500 tok/s, which gives $0.3333.
- `slo_met`: `p99 TTFT ≤ ttft_p99_ms` and `p99 TPOT ≤ tpot_p99_ms` and the level is valid.
- A level is **invalid** when its error rate exceeds `max_error_rate` or when no request succeeded.

Speculative decoding: `acceptance_rate = accepted / draft_tokens` and
`mean_acceptance_length = 1 + accepted / drafts`, from the counter deltas over the sweep. Absent or
zero counters give `None`.

## Reproducibility

Every random choice derives from `run.seed` through `derive_seed` (SHA-256 of the base and the
labels, truncated to 31 bits). Fingerprints cover the configuration that determines the reference
artifacts, so `--reuse-reference` cannot silently mix runs.
