# Serving Profile Card — fake-model on fake (fake-demo)

> **Recommendation:** `fast-ok`
> Throughput 1.13× of reference · p99 TTFT 30.4 ms · p99 TPOT 3.8 ms · $0.3932 per 1M output tokens
> Flip rate 0.00% [0.00%, 2.50%] · prefill KL 7.3e-05 [7.2e-05, 7.5e-05] nats

## Main table

| Config | Status | ×throughput | Output tok/s | p99 TTFT (ms) | p99 TPOT (ms) | Max conc. meeting SLO | $/1M tokens | Flip rate [CI] | Exact match | Prefill KL [CI] | Top-1 [CI] | Gen KL (prefix) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ref` (reference) | PASS | 1.00× | 1882.92 | 36.1 | 3.5 | 16 | $0.4426 | baseline | 100% | baseline | baseline | baseline |
| `identical` | PASS | 0.98× | 1842.19 | 35.0 | 3.5 | 16 | $0.4524 | 0.00% [0.00%, 2.50%] | 100.00% | 2.2e-13 [2.1e-13, 2.2e-13] | 100.00% [100.00%, 100.00%] | 2.2e-13 |
| `fast-ok` | PASS | 1.13× | 2119.32 | 30.4 | 3.8 | 8 | $0.3932 | 0.00% [0.00%, 2.50%] | 100.00% | 7.3e-05 [7.2e-05, 7.5e-05] | 100.00% [100.00%, 100.00%] | 7.3e-05 |
| `fast-noisy` | FAIL_QUALITY | 1.26× | 2363.17 | 27.4 | 3.6 | 16 | $0.3526 | 40.67% [33.13%, 48.67%] | 0.00% | 1.8e-01 [1.6e-01, 2.1e-01] | 91.51% [90.87%, 92.10%] | 1.2e-01 |
| `longctx-bad` | FAIL_QUALITY | 1.05× | 1984.33 | 31.4 | 3.7 | 8 | $0.4200 | 22.67% [16.70%, 30.00%] | 73.33% | 4.4e-01 [3.0e-01, 6.1e-01] | 83.43% [78.61%, 88.18%] | 1.2e-02 |
| `slow` | FAIL_SLO | 0.37× | 701.39 | 92.1 | 9.0 | n/a | $1.1881 | 0.00% [0.00%, 2.50%] | 100.00% | 2.2e-13 [2.1e-13, 2.2e-13] | 100.00% [100.00%, 100.00%] | 2.2e-13 |
| `spec-fake` | PASS | 1.01× | 1905.49 | 33.8 | 3.7 | 16 | $0.4373 | 0.00% [0.00%, 2.50%] | 100.00% | 2.2e-13 [2.1e-13, 2.2e-13] | 100.00% [100.00%, 100.00%] | 2.2e-13 |

## Per-slice results

### `identical`

| Slice | n | Flip rate [CI] | Prefill KL [CI] | Top-1 [CI] | Checks |
|---|---|---|---|---|---|
| [0, 512) | 50 | 0.00% [0.00%, 7.13%] | 2.2e-13 [2.1e-13, 2.3e-13] | 100.00% [100.00%, 100.00%] | flip: pass; KL: pass; top-1: pass |
| [512, 2048) | 50 | 0.00% [0.00%, 7.13%] | 2.2e-13 [2.1e-13, 2.3e-13] | 100.00% [100.00%, 100.00%] | flip: pass; KL: pass; top-1: pass |
| [2048, 65536) | 50 | 0.00% [0.00%, 7.13%] | 2.1e-13 [2.1e-13, 2.2e-13] | 100.00% [100.00%, 100.00%] | flip: pass; KL: pass; top-1: pass |

### `fast-ok`

| Slice | n | Flip rate [CI] | Prefill KL [CI] | Top-1 [CI] | Checks |
|---|---|---|---|---|---|
| [0, 512) | 50 | 0.00% [0.00%, 7.13%] | 7.3e-05 [7.1e-05, 7.6e-05] | 100.00% [100.00%, 100.00%] | flip: pass; KL: pass; top-1: pass |
| [512, 2048) | 50 | 0.00% [0.00%, 7.13%] | 7.4e-05 [7.1e-05, 7.7e-05] | 100.00% [100.00%, 100.00%] | flip: pass; KL: pass; top-1: pass |
| [2048, 65536) | 50 | 0.00% [0.00%, 7.13%] | 7.3e-05 [7.1e-05, 7.5e-05] | 100.00% [100.00%, 100.00%] | flip: pass; KL: pass; top-1: pass |

### `fast-noisy`

| Slice | n | Flip rate [CI] | Prefill KL [CI] | Top-1 [CI] | Checks |
|---|---|---|---|---|---|
| [0, 512) | 50 | 50.00% [36.64%, 63.36%] | 2.0e-01 [1.7e-01, 2.4e-01] | 91.16% [90.26%, 91.95%] | flip: fail; KL: fail; top-1: fail |
| [512, 2048) | 50 | 32.00% [20.76%, 45.81%] | 1.9e-01 [1.5e-01, 2.4e-01] | 91.54% [90.31%, 92.64%] | flip: fail; KL: fail; top-1: fail |
| [2048, 65536) | 50 | 40.00% [27.61%, 53.82%] | 1.5e-01 [1.4e-01, 1.8e-01] | 91.82% [90.93%, 92.60%] | flip: fail; KL: fail; top-1: fail |

### `longctx-bad`

| Slice | n | Flip rate [CI] | Prefill KL [CI] | Top-1 [CI] | Checks |
|---|---|---|---|---|---|
| [0, 512) | 50 | 0.00% [0.00%, 7.13%] | 1.8e-05 [1.8e-05, 1.9e-05] | 100.00% [100.00%, 100.00%] | flip: pass; KL: pass; top-1: pass |
| [512, 2048) | 50 | 0.00% [0.00%, 7.13%] | 1.8e-05 [1.8e-05, 1.9e-05] | 100.00% [100.00%, 100.00%] | flip: pass; KL: pass; top-1: pass |
| [2048, 65536) | 50 | 68.00% [54.19%, 79.24%] | 1.3e+00 [9.7e-01, 1.7e+00] | 50.30% [40.58%, 59.42%] | flip: fail; KL: fail; top-1: fail |

### `slow`

| Slice | n | Flip rate [CI] | Prefill KL [CI] | Top-1 [CI] | Checks |
|---|---|---|---|---|---|
| [0, 512) | 50 | 0.00% [0.00%, 7.13%] | 2.2e-13 [2.1e-13, 2.3e-13] | 100.00% [100.00%, 100.00%] | flip: pass; KL: pass; top-1: pass |
| [512, 2048) | 50 | 0.00% [0.00%, 7.13%] | 2.2e-13 [2.1e-13, 2.3e-13] | 100.00% [100.00%, 100.00%] | flip: pass; KL: pass; top-1: pass |
| [2048, 65536) | 50 | 0.00% [0.00%, 7.13%] | 2.1e-13 [2.1e-13, 2.2e-13] | 100.00% [100.00%, 100.00%] | flip: pass; KL: pass; top-1: pass |

### `spec-fake`

| Slice | n | Flip rate [CI] | Prefill KL [CI] | Top-1 [CI] | Checks |
|---|---|---|---|---|---|
| [0, 512) | 50 | 0.00% [0.00%, 7.13%] | 2.2e-13 [2.1e-13, 2.3e-13] | 100.00% [100.00%, 100.00%] | flip: pass; KL: pass; top-1: pass |
| [512, 2048) | 50 | 0.00% [0.00%, 7.13%] | 2.2e-13 [2.1e-13, 2.3e-13] | 100.00% [100.00%, 100.00%] | flip: pass; KL: pass; top-1: pass |
| [2048, 65536) | 50 | 0.00% [0.00%, 7.13%] | 2.1e-13 [2.1e-13, 2.2e-13] | 100.00% [100.00%, 100.00%] | flip: pass; KL: pass; top-1: pass |


## Gate failures

- `fast-noisy`: slice [0, 512), flip_rate upper bound 0.6336 vs threshold 0.1
- `fast-noisy`: slice [0, 512), prefill_kl_mean upper bound 0.2373 vs threshold 0.01
- `fast-noisy`: slice [0, 512), prefill_top1_agreement lower bound 0.9026 vs threshold 0.95
- `fast-noisy`: slice [512, 2048), flip_rate upper bound 0.4581 vs threshold 0.1
- `fast-noisy`: slice [512, 2048), prefill_kl_mean upper bound 0.2441 vs threshold 0.01
- `fast-noisy`: slice [512, 2048), prefill_top1_agreement lower bound 0.9031 vs threshold 0.95
- `fast-noisy`: slice [2048, 65536), flip_rate upper bound 0.5382 vs threshold 0.1
- `fast-noisy`: slice [2048, 65536), prefill_kl_mean upper bound 0.1771 vs threshold 0.01
- `fast-noisy`: slice [2048, 65536), prefill_top1_agreement lower bound 0.9093 vs threshold 0.95
- `longctx-bad`: slice [2048, 65536), flip_rate upper bound 0.7924 vs threshold 0.1
- `longctx-bad`: slice [2048, 65536), prefill_kl_mean upper bound 1.734 vs threshold 0.01
- `longctx-bad`: slice [2048, 65536), prefill_top1_agreement lower bound 0.4058 vs threshold 0.95

## Speculative decoding

- `spec-fake`: acceptance rate 60.00%, mean acceptance length 3.40

## Noise floor and determinism

- Reference against itself (concurrency 1 vs 8, 20 prompts): exact match 100.00%, prefix KL 0.0e+00, flip rate 0.00% [0.00%, 16.11%].
- `identical` determinism (concurrency 1 vs 8): identical-output rate 100.00%.
- `fast-ok` determinism (concurrency 1 vs 8): identical-output rate 100.00%.
- `fast-noisy` determinism (concurrency 1 vs 8): identical-output rate 100.00%.
- `longctx-bad` determinism (concurrency 1 vs 8): identical-output rate 100.00%.
- `slow` determinism (concurrency 1 vs 8): identical-output rate 100.00%.
- `spec-fake` determinism (concurrency 1 vs 8): identical-output rate 100.00%.
- Reference prefill-vs-decode self-consistency KL (mean per prompt): 0.0e+00 nats.

## Budget and SLO

- Flip rate upper bound must stay at or below 10.00%.
- Prefill KL mean upper bound must stay at or below 1.0e-02 nats.
- Prefill top-1 agreement lower bound must stay at or above 95.00%.
- At least 30 items per slice for each gated metric; 95% two-sided intervals, 500 bootstrap resamples.
- SLO at target concurrency 8: p99 TTFT ≤ 120.0 ms and p99 TPOT ≤ 6.0 ms.
- Thresholds were fixed in the configuration before the run (config hash `914d6f59e0cce279a6099c94565290c7df62feae61e7b2f097076d5b66d0dc27`).

## Environment

- GPU: fake
- Driver: n/a; CUDA: n/a
- Server: fake-0.1 (backend fake)
- Timing: simulated on the fake server's virtual clock, not measured on hardware
- day0 0.1.0; git `73310bf011e16bde8b6ec2af0318b9d1f6db39c9`
- Started 2026-11-18T06:23:02Z; ended None
- `ref`: 20.4 s wall time, 0.006 GPU-hours
- `identical`: 18.2 s wall time, 0.005 GPU-hours
- `fast-ok`: 18.5 s wall time, 0.005 GPU-hours
- `fast-noisy`: 18.8 s wall time, 0.005 GPU-hours
- `longctx-bad`: 18.9 s wall time, 0.005 GPU-hours
- `slow`: 32.9 s wall time, 0.009 GPU-hours
- `spec-fake`: 17.6 s wall time, 0.005 GPU-hours

## Reproduce

```bash
day0 run examples/fake.yaml
```

## Limitations

- Top-k KL is an approximation; see `docs/kl_validation.md`.
- The prefill path does not exercise speculative decoding, and vLLM ignores the prefix cache for prompt-logprob requests.
- Prompts above `prefill_max_prompt_tokens` are measured only on the generation path.
- Single GPU, single runtime, single model per run; results do not transfer across hardware.
- Many comparisons (configurations × slices × metrics) with no multiplicity correction.
- Client-side load generation can become CPU-bound at high concurrency.
- Dataset licences and the synthetic nature of the long-context suite; see `docs/limitations.md`.

See `docs/limitations.md` for the full list.
