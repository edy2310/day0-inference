# Design

day0 answers one question for an LLM serving platform: for this model, on this GPU, under this
workload and latency SLO, which serving configuration is the fastest that is still
quality-equivalent to the reference?

day0 is a **client and orchestrator**. It never imports vLLM. It launches `vllm serve` as a
subprocess and talks to it over HTTP, so the same code path measures the real server and the
built-in fake server.

## Data flow

```text
YAML config ──► load_config ──► Day0Config
                                  │
                                  ▼
                        run_experiment (orchestrator)
                                  │
     ┌────────────────────────────┼────────────────────────────────────┐
     ▼                            ▼                                    ▼
 REFERENCE phase              CANDIDATE phase (per configuration)   REPORT phase
 start server                 start server                          gate.evaluate
 tokenizer fingerprint        tokenizer check (must match)          gate.recommend
 tokenize suites              quality (generation + prefill)        report.card
 quality (generation+prefill) determinism (optional)                report.pareto
 noise floor                  perf sweep                            exit code
 perf workload + sweep        stop server                           
 write artifacts              write artifacts + status
 stop server
```

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | Strict pydantic models, cross-field validation, canonical hashes, path resolution |
| `seeds.py` | `derive_seed` (SHA-256 based) |
| `errors.py` | The error types the orchestrator distinguishes |
| `client.py` | Async HTTP client. Sends token-id prompts, asserts prompt length, times streams |
| `parsing.py` | SSE, logprobs (both key forms), prompt logprobs, Prometheus text |
| `suites.py` | Suite loading, answer extraction, slices, tokenizer probes and fingerprint |
| `backends/` | `vllm`, `fake`, `external`: start, health, stop. Process groups and cooldown |
| `quality/` | Reference and candidate runs, the estimator, flips, determinism |
| `perf/` | Deterministic workload, closed-loop load generator, latency metrics |
| `scrape.py` | Counter deltas, speculative-decoding summary, cache gauges |
| `env.py` | Environment capture with a hashed host name |
| `stats.py` | Wilson intervals, percentile bootstrap, percentiles (pure) |
| `gate.py` | Statuses, slice coverage, recommendation, Pareto front (pure) |
| `artifacts.py` | Atomic writes, deterministic gzip JSONL, record codecs, loaders |
| `report/` | Build card data from artifacts, render `card.md` and `card.json`, draw `pareto.png` |
| `fake/` | Deterministic toy model and the vLLM-shaped fake server |
| `orchestrator.py` | The run flow, error isolation, reuse, `check` |
| `cli.py` | `run`, `check`, `report`, `validate`, `fake-server`, `--version` |

## Key decisions (§6)

- **Token ids everywhere, `/v1/completions` only.** The reference tokenizes each suite item once
  with `/tokenize` and the chat template. Every later request sends token ids, so all
  configurations see byte-identical inputs. After each completion, day0 asserts that the server
  counted exactly the prompt length it was sent. This catches double BOS and similar template bugs.
- **Greedy decoding for quality.** `temperature=0`, `top_p=1`, `n=1`, no `top_k`.
- **KL direction and units.** `KL(P_ref || Q_cand)` in nats. The unit of analysis is the prompt.
- **Pre-registered thresholds.** Budgets and SLOs live in the YAML and are copied into the card with
  the configuration hash.
- **Reference reuse.** `--reuse-reference` copies a stored reference after its fingerprint matches
  the configuration. Otherwise it raises `ReferenceMismatchError`.

## Error isolation

A failing candidate is recorded as `ERROR` with its message. Its server is still stopped, and the
run continues (§5.2). A failing reference aborts the run with exit code 2, because every metric is
relative to it. The CLI maps any unexpected exception to exit code 2 as well.

## Artifacts

```text
RUN_DIR/
  run.json                 metadata: version, git sha, timestamps, config and fingerprint hashes,
                           command line, candidate order, statuses
  config.resolved.yaml     the configuration with absolute paths, so the run can be re-read
  env.json                 Python, platform, server version, GPUs, CUDA, hashed host name
  reference/
    server.log, status.json, fingerprint.json, tokenizer_fingerprint.txt
    quality.jsonl.gz       ReferenceRecord per item (prompt ids, outputs, top-k, prefill)
    noise_floor.json       reference against itself at two concurrencies
    perf_workload.jsonl.gz the prompts every configuration reuses
    perf.json              level summaries, target summary, speculative-decoding summary
    metrics_snapshots.json raw /metrics before and after each level
  candidates/<name>/
    server.log, status.json
    quality.jsonl.gz       CandidateRecord per item, with per-item metrics
    determinism.json       if enabled
    perf.json, metrics_snapshots.json
    summary.json           per-slice aggregates, gate result, errors (written by report)
  card.md, card.json, pareto.png
```

JSON is written with sorted keys and indentation. Gzip JSONL uses `mtime=0`, so the same records
always produce the same bytes. Every write goes to `*.tmp` first and is moved into place atomically.

## Reporting from artifacts

`day0 report RUN_DIR` reads only the run directory. `day0 run` calls the same code at the end.
Both produce the same `card.md` for the same artifacts. The end-to-end test checks this
byte-for-byte.

## Fake server

The fake server (`day0/fake/`) lets every layer run without a GPU:

- A deterministic toy model. Logits are a pure function of the last three tokens, the position,
  and the seeds. Random draws come from a counter-based generator, so a position gives the same
  logits alone or in a batch. The echo rule makes the answer verifiable. See `docs/decisions.md`
  (D-08, D-09, D-22) for the deviations from §13.1.
- A vLLM-shaped API: health (with startup delay), version, models, tokenize, completions (greedy,
  logprobs, prompt logprobs, SSE with continuous usage), and Prometheus metrics with optional
  speculative-decoding counters.
- Latency simulation: time to first token and time per token grow with prompt length and with the
  concurrency a request is served at, and are divided by `speedup`. The server never waits. Each
  content chunk carries its simulated time, and the perf runner times the fake on that virtual clock
  (D-39).

Its outcomes are calibrated so that §18.1 holds. They are a test fixture. They say nothing about
real models or hardware.
