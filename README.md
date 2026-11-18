# day0

Find the fastest LLM serving configuration that is still quality-equivalent to the reference.

<!-- RESULTS: fill after a real GPU run. Placeholder until results/published/llama31-8b-h100 exists. -->

<!-- RESULTS: copy the recommendation box from results/published/llama31-8b-h100/card.md once a
real run is published. Nothing in this section is a measurement yet. -->

## Quickstart with the fake server (no GPU)

Requires Python 3.11 or 3.12.

```bash
make install        # creates the editable install with dev tools into the active environment
make lint typecheck test
make fake-demo      # runs examples/fake.yaml and prints the card path
```

`make fake-demo` writes `results/fake-demo-<timestamp>/`, with `card.md`, `card.json`, and
`pareto.png`. The fake server simulates latency with a formula on a virtual clock, so its numbers are
the same on every machine. Its outcomes are fixed by construction to test the pipeline, so they are
not evidence about any real model.

Rebuild a card from its artifacts at any time:

```bash
day0 report results/fake-demo-<timestamp>
```

## Real-run quickstart

Follow [docs/runbook_gpu.md](docs/runbook_gpu.md). It covers the install script, the gated model,
suite building, preflight validation, the first run with fixtures, the full run, and publishing.

```bash
day0 validate examples/llama31-8b-h100.yaml --check-flags
day0 run examples/llama31-8b-h100.yaml
```

## How it works

day0 launches each serving configuration once and measures two things on the same server:

- **Performance.** A closed-loop load generator sweeps concurrency levels with a fixed,
  reference-tokenized workload. It reports TTFT, TPOT, ITL, throughput, goodput, and cost per
  million output tokens, with an SLO check at the target concurrency.
- **Quality equivalence.** Two paths are compared against the reference. The **prefill path**
  teacher-forces the reference output and compares full top-k distributions position by position
  (KL, top-1 agreement, log-likelihood delta). The **generation path** runs greedy decoding and
  reports flips, exact match, and the first divergence.

Results are pre-registered thresholds with statistics: Wilson intervals for proportions, percentile
bootstrap for means over prompts, and gates evaluated per context-length slice. The gate returns a
status (`PASS`, `FAIL_QUALITY`, `FAIL_SLO`, `INSUFFICIENT_DATA`, or `ERROR`), a recommendation, and
an exit code for CI. See [docs/design.md](docs/design.md) and [docs/metrics.md](docs/metrics.md).

## Related work

- **infer-check** runs differential correctness tests across backends, quantization, and
  concurrency. day0 differs by combining quality equivalence and performance in one decision, with
  statistical gates and a recommended configuration.
- **llama.cpp's KL-divergence mode** compares a quantized model with its full-precision reference on
  logits. day0 applies the same idea to serving configurations, and adds the flip and generation
  path that accuracy-free comparisons need.
- **`vllm bench serve`** measures serving performance for one configuration. day0 measures
  performance for several configurations and gates each one on quality.
- **lm-evaluation-harness** scores accuracy on benchmark tasks. day0 uses accuracy as context, and
  gates on flips and KL, because accuracy can stay flat while individual answers change.

## Limitations

See [docs/limitations.md](docs/limitations.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
