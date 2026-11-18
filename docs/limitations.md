# Limitations

day0 reports what it measured, under the conditions it measured. These limits apply to every
result.

## Measurement

- **The top-k KL is an approximation.** It sees the top-k of each distribution and merges the rest
  into a tail bucket. It is a lower bound with exact probabilities, but imputation for missing
  tokens can move it either way. The size of the error is measured by `scripts/validate_kl.py` and
  reported in [`kl_validation.md`](kl_validation.md). That validation has not been run yet.
- **The prefill path does not exercise speculative decoding.** Teacher-forced scoring runs a
  prefill, so a speculative-decoding candidate is only compared on its generation path there.
- **vLLM ignores the prefix cache for prompt-logprob requests.** Prefill numbers therefore do not
  depend on prefix caching. The generation path does.
- **Long prompts skip the prefill path.** Prompts longer than `prefill_max_prompt_tokens` are
  measured only on the generation path, because `prompt_logprobs` responses grow with
  prompt length times k. The card reports the coverage per slice.
- **Single GPU, single runtime, single model per run.** Results do not transfer across hardware,
  driver or CUDA versions, vLLM versions, or models.
- **Many comparisons, no multiplicity correction.** Configurations, slices, and metrics are all
  tested at the same confidence. With enough configurations, some pass or fail by chance.
- **Client-side load generation can become CPU-bound at high concurrency.** The load generator is
  a single Python process. At very high concurrency it can limit the throughput it measures.
- **Closed-loop load only.** Requests are sent back to back at a fixed concurrency. Open-loop
  arrival patterns and burstiness are not modelled.
- **Warmup is fixed.** `warmup_requests` per level is a fixed count, not a steady-state detector.

## Data

- **Dataset licences.** `gsm8k_200.jsonl` is sampled from the GSM8K test split (MIT licence). It is
  built from a file the user provides with `--gsm8k-jsonl`, and it is not committed by this
  repository's build (see [`runbook_gpu.md`](runbook_gpu.md)). `chat_open_100.jsonl` was written for
  this project, and no text was copied from another dataset or website.
- **The long-context suite is synthetic.** `longctx_synth.jsonl` uses filler text from a built-in
  word list with embedded facts. It measures retrieval and summation over that filler. It is not
  representative of real long documents.
- **Token counts are estimates at build time.** The suite builder assumes 4 characters per token.
  Slices use the tokenizer's real counts at run time.
- **Small suites.** 200, 100, and 100 items give wide intervals on the per-slice numbers. The
  `min_prompts_per_slice` rule prevents a gate decision on too few items, but it does not make a
  small suite large.

## Fake server

- **The fake server measures nothing about real models.** Its logits come from a toy function of
  the last three tokens. Its latencies come from a formula. Its outcomes in the demo are fixed by
  construction and exist to test the pipeline.
- **Its prompt logprobs are slow to serialize.** Each position carries k+1 entries, so the demo uses
  `top_k_logprobs: 5`. Real runs use 20.

## Scope

- **No multi-node or tensor-parallel orchestration.** `--tensor-parallel-size` can be passed as a raw
  argument, and nothing more is managed.
- **No chat completions endpoint in the measurement path.** Chat templates vary between server
  versions, so every prompt is tokenized once on the reference and sent as token ids.
- **No automatic search.** Candidates are listed explicitly in the configuration.
- **No SGLang or TensorRT-LLM launchers.** Use the `external` backend for those.
- **Real GPU results are not in this repository yet.** Every number in the README and in these
  docs is a placeholder until a real run is published under `results/published/`.
