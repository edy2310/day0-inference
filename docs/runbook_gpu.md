# GPU runbook: a real Llama-3.1-8B profile on one H100

This runbook produces a serving profile card for `meta-llama/Llama-3.1-8B-Instruct` on one H100
80GB. Every command is run from the repository root. Each step says what it produces.

No numbers from a real run are in this repository yet. Timings below are estimates to plan with,
not results.

## 1. Rent the hardware

Rent one H100 80GB whose host has an NVIDIA driver of 580 or newer. That driver supports CUDA 13.0,
which the pinned vLLM needs (`docs/decisions.md`, D-34). `nvidia-smi` prints the driver version. The
setup script stops on an older driver. Record the price per GPU-hour in
`examples/llama31-8b-h100.yaml` under `hardware.gpu_hour_usd`. The default `3.00` is a placeholder.

## 2. Install

```bash
bash scripts/gpu_setup.sh
source .venv/bin/activate
```

The script stops at the first error and prints each step. It checks the driver (580 or newer). Then
it installs `uv`, creates `.venv` with Python 3.11, and installs vLLM 0.34.0 and day0 with the `dev`
and `suites` extras in one resolve. Last, it checks `nvidia-smi`, `vllm --version`, `day0 --version`,
the `datasets` import, and that `vllm serve --help=all` lists the flags day0 uses.

To pin another vLLM version, set `VLLM_VERSION`:

```bash
VLLM_VERSION=<version> bash scripts/gpu_setup.sh
```

Keep the environment active for the rest of the steps. `make` and `pytest` use the `python`,
`pytest`, and `vllm` on `PATH`. Without the activation, the GPU tests skip themselves.

## 3. Hugging Face access

`meta-llama/Llama-3.1-8B-Instruct` is gated. Accept the licence on its model page, then export a
token for this shell only. Do not commit the token.

```bash
export HF_TOKEN=<your token>
```

Ungated alternative for a dry run: `Qwen/Qwen2.5-7B-Instruct`. Change `model.id` and the
`--model` path accordingly.

## 4. Build the suites

```bash
python scripts/build_suites.py --gsm8k-from-hub
sha256sum suites/*.jsonl
```

This writes all three suites. `gsm8k_200.jsonl` is 200 items sampled with a fixed seed from the
official GSM8K test split (MIT licence). The `datasets` library downloads the split from the Hugging
Face hub. Record the checksums in `docs/decisions.md`, and commit the three suites, which the specification
asks for. If the hub is not reachable, download the
split's `test.jsonl` from the dataset page and build from that file instead:

```bash
python scripts/build_suites.py --gsm8k-jsonl /path/to/test.jsonl
```

## 5. Preflight

```bash
day0 validate examples/llama31-8b-h100.yaml --check-flags
```

This prints the exact `vllm serve` command for each configuration. With `--check-flags`, it warns
about every flag that `vllm serve --help=all` does not list. Warnings never fail the command.
Fix them before the run, because flags change between vLLM versions.

Also fix these before the run, and record the decisions in `docs/decisions.md`:

- `hardware.gpu_hour_usd`: the real price.
- `budget`: the thresholds. They are starting points.
- `slo`: the product's latency targets. They are starting points.

## 6. First run with fixtures

```bash
day0 run examples/llama31-8b-h100.yaml --only fp8-weights --capture-fixtures
```

`--capture-fixtures` writes the first raw response of each request kind to `RUN_DIR/fixtures/`. The
reference's captures are in `RUN_DIR/fixtures/reference/`. Copy only those into
`tests/fixtures/vllm/`, then run the tests:

```bash
cp RUN_DIR/fixtures/reference/*.json tests/fixtures/vllm/
make test
DAY0_GPU_TESTS=1 pytest -q -rs tests/gpu
```

Expect 6 passed for the GPU tests. `-rs` prints the reason for any skip. The parser tests load these captures when
they are present, and they skip when they are absent. A failure in the fixture test means the
captured file does not parse the way the tests assume. Keep the output and record it in
`docs/decisions.md`, together with what the smoke test showed: the flags accepted, the logprobs
format, and the usage counts.

## 7. Full run

```bash
tmux new -s day0
day0 run examples/llama31-8b-h100.yaml
```

`tmux` keeps the run alive if the connection drops. The specification's estimate is 3 to 5 hours for
eight configurations, and this is an estimate, not a measurement. The run writes its progress to the
terminal. Its outputs go to `results/llama31-8b-h100-<timestamp>/`.

Exit codes: `0` means a configuration is recommended, `1` means none is, and `2` means a
configuration or reference error.

## 8. Measure the top-k KL error

`scripts/validate_kl.py` measures how far the top-k estimate is from the exact KL. It needs `torch`
and `transformers` 5, which stay out of `.venv`, because torch can conflict with vLLM's pins. Use a
second environment:

```bash
uv venv --python 3.11 /workspace/kl-venv
uv pip install -p /workspace/kl-venv/bin/python -e ".[validate]"
/workspace/kl-venv/bin/python scripts/validate_kl.py
```

The script writes `docs/kl_validation.md`. It uses CUDA when a GPU is visible. Its default model is
`Qwen/Qwen2.5-0.5B-Instruct`, because it holds three copies of the model at once, and an 8B model
does not fit one 80 GB GPU that way. Commit the document with the published run.

## 9. Publish

```bash
RUN=results/llama31-8b-h100-<timestamp>
mkdir -p results/published/llama31-8b-h100
rsync -a --exclude 'server.log' "$RUN/" results/published/llama31-8b-h100/
find results/published -name server.log -size +5M -print
```

Server logs larger than 5 MB are excluded. The `find` line lists any that are still there.

Then check the card, commit, and shut the instance down:

```bash
cat results/published/llama31-8b-h100/card.md
git add results/published/llama31-8b-h100 && git commit -m "results: llama31-8b-h100 profile"
```

## 10. Rebuild the card without a GPU

Anyone can rebuild the card from the published artifacts:

```bash
day0 report results/published/llama31-8b-h100
```

The output is byte-identical to the card that was published.

## Check a new runtime against a stored reference

When the serving runtime changes, run only the affected candidate against the stored reference:

```bash
day0 check examples/llama31-8b-h100.yaml --candidate fp8-weights \
  --reference results/published/llama31-8b-h100/reference
```

Exit code `0` means the candidate passes, `1` means it fails a gate, and `2` means an error. This
is the CI gate for runtime upgrades. It needs the same configuration and the same suites, or it
refuses to run.

## Troubleshooting

- **The reference fails to start.** Read `reference/server.log`. The last 50 lines are also in the
  error message.
- **The reference returns no token ids or top-k logprobs.** Check that `--logprobs-mode
  raw_logprobs` is in the command. `day0 validate --check-flags` shows the command.
- **`port 8100 is already in use`.** Another server is still running. Stop it, then retry.
- **GPU memory stays in use between configurations.** The backend waits up to 60 s for
  `nvidia-smi` to report under 2 GiB, after the `cooldown_s` pause. Increase `cooldown_s` if needed.
- **The setup script stops with "driver ... is older than 580".** Use a host with a newer driver.
  `DAY0_SKIP_DRIVER_CHECK=1` runs the install anyway, and vLLM may then fail to start.
- **The GPU tests are skipped.** `vllm` is not on `PATH`. Activate `.venv` (step 2), then run
  `pytest -rs` again to see the reason.
