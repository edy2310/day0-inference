"""Measure how far the top-k KL estimator is from the exact full-vocabulary KL (§16).

This script is offline and CPU-only. It is not run in CI. It needs the ``validate`` extra
(``torch`` and ``transformers``), and it downloads the model named by ``--model``. It writes
``docs/kl_validation.md``.

For each prompt it generates greedily in float32, then runs teacher-forced passes with:
  (a) float32 (the reference),
  (b) bfloat16,
  (c) float32 with Gaussian weight noise, sigma = 1e-3 times the standard deviation of each tensor.

At every position it computes the exact KL from full logits and the top-k estimate for k in
{5, 10, 20}. The document reports correlations and error statistics per perturbation and k.

Usage:
    python scripts/validate_kl.py [--model NAME] [--output docs/kl_validation.md]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from day0.quality.metrics import topk_kl
from day0.suites import load_suite

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "kl_validation.md"
DEFAULT_SUITE = REPO_ROOT / "suites" / "chat_open_100.jsonl"
N_PROMPTS = 50
N_TOKENS = 64
KS = (5, 10, 20)
SIGMA_REL = 1e-3
SEED = 1234


def _exact_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Exact KL divergence between two full distributions, in nats.

    Args:
        p: Reference probabilities.
        q: Candidate probabilities.

    Returns:
        ``sum p * log(p / q)``.
    """
    return float(np.sum(p * (np.log(p) - np.log(q))))


def _topk(probs: np.ndarray, k: int) -> dict[int, float]:
    """Top-k of a probability vector as token id to natural-log probability.

    Args:
        probs: Probabilities over the vocabulary.
        k: Number of tokens to keep.

    Returns:
        The top-k dictionary.
    """
    order = np.argsort(-probs, kind="stable")[:k]
    return {int(token): float(math.log(probs[token])) for token in order}


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation.

    Args:
        x: First sample.
        y: Second sample.

    Returns:
        The correlation, or NaN when a sample has no variance.
    """
    if np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, without ties correction.

    Args:
        x: First sample.
        y: Second sample.

    Returns:
        The correlation of the ranks.
    """
    rank_x = np.argsort(np.argsort(x)).astype(np.float64)
    rank_y = np.argsort(np.argsort(y)).astype(np.float64)
    return _pearson(rank_x, rank_y)


def _load(model_name: str, dtype: Any) -> tuple[Any, Any]:
    """Load a tokenizer and a causal language model.

    Args:
        model_name: Hugging Face model id or local path.
        dtype: Torch dtype for the weights.

    Returns:
        ``(tokenizer, model)`` in evaluation mode.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.eval()
    return tokenizer, model


def _prompt_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    """Tokenize chat messages with the generation prompt.

    Args:
        tokenizer: Tokenizer with a chat template.
        messages: Chat messages.

    Returns:
        Token ids.
    """
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    return [int(t) for t in ids]


def _probs(model: Any, ids: list[int], start: int) -> np.ndarray:
    """Next-token distributions for the generated positions of one sequence.

    Args:
        model: Causal language model.
        ids: Prompt plus generated tokens.
        start: Index of the first generated token.

    Returns:
        Array of shape ``(n_generated, vocab)`` in float64.
    """
    import torch

    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0].to(torch.float64)
    # The distribution that predicts token i is the row at position i - 1.
    rows = logits[start - 1 : len(ids) - 1]
    return torch.softmax(rows, dim=-1).numpy()


def main() -> None:
    """Run the validation and write the document."""
    import torch

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    tokenizer, ref_model = _load(args.model, torch.float32)
    _, bf16_model = _load(args.model, torch.bfloat16)
    _, noisy_model = _load(args.model, torch.float32)
    generator = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for param in noisy_model.parameters():
            sigma = SIGMA_REL * float(param.std())
            param.add_(torch.randn(param.shape, generator=generator) * sigma)

    items = load_suite(args.suite)[:N_PROMPTS]
    results: dict[str, dict[int, list[tuple[float, float]]]] = {
        "bfloat16": {k: [] for k in KS},
        "fp32 + weight noise": {k: [] for k in KS},
    }
    for item in items:
        ids = _prompt_ids(tokenizer, item.messages)
        with torch.no_grad():
            generated = ref_model.generate(
                torch.tensor([ids]), max_new_tokens=N_TOKENS, do_sample=False
            )[0].tolist()
        full = generated
        start = len(ids)
        if len(full) <= start:
            continue
        ref_probs = _probs(ref_model, full, start)
        for name, model in (("bfloat16", bf16_model), ("fp32 + weight noise", noisy_model)):
            cand_probs = _probs(model, full, start)
            for k in KS:
                exact: list[float] = []
                estimate: list[float] = []
                for p_row, q_row in zip(ref_probs, cand_probs, strict=True):
                    exact.append(_exact_kl(p_row, q_row))
                    estimate.append(topk_kl(_topk(p_row, k), _topk(q_row, k)))
                results[name][k].append((float(np.mean(exact)), float(np.mean(estimate))))

    lines = _render(args, results, len(items))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


def _render(
    args: argparse.Namespace, results: dict[str, dict[int, list[tuple[float, float]]]], n_items: int
) -> list[str]:
    """Render the validation document.

    Args:
        args: Parsed arguments.
        results: Per-prompt (exact, estimate) means, by perturbation and k.
        n_items: Number of prompts that were available.

    Returns:
        Lines of Markdown.
    """
    import torch
    import transformers

    lines = [
        "# Top-k KL estimator validation",
        "",
        f"- Generated by `scripts/validate_kl.py` on {dt.datetime.now(dt.UTC).date().isoformat()}.",
        f"- Command: `python scripts/validate_kl.py --model {args.model}`",
        f"- Model: `{args.model}`; prompts: {min(n_items, N_PROMPTS)} from `{args.suite.name}`; "
        f"{N_TOKENS} greedy tokens each.",
        f"- Versions: Python {platform.python_version()}, torch {torch.__version__}, "
        f"transformers {transformers.__version__}.",
        "",
        "Per-prompt means of the exact KL and of the top-k estimate. Columns: Pearson and Spearman "
        "correlation between the estimate and the exact KL, mean absolute error, median relative "
        "error, and the fraction of prompts where the estimate is lower than the exact value.",
        "",
    ]
    for perturbation, by_k in results.items():
        lines += [
            f"## {perturbation}",
            "",
            "| k | Pearson | Spearman | MAE (nats) | Median rel. error "
            "| Fraction estimate < exact |",
            "|---|---|---|---|---|---|",
        ]
        for k in KS:
            pairs = np.asarray(by_k[k], dtype=np.float64)
            if pairs.size == 0:
                lines.append(f"| {k} | n/a | n/a | n/a | n/a | n/a |")
                continue
            exact, estimate = pairs[:, 0], pairs[:, 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(estimate - exact) / np.where(exact > 0, exact, np.nan)
            lines.append(
                f"| {k} | {_pearson(exact, estimate):.4f} | {_spearman(exact, estimate):.4f} | "
                f"{float(np.mean(np.abs(estimate - exact))):.3e} | "
                f"{float(np.nanmedian(rel)):.3f} | {float(np.mean(estimate < exact)):.2f} |"
            )
        lines.append("")
    lines.append(
        "Raw per-prompt values: "
        + json.dumps(
            {name: {str(k): len(v) for k, v in by_k.items()} for name, by_k in results.items()}
        )
    )
    return lines


if __name__ == "__main__":
    main()
