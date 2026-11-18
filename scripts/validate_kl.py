"""Measure how far the top-k KL estimator is from the exact full-vocabulary KL (§16).

This script is offline and is not run in CI. It needs the ``validate`` extra (``torch`` and
``transformers`` 5.x), and it downloads the model named by ``--model``. It runs on the first GPU
when one is visible and on the CPU otherwise. It writes ``docs/kl_validation.md``.

For each prompt it generates ``N_TOKENS`` tokens greedily (no repetition penalty) in float32. That
sequence is the reference. Teacher-forced passes over the same sequence then use:
  (a) bfloat16 weights,
  (b) float32 weights with Gaussian noise, sigma = 1e-3 times the standard deviation of each tensor.

At every generated position it computes the exact KL from the full distributions and the top-k
estimate for k in {5, 10, 20}. The document reports correlations and error statistics, per
perturbation and k, over the per-prompt means.

The default model is small on purpose. The script holds three copies of the model at once, so an
8B model needs about 80 GB for the weights alone.

Usage:
    python scripts/validate_kl.py [--model NAME] [--device cuda|cpu] [--output PATH]
"""

from __future__ import annotations

import argparse
import datetime as dt
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
BF16 = "bfloat16"
NOISY = "fp32 + weight noise"


def _exact_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Exact KL divergence between two full distributions, in nats.

    Tokens with ``p == 0`` contribute nothing, by the convention ``0 log 0 = 0``.

    Args:
        p: Reference probabilities.
        q: Candidate probabilities.

    Returns:
        ``sum p * log(p / q)``.
    """
    support = p > 0
    return float(np.sum(p[support] * (np.log(p[support]) - np.log(q[support]))))


def _ranked(probs: np.ndarray, depth: int) -> np.ndarray:
    """Token ids of the ``depth`` most probable tokens, most probable first.

    A partial selection keeps the cost linear in the vocabulary size. Ties at the cut-off can be
    resolved differently than by a full sort, but the same input always gives the same order.

    Args:
        probs: Probabilities over the vocabulary.
        depth: Number of tokens to return. Must not exceed the vocabulary size.

    Returns:
        Token ids, of length ``depth``.
    """
    top = np.argpartition(-probs, depth - 1)[:depth]
    return top[np.argsort(-probs[top], kind="stable")]


def _topk(probs: np.ndarray, order: np.ndarray, k: int) -> dict[int, float]:
    """Top-k of a probability vector as token id to natural-log probability.

    Args:
        probs: Probabilities over the vocabulary.
        order: Token ids from ``_ranked``, most probable first, with at least ``k`` entries.
        k: Number of tokens to keep.

    Returns:
        The top-k dictionary.
    """
    return {int(token): math.log(float(probs[token])) for token in order[:k]}


def _prompt_means(ref_probs: np.ndarray, cand_probs: np.ndarray) -> dict[int, tuple[float, float]]:
    """Per-prompt means of the exact KL and of the top-k estimate, for each k in ``KS``.

    Args:
        ref_probs: Reference distributions, one row per generated position.
        cand_probs: Candidate distributions with the same shape.

    Returns:
        ``{k: (mean exact KL, mean top-k estimate)}``, in nats.
    """
    depth = max(KS)
    exact: list[float] = []
    estimate: dict[int, list[float]] = {k: [] for k in KS}
    for p_row, q_row in zip(ref_probs, cand_probs, strict=True):
        p_order = _ranked(p_row, depth)
        q_order = _ranked(q_row, depth)
        exact.append(_exact_kl(p_row, q_row))
        for k in KS:
            estimate[k].append(topk_kl(_topk(p_row, p_order, k), _topk(q_row, q_order, k)))
    return {k: (float(np.mean(exact)), float(np.mean(estimate[k]))) for k in KS}


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


def _load(model_name: str, dtype: Any, device: Any) -> tuple[Any, Any]:
    """Load a tokenizer and a causal language model on one device.

    Args:
        model_name: Hugging Face model id or local path.
        dtype: Torch dtype for the weights.
        device: Torch device to place the model on.

    Returns:
        ``(tokenizer, model)`` with the model in evaluation mode.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


def _prompt_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    """Tokenize chat messages with the generation prompt.

    ``return_dict=False`` is passed explicitly. In transformers 5 the tokenizer returns a
    dictionary by default, and the token ids would then be the dictionary's keys.

    Args:
        tokenizer: Tokenizer with a chat template.
        messages: Chat messages.

    Returns:
        Token ids.
    """
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=False
    )
    if ids and isinstance(ids[0], list):  # a batch of one conversation
        ids = ids[0]
    return [int(token) for token in ids]


def _probs(model: Any, ids: list[int], start: int, device: Any) -> np.ndarray:
    """Next-token distributions for the generated positions of one sequence.

    Args:
        model: Causal language model on ``device``.
        ids: Prompt plus generated tokens.
        start: Index of the first generated token.
        device: Device of the model.

    Returns:
        Array of shape ``(n_generated, vocab)`` in float64.
    """
    import torch

    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device)).logits[0].to(torch.float64)
        # The distribution that predicts token i is the row at position i - 1.
        rows = logits[start - 1 : len(ids) - 1]
        return torch.softmax(rows, dim=-1).cpu().numpy()


def main() -> None:
    """Run the validation and write the document."""
    import torch

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="default: cuda when a GPU is visible, otherwise cpu",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(SEED)
    tokenizer, ref_model = _load(args.model, torch.float32, device)
    _, bf16_model = _load(args.model, torch.bfloat16, device)
    _, noisy_model = _load(args.model, torch.float32, device)
    # Noise is drawn on the CPU so that the same seed gives the same weights on any device.
    generator = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for param in noisy_model.parameters():
            sigma = SIGMA_REL * float(param.std())
            noise = torch.randn(param.shape, generator=generator) * sigma
            param.add_(noise.to(device=param.device, dtype=param.dtype))

    items = load_suite(args.suite)[:N_PROMPTS]
    results: dict[str, dict[int, list[tuple[float, float]]]] = {
        BF16: {k: [] for k in KS},
        NOISY: {k: [] for k in KS},
    }
    for item in items:
        ids = _prompt_ids(tokenizer, item.messages)
        start = len(ids)
        with torch.no_grad():
            input_ids = torch.tensor([ids], device=device)
            generated = ref_model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=N_TOKENS,
                do_sample=False,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )[0].tolist()
        if len(generated) <= start:
            continue
        ref_probs = _probs(ref_model, generated, start, device)
        for name, model in ((BF16, bf16_model), (NOISY, noisy_model)):
            means = _prompt_means(ref_probs, _probs(model, generated, start, device))
            for k, pair in means.items():
                results[name][k].append(pair)

    lines = _render(args, results, len(items), device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


def _render(
    args: argparse.Namespace,
    results: dict[str, dict[int, list[tuple[float, float]]]],
    n_items: int,
    device: Any,
) -> list[str]:
    """Render the validation document.

    Args:
        args: Parsed arguments.
        results: Per-prompt (exact, estimate) means, by perturbation and k.
        n_items: Number of prompts that were tried.
        device: Device the models ran on.

    Returns:
        Lines of Markdown.
    """
    import torch
    import transformers

    n_used = len(results[BF16][KS[0]])
    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    lines = [
        "# Top-k KL estimator validation",
        "",
        f"- Generated by `scripts/validate_kl.py` on {dt.datetime.now(dt.UTC).date().isoformat()}.",
        f"- Command: `python scripts/validate_kl.py --model {args.model}`",
        f"- Model: `{args.model}`; prompts with generated tokens: {n_used} of {n_items} tried "
        f"from `{args.suite.name}`; {N_TOKENS} greedy tokens each.",
        f"- Device: {hardware}.",
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
    return lines


if __name__ == "__main__":
    main()
