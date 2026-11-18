"""Tests for the pure helpers of ``scripts/validate_kl.py``.

The script imports torch only inside the functions that need it, so the helpers here run in the
unit-test environment without a model.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_kl.py"


def _load_script() -> ModuleType:
    """Import ``scripts/validate_kl.py`` as a module.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location("validate_kl", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _random_rows(rows: int, vocab: int, seed: int = 0) -> np.ndarray:
    """Random probability rows, each summing to one.

    Args:
        rows: Number of distributions.
        vocab: Vocabulary size.
        seed: Random seed.

    Returns:
        Array of shape ``(rows, vocab)``.
    """
    logits = np.random.default_rng(seed).normal(size=(rows, vocab))
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def test_exact_kl_is_zero_for_identical_distributions() -> None:
    """KL of a distribution with itself is zero."""
    kl = _load_script()
    p = np.array([0.2, 0.3, 0.5])
    assert kl._exact_kl(p, p) == pytest.approx(0.0, abs=1e-12)


def test_exact_kl_matches_the_formula() -> None:
    """The value is the sum of p log(p / q) in nats."""
    kl = _load_script()
    p = np.array([0.5, 0.5])
    q = np.array([0.25, 0.75])
    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert kl._exact_kl(p, q) == pytest.approx(expected)


def test_exact_kl_skips_zero_probability_tokens() -> None:
    """Tokens with zero reference probability add nothing, instead of producing NaN."""
    kl = _load_script()
    p = np.array([1.0, 0.0])
    q = np.array([0.5, 0.5])
    assert kl._exact_kl(p, q) == pytest.approx(math.log(2.0))


def test_ranked_returns_the_most_probable_tokens_first() -> None:
    """The first entries are the largest probabilities, in decreasing order."""
    kl = _load_script()
    probs = np.array([0.1, 0.4, 0.05, 0.3, 0.15])
    assert list(kl._ranked(probs, 3)) == [1, 3, 4]


def test_topk_keeps_the_k_largest_as_natural_logs() -> None:
    """The top-k dictionary maps token ids to natural-log probabilities."""
    kl = _load_script()
    probs = np.array([0.1, 0.4, 0.05, 0.3, 0.15])
    top = kl._topk(probs, kl._ranked(probs, 5), 2)
    assert set(top) == {1, 3}
    assert top[1] == pytest.approx(math.log(0.4))
    assert top[3] == pytest.approx(math.log(0.3))


def test_prompt_means_is_zero_for_identical_rows() -> None:
    """Identical reference and candidate give zero exact KL and zero estimate for every k."""
    kl = _load_script()
    probs = _random_rows(4, 30)
    means = kl._prompt_means(probs, probs)
    assert set(means) == set(kl.KS)
    for exact, estimate in means.values():
        assert exact == pytest.approx(0.0, abs=1e-12)
        assert estimate == pytest.approx(0.0, abs=1e-9)


def test_prompt_means_exact_part_is_the_mean_of_the_row_values() -> None:
    """The exact part is the mean of the per-position exact KL, for every k."""
    kl = _load_script()
    ref = _random_rows(5, 30, seed=1)
    cand = _random_rows(5, 30, seed=2)
    expected = float(np.mean([kl._exact_kl(p, q) for p, q in zip(ref, cand, strict=True)]))
    for exact, _ in kl._prompt_means(ref, cand).values():
        assert exact == pytest.approx(expected)


def test_pearson_and_spearman_on_known_relations() -> None:
    """A linear relation gives Pearson 1, and any increasing transform gives Spearman 1."""
    kl = _load_script()
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert kl._pearson(x, 2 * x + 1) == pytest.approx(1.0)
    assert kl._spearman(x, np.exp(x)) == pytest.approx(1.0)
    assert math.isnan(kl._pearson(x, np.ones(4)))


class _FakeTokenizer:
    """Records the arguments of ``apply_chat_template`` and returns a fixed value."""

    def __init__(self, output: Any) -> None:
        self.output = output
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
        """Record the keyword arguments and return the fixed output.

        Args:
            messages: Chat messages (not inspected).
            **kwargs: Keyword arguments of the call.

        Returns:
            The fixed output.
        """
        self.calls.append(kwargs)
        return self.output


def test_prompt_ids_asks_for_plain_token_ids() -> None:
    """The call passes return_dict=False, so transformers 5 does not return a dictionary."""
    kl = _load_script()
    tokenizer = _FakeTokenizer([1, 2, 3])
    ids = kl._prompt_ids(tokenizer, [{"role": "user", "content": "hi"}])
    assert ids == [1, 2, 3]
    assert tokenizer.calls == [
        {"add_generation_prompt": True, "tokenize": True, "return_dict": False}
    ]


def test_prompt_ids_unwraps_a_batch_of_one() -> None:
    """A list of one list of ids is unwrapped to a flat list."""
    kl = _load_script()
    assert kl._prompt_ids(_FakeTokenizer([[4, 5]]), [{"role": "user", "content": "x"}]) == [4, 5]
