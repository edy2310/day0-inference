"""Tests for the fake model's vectorized logits and the prompt-logprob text builder."""

from __future__ import annotations

import json

import numpy as np

from day0.fake.model import (
    TOKEN_TEXT,
    ModelParams,
    encode_text,
    log_softmax_rows,
    logits_rows,
    plan_turn,
    render_chat,
    top_k_rows,
)
from day0.fake.server import _prompt_logprobs_json


def _expected_prompt_logprobs(
    ids: list[int], plan: object, params: ModelParams, k: int
) -> list[dict[str, object] | None]:
    """Build the prompt logprobs the straightforward way, one dictionary per position.

    Args:
        ids: Prompt token ids.
        plan: Turn plan of the prompt.
        params: Model parameters.
        k: Top-k size.

    Returns:
        One entry per position, as the JSON structure should read.
    """
    rows: list[dict[str, object] | None] = [None]
    positions = np.arange(1, len(ids), dtype=np.int64)
    logprobs = log_softmax_rows(logits_rows(ids, positions, plan, params))  # type: ignore[arg-type]
    ranked_ids, ranked_values = top_k_rows(logprobs, k)
    for row, position in enumerate(positions.tolist()):
        entries: dict[str, object] = {}
        for rank, (token, value) in enumerate(
            zip(ranked_ids[row].tolist(), ranked_values[row].tolist(), strict=True), start=1
        ):
            entries[str(token)] = {
                "logprob": value,
                "rank": rank,
                "decoded_token": TOKEN_TEXT[token],
            }
        actual = ids[position]
        if str(actual) not in entries:
            actual_lp = float(logprobs[row, actual])
            rank = int(np.sum(logprobs[row] > actual_lp)) + 1
            entries[str(actual)] = {
                "logprob": actual_lp,
                "rank": rank,
                "decoded_token": TOKEN_TEXT[actual],
            }
        rows.append(entries)
    return rows


def test_prompt_logprob_text_equals_the_dictionary_form() -> None:
    """The fast text builder decodes to the same structure as the dictionary form.

    It is checked with and without position-dependent noise, which uses the per-row path.
    """
    text = render_chat([("user", "Count to three. <<ECHO:3>>")], add_generation_prompt=True)
    ids = encode_text(text, add_special_tokens=True) + list(b"The answer is 3.") + [257]
    plan = plan_turn(ids, ModelParams())
    for params in (
        ModelParams(noise=0.3),
        ModelParams(noise=0.3, long_ctx_start=5, long_ctx_factor=40.0),
    ):
        expected = _expected_prompt_logprobs(ids, plan, params, k=4)
        actual = json.loads(_prompt_logprobs_json(ids, plan_turn(ids, params), params, 4))
        assert actual == expected


def test_logits_do_not_depend_on_batch_composition() -> None:
    """A position gives the same logits alone and inside a larger batch."""
    ids = encode_text("the quick brown fox jumps", add_special_tokens=True)
    params = ModelParams(noise=0.5, long_ctx_start=3, long_ctx_factor=10.0)
    plan = plan_turn(ids, params)
    positions = np.arange(1, len(ids), dtype=np.int64)
    batch = logits_rows(ids, positions, plan, params)
    for row, position in enumerate(positions.tolist()):
        alone = logits_rows(ids, np.array([position], dtype=np.int64), plan, params)[0]
        assert np.array_equal(alone, batch[row])
