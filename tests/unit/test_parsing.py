"""Tests for the response parsers, using payloads shaped like vLLM's responses."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator

import pytest

from day0.errors import ServerResponseError
from day0.parsing import (
    aiter_sse_payloads,
    iter_sse_payloads,
    parse_completion,
    parse_output_token_ids,
    parse_prometheus,
    parse_prompt_logprobs,
    parse_token_key,
    parse_top_logprobs,
)


def test_sse_yields_payloads_and_stops_at_done() -> None:
    """Data lines are yielded in order and nothing after ``[DONE]`` is read."""
    lines = [
        'data: {"a": 1}',
        "",
        ": keep-alive comment",
        'data: {"a": 2}',
        "event: ignored",
        "data: [DONE]",
        'data: {"a": 3}',
    ]
    assert list(iter_sse_payloads(lines)) == ['{"a": 1}', '{"a": 2}']


def test_sse_accepts_data_without_space() -> None:
    """The space after ``data:`` is optional in the SSE grammar."""
    assert list(iter_sse_payloads(['data:{"x": 1}', "data: [DONE]"])) == ['{"x": 1}']


async def test_async_sse_matches_sync_version() -> None:
    """The async iterator yields the same payloads as the synchronous one."""
    lines = ['data: {"a": 1}', ": c", "", 'data: {"a": 2}', "data: [DONE]"]

    async def gen() -> AsyncIterator[str]:
        for line in lines:
            yield line

    collected = [payload async for payload in aiter_sse_payloads(gen())]
    assert collected == list(iter_sse_payloads(lines))


def test_parse_token_key_accepts_both_forms() -> None:
    """Token keys may be plain numbers, ``token_id:`` strings, or integers."""
    assert parse_token_key("84") == 84
    assert parse_token_key("token_id:84") == 84
    assert parse_token_key(84) == 84
    with pytest.raises(ServerResponseError):
        parse_token_key("T")


def test_output_ids_from_token_ids_field() -> None:
    """``token_ids`` on the choice takes precedence."""
    choice = {"token_ids": [1, 2, 3], "logprobs": {"tokens": ["token_id:9"]}}
    assert parse_output_token_ids(choice) == [1, 2, 3]


def test_output_ids_from_logprobs_tokens() -> None:
    """Tokens in ``token_id:<n>`` form map back to ids."""
    choice = {"logprobs": {"tokens": ["token_id:72", "token_id:105"]}}
    assert parse_output_token_ids(choice) == [72, 105]


def test_output_ids_unavailable_for_text_tokens() -> None:
    """Text tokens cannot be mapped to ids, so the result is ``None``, not a guess."""
    assert parse_output_token_ids({"logprobs": {"tokens": ["H", "i"]}}) is None
    assert parse_output_token_ids({}) is None


def test_top_logprobs_null_entries_become_empty_dicts() -> None:
    """A ``null`` row becomes an empty distribution, and keys are parsed to ints."""
    choice = {
        "logprobs": {
            "top_logprobs": [{"token_id:84": -0.1, "token_id:72": -2.3}, None],
        }
    }
    rows = parse_top_logprobs(choice)
    assert rows == [{84: -0.1, 72: -2.3}, {}]


def test_prompt_logprobs_position_zero_is_null() -> None:
    """Position 0 is ``None`` and the actual token is present in later positions."""
    choice = {
        "prompt_logprobs": [
            None,
            {
                "84": {"logprob": -1.2, "rank": 1, "decoded_token": "T"},
                "72": {"logprob": -2.0, "rank": 2, "decoded_token": "H"},
            },
        ]
    }
    rows = parse_prompt_logprobs(choice, {})
    assert rows is not None
    assert rows[0] is None
    assert rows[1] == {84: -1.2, 72: -2.0}


def test_prompt_logprobs_read_from_top_level_and_accept_prefix() -> None:
    """The field may sit at the top level, and keys may use the ``token_id:`` prefix."""
    payload = {"prompt_logprobs": [None, {"token_id:5": {"logprob": -0.5}}]}
    rows = parse_prompt_logprobs({}, payload)
    assert rows == [None, {5: -0.5}]


def test_prompt_logprobs_absent_returns_none() -> None:
    """Without the field the result is ``None``, not an empty list."""
    assert parse_prompt_logprobs({}, {}) is None


def _completion_payload() -> dict[str, object]:
    """Return a completion payload shaped like the vLLM response in the specification.

    Returns:
        A decoded JSON object.
    """
    return {
        "id": "cmpl-1",
        "object": "text_completion",
        "choices": [
            {
                "index": 0,
                "text": "Hi",
                "finish_reason": "stop",
                "logprobs": {
                    "tokens": ["token_id:72", "token_id:105"],
                    "token_logprobs": [-0.1, -0.2],
                    "top_logprobs": [{"token_id:72": -0.1}, {"token_id:105": -0.2}],
                    "text_offset": [0, 1],
                },
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }


def test_parse_completion_extracts_every_field() -> None:
    """A complete response yields text, ids, logprobs, and usage."""
    parsed = parse_completion(_completion_payload())
    assert parsed.text == "Hi"
    assert parsed.finish_reason == "stop"
    assert parsed.output_ids == [72, 105]
    assert parsed.gen_token_logprobs == [-0.1, -0.2]
    assert parsed.gen_topk == [{72: -0.1}, {105: -0.2}]
    assert parsed.prompt_topk is None
    assert (parsed.prompt_tokens, parsed.completion_tokens) == (4, 2)


def test_parse_completion_rejects_missing_choices_or_usage() -> None:
    """A response without choices or usage is an error, not a partial result."""
    with pytest.raises(ServerResponseError, match="no choices"):
        parse_completion({"choices": [], "usage": {}})
    payload = _completion_payload()
    del payload["usage"]
    with pytest.raises(ServerResponseError, match="usage"):
        parse_completion(payload)


def test_prometheus_labels_are_sorted_and_values_parsed() -> None:
    """Labels are sorted in the key, and special values parse correctly."""
    text = "\n".join(
        [
            "# HELP vllm:requests Number of requests.",
            "# TYPE vllm:requests counter",
            'vllm:requests{model_name="m",engine="0"} 12.5',
            "vllm:plain_gauge 3",
            "vllm:huge 1.5e3",
            "vllm:infinite +Inf",
            "vllm:missing NaN",
            "this line is malformed",
            "",
        ]
    )
    samples = parse_prometheus(text)
    assert samples['vllm:requests{engine="0",model_name="m"}'] == 12.5
    assert samples["vllm:plain_gauge"] == 3.0
    assert samples["vllm:huge"] == 1500.0
    assert math.isinf(samples["vllm:infinite"])
    assert math.isnan(samples["vllm:missing"])
    assert "this" not in samples


def test_prometheus_timestamp_is_ignored() -> None:
    """An optional trailing timestamp does not change the value."""
    assert parse_prometheus('m{a="b"} 7 1700000000000') == {'m{a="b"}': 7.0}
