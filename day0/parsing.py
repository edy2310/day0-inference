"""Tolerant parsers for vLLM-style responses.

Covers server-sent events, token logprobs and top-k logprobs in both the ``token_id:<n>`` and
plain-number key forms, prompt logprobs (where position 0 is ``null``), and the Prometheus text
format. The tests load real fixtures when they are present (see ``tests/fixtures/vllm``).
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from day0.errors import ServerResponseError

TopK = dict[int, float]
_TOKEN_PREFIX = "token_id:"
_DATA_PREFIX = "data:"
_DONE = "[DONE]"
_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+(?P<value>\S+)"
    r"(?:\s+-?\d+)?\s*$"
)
_LABEL_PAIR = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class ParsedCompletion:
    """Fields of one completion response that day0 needs.

    Attributes:
        text: Generated text.
        finish_reason: ``stop``, ``length``, or ``None``.
        output_ids: Generated token ids, or ``None`` when the server did not return them.
        gen_token_logprobs: Logprob of each generated token, or ``None``.
        gen_topk: Top-k distribution per generated position, or ``None``.
        prompt_topk: Top-k distribution per prompt position (position 0 is ``None``), or ``None``.
        prompt_tokens: Prompt length reported by the server.
        completion_tokens: Output length reported by the server.
    """

    text: str
    finish_reason: str | None
    output_ids: list[int] | None
    gen_token_logprobs: list[float] | None
    gen_topk: list[TopK] | None
    prompt_topk: list[TopK | None] | None
    prompt_tokens: int
    completion_tokens: int


def _sse_payload(line: str) -> str | None:
    """Return the data payload of one SSE line.

    Args:
        line: One line of the event stream, without its newline.

    Returns:
        The payload, or ``None`` for empty lines, comments, and non-data fields.
    """
    if not line or line.startswith(":") or not line.startswith(_DATA_PREFIX):
        return None
    return line[len(_DATA_PREFIX) :].lstrip(" ")


def iter_sse_payloads(lines: Iterable[str]) -> Iterator[str]:
    """Yield the data payloads of a server-sent event stream until ``[DONE]``.

    Args:
        lines: Lines of the stream, without newlines.

    Yields:
        Each JSON payload as a string.
    """
    for line in lines:
        payload = _sse_payload(line)
        if payload is None:
            continue
        if payload == _DONE:
            return
        yield payload


async def aiter_sse_payloads(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Async version of :func:`iter_sse_payloads`.

    Args:
        lines: Async iterator over the lines of the stream.

    Yields:
        Each JSON payload as a string, as soon as its line arrives.
    """
    async for line in lines:
        payload = _sse_payload(line)
        if payload is None:
            continue
        if payload == _DONE:
            return
        yield payload


def parse_token_key(key: str | int) -> int:
    """Convert a token key to an integer id.

    Args:
        key: Either an integer, ``"token_id:<n>"``, or a plain numeric string.

    Returns:
        The token id.

    Raises:
        ServerResponseError: If the key is not a token id.
    """
    if isinstance(key, int):
        return key
    text = key.strip()
    if text.startswith(_TOKEN_PREFIX):
        text = text[len(_TOKEN_PREFIX) :]
    try:
        return int(text)
    except ValueError as exc:
        raise ServerResponseError(f"cannot parse token key {key!r} as a token id") from exc


def parse_output_token_ids(choice: Mapping[str, Any]) -> list[int] | None:
    """Return the generated token ids of a choice, if the server exposed them.

    Preference order: ``token_ids`` on the choice, then ``logprobs.tokens`` in the
    ``token_id:<n>`` form. Text tokens cannot be mapped back to ids, so they yield ``None``.

    Args:
        choice: One entry of ``choices``.

    Returns:
        The token ids, or ``None``.
    """
    token_ids = choice.get("token_ids")
    if isinstance(token_ids, list):
        return [parse_token_key(item) for item in token_ids]
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, Mapping):
        tokens = logprobs.get("tokens")
        if isinstance(tokens, list) and all(
            isinstance(item, str) and item.startswith(_TOKEN_PREFIX) for item in tokens
        ):
            return [parse_token_key(item) for item in tokens]
    return None


def parse_token_logprobs(choice: Mapping[str, Any]) -> list[float] | None:
    """Return the logprob of each generated token, if present.

    Args:
        choice: One entry of ``choices``.

    Returns:
        The logprobs, or ``None``.
    """
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, Mapping):
        return None
    values = logprobs.get("token_logprobs")
    if not isinstance(values, list):
        return None
    return [float(value) for value in values]


def parse_top_logprobs(choice: Mapping[str, Any]) -> list[TopK] | None:
    """Return the top-k distribution of each generated position, if present.

    A ``null`` entry becomes an empty dictionary.

    Args:
        choice: One entry of ``choices``.

    Returns:
        One dictionary per position, or ``None``.
    """
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, Mapping):
        return None
    rows = logprobs.get("top_logprobs")
    if not isinstance(rows, list):
        return None
    result: list[TopK] = []
    for row in rows:
        if row is None:
            result.append({})
            continue
        if not isinstance(row, Mapping):
            raise ServerResponseError("top_logprobs entry is neither null nor an object")
        result.append({parse_token_key(key): float(value) for key, value in row.items()})
    return result


def parse_prompt_logprobs(
    choice: Mapping[str, Any], payload: Mapping[str, Any]
) -> list[TopK | None] | None:
    """Return the per-position prompt distributions, if present.

    The field is read from the choice first, then from the top level of the response. Position 0
    is ``null``. Each other position maps token ids to ``{"logprob": ...}`` objects.

    Args:
        choice: One entry of ``choices``.
        payload: The full response object.

    Returns:
        One entry per prompt position, or ``None``.
    """
    rows = choice.get("prompt_logprobs")
    if rows is None:
        rows = payload.get("prompt_logprobs")
    if rows is None:
        return None
    if not isinstance(rows, list):
        raise ServerResponseError("prompt_logprobs must be a list")
    result: list[TopK | None] = []
    for row in rows:
        if row is None:
            result.append(None)
            continue
        if not isinstance(row, Mapping):
            raise ServerResponseError("prompt_logprobs entry is neither null nor an object")
        entries: TopK = {}
        for key, value in row.items():
            logprob = value["logprob"] if isinstance(value, Mapping) else value
            entries[parse_token_key(key)] = float(logprob)
        result.append(entries)
    return result


def parse_completion(payload: Mapping[str, Any]) -> ParsedCompletion:
    """Parse a non-streaming ``/v1/completions`` response.

    Args:
        payload: The decoded JSON object.

    Returns:
        The parsed fields.

    Raises:
        ServerResponseError: If ``choices`` or ``usage`` is missing or malformed.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ServerResponseError("completion response has no choices")
    choice = choices[0]
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        raise ServerResponseError("completion response has no usage object")
    finish = choice.get("finish_reason")
    return ParsedCompletion(
        text=str(choice.get("text", "")),
        finish_reason=str(finish) if finish is not None else None,
        output_ids=parse_output_token_ids(choice),
        gen_token_logprobs=parse_token_logprobs(choice),
        gen_topk=parse_top_logprobs(choice),
        prompt_topk=parse_prompt_logprobs(choice, payload),
        prompt_tokens=int(usage["prompt_tokens"]),
        completion_tokens=int(usage["completion_tokens"]),
    )


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse the Prometheus text exposition format.

    Keys are the metric name when there are no labels, and ``name{a="x",b="y"}`` with labels
    sorted by name otherwise. Malformed lines are skipped.

    Args:
        text: The response body of ``GET /metrics``.

    Returns:
        A mapping from metric key to value. ``+Inf``, ``-Inf``, ``NaN``, and scientific notation
        are accepted.
    """
    samples: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if match is None:
            continue
        name = match.group("name")
        labels = match.group("labels")
        key = name
        if labels:
            pairs = sorted(_LABEL_PAIR.findall(labels))
            key = name + "{" + ",".join(f'{k}="{v}"' for k, v in pairs) + "}"
        samples[key] = _parse_float(match.group("value"))
    return samples


def _parse_float(text: str) -> float:
    """Parse a Prometheus sample value.

    Args:
        text: The value token, for example ``1.5e3``, ``+Inf``, or ``NaN``.

    Returns:
        The float value. Python's ``float`` accepts every spelling used by the format.
    """
    return float(text)
