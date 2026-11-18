"""Starlette app that imitates the parts of the vLLM OpenAI-compatible server that day0 uses.

Run it directly with ``python -m day0.fake.server --port 8199``. The request and response shapes
follow §13.3 of the specification, so the same client and parsers work against both servers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from day0.fake.model import (
    EOS_ID,
    TOKEN_TEXT,
    ModelParams,
    TurnPlan,
    context_tails,
    decode,
    encode_text,
    greedy_token,
    log_softmax,
    log_softmax_rows,
    logits_rows,
    next_token_logits,
    plan_turn,
    render_chat,
    token_text,
    top_k_pairs,
    top_k_rows,
)

_PROMPT_CHUNK = 4096

FAKE_VERSION = "fake-0.1"
_MILLISECONDS = 1000.0


@dataclass(frozen=True)
class FakeServerConfig:
    """Server-level settings: timing, limits, and the model parameters.

    Attributes:
        model_name: Served model id.
        max_model_len: Maximum prompt plus output length.
        max_logprobs: Upper bound for ``logprobs`` and ``prompt_logprobs``.
        startup_delay_s: Seconds during which ``/health`` answers 503.
        speedup: Divides every simulated latency. Values below 1 slow the server down.
        ttft_base_ms: Fixed part of the time to first token.
        prefill_ms_per_1k: Time to first token per 1000 prompt tokens.
        itl_base_ms: Time between output tokens at concurrency 1.
        concurrency_penalty: Latency growth per concurrently active request.
        spec_accept_rate: If set, ``/metrics`` exposes speculative-decoding counters.
        params: Model parameters.
    """

    model_name: str = "fake-model"
    max_model_len: int = 65536
    max_logprobs: int = 20
    startup_delay_s: float = 0.0
    speedup: float = 1.0
    ttft_base_ms: float = 20.0
    prefill_ms_per_1k: float = 5.0
    itl_base_ms: float = 2.0
    concurrency_penalty: float = 0.02
    spec_accept_rate: float | None = None
    params: ModelParams = field(default_factory=ModelParams)


@dataclass
class _ServerState:
    """Mutable state shared by the request handlers."""

    config: FakeServerConfig
    started_at: float = field(default_factory=time.monotonic)
    active: int = 0
    spec_drafts: int = 0
    spec_draft_tokens: int = 0
    spec_accepted: int = 0

    def slowdown(self) -> float:
        """Return the latency multiplier for the current number of active requests.

        Returns:
            ``(1 + penalty * active) / speedup``.
        """
        return (1.0 + self.config.concurrency_penalty * self.active) / self.config.speedup

    def ttft_seconds(self, prompt_len: int) -> float:
        """Simulated time to first token.

        Args:
            prompt_len: Prompt length in tokens.

        Returns:
            Seconds to wait before the first output token.
        """
        base = self.config.ttft_base_ms + self.config.prefill_ms_per_1k * prompt_len / 1000.0
        return base * self.slowdown() / _MILLISECONDS

    def itl_seconds(self) -> float:
        """Simulated time between two output tokens.

        Returns:
            Seconds to wait before each subsequent output token.
        """
        return self.config.itl_base_ms * self.slowdown() / _MILLISECONDS

    def record_spec(self, generated: int) -> None:
        """Update the speculative-decoding counters for one finished generation.

        Args:
            generated: Number of output tokens of the finished request.
        """
        rate = self.config.spec_accept_rate
        if rate is None or generated <= 0:
            return
        draft_len = 4
        steps = math.ceil(generated / (1.0 + draft_len * rate))
        self.spec_drafts += steps
        self.spec_draft_tokens += draft_len * steps
        self.spec_accepted += int(round(rate * draft_len * steps))


class _Decoder:
    """Greedy decoding of one request, one token at a time."""

    def __init__(
        self,
        prompt: list[int],
        plan: TurnPlan,
        params: ModelParams,
        *,
        max_tokens: int,
        ignore_eos: bool,
        logprobs: int | None,
    ) -> None:
        """Prepare decoding of one request.

        Args:
            prompt: Prompt token ids.
            plan: Turn plan of the prompt.
            params: Model parameters.
            max_tokens: Output budget.
            ignore_eos: Whether EOS is blocked.
            logprobs: Number of top logprobs to keep per token, or ``None``.
        """
        self.buffer = np.zeros(len(prompt) + max_tokens, dtype=np.int64)
        self.buffer[: len(prompt)] = prompt
        self.length = len(prompt)
        self.plan = plan
        self.params = params
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.logprobs = logprobs
        self.out: list[int] = []
        self.logprob_rows: list[tuple[float, list[tuple[int, float]]]] = []
        self.finish_reason: str | None = None
        self._finished = False

    def step(self) -> int | None:
        """Generate one token.

        Returns:
            The new token id, or ``None`` when generation has finished.
        """
        if self._finished:
            return None
        if len(self.out) >= self.max_tokens:
            self._finished = True
            self.finish_reason = "length"
            return None
        logits = next_token_logits(self.buffer, self.length, self.plan, self.params)
        token = greedy_token(logits, ignore_eos=self.ignore_eos)
        if token == EOS_ID and not self.ignore_eos:
            self._finished = True
            self.finish_reason = "stop"
            return None
        if self.logprobs is not None:
            lp = log_softmax(logits)
            pairs = (
                top_k_pairs(lp, self.logprobs) if self.logprobs > 0 else [(token, float(lp[token]))]
            )
            self.logprob_rows.append((float(lp[token]), pairs))
        self.buffer[self.length] = token
        self.length += 1
        self.out.append(token)
        return token


def _token_key(token_id: int, return_ids: bool) -> str:
    """Name a token the way ``return_tokens_as_token_ids`` requests.

    Args:
        token_id: Token id.
        return_ids: If true, return ``token_id:<id>``; otherwise the decoded text.

    Returns:
        The token key.
    """
    return f"token_id:{token_id}" if return_ids else token_text(token_id)


def _logprobs_payload(
    rows: Sequence[tuple[float, list[tuple[int, float]]]], tokens: list[int], return_ids: bool
) -> dict[str, Any]:
    """Build the ``logprobs`` object of a non-streaming completion choice.

    Args:
        rows: Per generated token: chosen logprob and top-k pairs.
        tokens: Generated token ids.
        return_ids: Whether tokens are named ``token_id:<id>``.

    Returns:
        The ``logprobs`` object.
    """
    offsets: list[int] = []
    cursor = 0
    for token in tokens:
        offsets.append(cursor)
        cursor += len(token_text(token))
    return {
        "tokens": [_token_key(token, return_ids) for token in tokens],
        "token_logprobs": [row[0] for row in rows],
        "top_logprobs": [{_token_key(tid, return_ids): lp for tid, lp in row[1]} for row in rows],
        "text_offset": offsets,
    }


_PROMPT_SENTINEL = "__day0_prompt_logprobs__"
_JSON_TOKEN: tuple[str, ...] = tuple(json.dumps(text, ensure_ascii=False) for text in TOKEN_TEXT)


def _prompt_logprobs_json(ids: list[int], plan: TurnPlan, params: ModelParams, k: int) -> str:
    """JSON text of ``prompt_logprobs`` for a prompt.

    Position ``p`` describes the distribution for token ``ids[p]`` given ``ids[:p]``. It holds the
    actual token plus the top-k tokens, so it can contain up to ``k + 1`` entries. The text has the
    same structure as ``json.dumps`` of the equivalent dictionaries (checked by a unit test).

    Where the distribution depends only on the 3-token context (before the assistant marker, and
    with no position-dependent noise), each distinct context is computed once for the whole prompt.
    Other positions are computed in chunks. The handler runs this in a thread, so the event loop
    keeps serving requests.

    Args:
        ids: Prompt token ids.
        plan: Turn plan of the prompt.
        params: Model parameters.
        k: Number of top tokens per position.

    Returns:
        A JSON array with one element per prompt position. Position 0 is ``null``.
    """
    n_positions = len(ids) - 1
    if n_positions < 1:
        return "[null]"
    positions = np.arange(1, len(ids), dtype=np.int64)
    actual = np.asarray(ids[1:], dtype=np.int64)
    context_only = params.noise == 0 or params.long_ctx_factor == 0
    if context_only:
        if plan.assistant_end is not None:
            shared = positions < plan.assistant_end
        else:
            shared = np.ones(n_positions, dtype=bool)
    else:
        shared = np.zeros(n_positions, dtype=bool)

    pieces: list[str] = [""] * n_positions
    shared_index = np.nonzero(shared)[0]
    if shared_index.size:
        _fill_shared(
            ids,
            plan,
            params,
            k,
            positions[shared_index],
            actual[shared_index],
            shared_index,
            pieces,
        )
    own_index = np.nonzero(~shared)[0]
    for start in range(0, own_index.size, _PROMPT_CHUNK):
        chosen = own_index[start : start + _PROMPT_CHUNK]
        _fill_rows(ids, plan, params, k, positions[chosen], actual[chosen], chosen, pieces)
    return "[null," + ",".join(pieces) + "]"


def _fill_shared(
    ids: list[int],
    plan: TurnPlan,
    params: ModelParams,
    k: int,
    positions: np.ndarray,
    actual: np.ndarray,
    slots: np.ndarray,
    pieces: list[str],
) -> None:
    """Fill positions whose distribution depends only on their 3-token context.

    Args:
        ids: Prompt token ids.
        plan: Turn plan of the prompt.
        params: Model parameters.
        k: Top-k size.
        positions: Context lengths of the shared positions.
        actual: Actual next token of each shared position.
        slots: Index of each shared position in ``pieces``.
        pieces: Output list, one entry per prompt position.
    """
    tails = context_tails(ids, positions)
    unique, first, inverse_raw = np.unique(tails, axis=0, return_index=True, return_inverse=True)
    inverse = inverse_raw.reshape(-1)
    representatives = positions[first.reshape(-1)]
    logprobs = log_softmax_rows(logits_rows(ids, representatives, plan, params))
    top_ids, top_values = top_k_rows(logprobs, k)
    sorted_rows = np.sort(logprobs, axis=1)
    actual_lp = logprobs[inverse, actual]
    actual_rank = np.zeros(positions.shape[0], dtype=np.int64)
    for context in range(unique.shape[0]):
        group = np.nonzero(inverse == context)[0]
        at_or_below = np.searchsorted(sorted_rows[context], actual_lp[group], side="right")
        actual_rank[group] = (sorted_rows.shape[1] - at_or_below) + 1

    texts = [
        _top_text(top_ids[context].tolist(), top_values[context].tolist())
        for context in range(unique.shape[0])
    ]
    members = [set(top_ids[context].tolist()) for context in range(unique.shape[0])]
    inverse_list = inverse.tolist()
    actual_list = actual.tolist()
    lp_list = actual_lp.tolist()
    rank_list = actual_rank.tolist()
    for row, slot in enumerate(slots.tolist()):
        context = inverse_list[row]
        token = actual_list[row]
        if token in members[context]:
            body = texts[context]
        else:
            entry = _entry_text(token, lp_list[row], rank_list[row])
            body = f"{texts[context]},{entry}" if texts[context] else entry
        pieces[slot] = "{" + body + "}"


def _fill_rows(
    ids: list[int],
    plan: TurnPlan,
    params: ModelParams,
    k: int,
    positions: np.ndarray,
    actual: np.ndarray,
    slots: np.ndarray,
    pieces: list[str],
) -> None:
    """Fill positions whose distribution is computed one chunk at a time.

    Args:
        ids: Prompt token ids.
        plan: Turn plan of the prompt.
        params: Model parameters.
        k: Top-k size.
        positions: Context lengths of the positions.
        actual: Actual next token of each position.
        slots: Index of each position in ``pieces``.
        pieces: Output list, one entry per prompt position.
    """
    logprobs = log_softmax_rows(logits_rows(ids, positions, plan, params))
    top_ids, top_values = top_k_rows(logprobs, k)
    rows = np.arange(positions.shape[0])
    actual_lp = logprobs[rows, actual]
    actual_rank = np.sum(logprobs > actual_lp[:, None], axis=1) + 1
    id_rows = top_ids.tolist()
    value_rows = top_values.tolist()
    actual_list = actual.tolist()
    lp_list = actual_lp.tolist()
    rank_list = actual_rank.tolist()
    for row, slot in enumerate(slots.tolist()):
        token = actual_list[row]
        parts = [_top_text(id_rows[row], value_rows[row])]
        if token not in id_rows[row]:
            parts.append(_entry_text(token, lp_list[row], rank_list[row]))
        pieces[slot] = "{" + ",".join(part for part in parts if part) + "}"


def _entry_text(token_id: int, logprob: float, rank: int) -> str:
    """JSON text of one ``prompt_logprobs`` entry, ``"<id>":{...}``.

    Args:
        token_id: Token id used as the key.
        logprob: Natural-log probability.
        rank: One-based rank.

    Returns:
        The entry text.
    """
    return (
        f'"{token_id}":{{"logprob":{logprob!r},"rank":{rank},'
        f'"decoded_token":{_JSON_TOKEN[token_id]}}}'
    )


def _top_text(ids: list[int], values: list[float]) -> str:
    """JSON text of the top-k entries of one distribution, without the braces.

    Args:
        ids: Token ids in rank order.
        values: Log-probabilities in the same order.

    Returns:
        Comma-separated entries, or an empty string when there are none.
    """
    return ",".join(
        _entry_text(token_id, value, rank)
        for rank, (token_id, value) in enumerate(zip(ids, values, strict=True), start=1)
    )


def _error(status: int, message: str) -> JSONResponse:
    """Build an error response in the shape vLLM uses.

    Args:
        status: HTTP status code.
        message: Human-readable message.

    Returns:
        The JSON error response.
    """
    body = {"error": {"message": message, "type": "invalid_request_error", "code": status}}
    return JSONResponse(body, status_code=status)


async def _json_object(request: Request) -> dict[str, Any] | Response:
    """Parse a JSON request body that must be an object.

    Args:
        request: Incoming request.

    Returns:
        The parsed object, or an error response when the body is not a JSON object.
    """
    try:
        body: object = await request.json()
    except ValueError:
        return _error(400, "request body must be valid JSON")
    if not isinstance(body, dict):
        return _error(400, "request body must be a JSON object")
    return body


def _prompt_ids(prompt: object) -> list[int] | None:
    """Turn a ``prompt`` field into token ids.

    Args:
        prompt: A string (tokenized with BOS) or a list of token ids.

    Returns:
        Token ids, or ``None`` when the field has the wrong shape or is empty.
    """
    if isinstance(prompt, str):
        ids = encode_text(prompt, add_special_tokens=True)
    elif isinstance(prompt, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in prompt
    ):
        ids = [int(item) for item in prompt]
    else:
        return None
    return ids or None


def create_app(config: FakeServerConfig) -> Starlette:
    """Build the Starlette application with the given settings.

    Args:
        config: Server settings.

    Returns:
        A Starlette application that serves the vLLM-shaped endpoints.
    """
    state = _ServerState(config=config)
    params = config.params

    async def health(request: Request) -> Response:
        """Answer 503 during the configured startup delay, then 200."""
        del request
        if time.monotonic() - state.started_at < config.startup_delay_s:
            return Response(status_code=503)
        return Response(status_code=200)

    async def models(request: Request) -> Response:
        """List the single served model."""
        del request
        data = [{"id": config.model_name, "object": "model", "max_model_len": config.max_model_len}]
        return JSONResponse({"object": "list", "data": data})

    async def version(request: Request) -> Response:
        """Report the fake server version."""
        del request
        return JSONResponse({"version": FAKE_VERSION})

    async def tokenize(request: Request) -> Response:
        """Tokenize a prompt string or a list of chat messages."""
        body = await _json_object(request)
        if isinstance(body, Response):
            return body
        if "messages" in body:
            raw_messages = body["messages"]
            if not isinstance(raw_messages, list) or not all(
                isinstance(m, dict)
                and isinstance(m.get("role"), str)
                and isinstance(m.get("content"), str)
                for m in raw_messages
            ):
                return _error(400, "messages must be a list of {role, content} strings")
            pairs = [(str(m["role"]), str(m["content"])) for m in raw_messages]
            add_generation = bool(body.get("add_generation_prompt", True))
            text = render_chat(pairs, add_generation_prompt=add_generation)
            ids = encode_text(text, add_special_tokens=True)
        elif isinstance(body.get("prompt"), str):
            ids = encode_text(
                str(body["prompt"]), add_special_tokens=bool(body.get("add_special_tokens", True))
            )
        else:
            return _error(400, "provide either 'prompt' (string) or 'messages'")
        return JSONResponse(
            {"count": len(ids), "max_model_len": config.max_model_len, "tokens": ids}
        )

    async def completions(request: Request) -> Response:
        """Serve ``POST /v1/completions`` with greedy decoding, logprobs, and streaming."""
        body = await _json_object(request)
        if isinstance(body, Response):
            return body
        model = body.get("model")
        if model is not None and model != config.model_name:
            return _error(404, f"The model '{model}' does not exist.")
        ids = _prompt_ids(body.get("prompt"))
        if ids is None:
            return _error(400, "prompt must be a non-empty string or a list of token ids")
        max_tokens = int(body.get("max_tokens", 16))
        if max_tokens < 1:
            return _error(400, "max_tokens must be at least 1")
        if float(body.get("temperature", 0.0)) > 0.0:
            return _error(400, "the fake server is greedy only; temperature must be 0")
        if int(body.get("n", 1)) != 1:
            return _error(400, "the fake server supports n=1 only")
        for name in ("logprobs", "prompt_logprobs"):
            value = body.get(name)
            if value is None:
                continue
            if not isinstance(value, int) or value < 0:
                return _error(400, f"{name} must be a non-negative integer")
            if value > config.max_logprobs:
                return _error(
                    400, f"{name}={value} is greater than max_logprobs={config.max_logprobs}"
                )
        if len(ids) + max_tokens > config.max_model_len:
            return _error(
                400,
                f"This model's maximum context length is {config.max_model_len} tokens, but the "
                f"request has {len(ids)} prompt tokens and max_tokens={max_tokens}.",
            )

        ignore_eos = bool(body.get("ignore_eos", False))
        return_ids = bool(body.get("return_tokens_as_token_ids", False))
        logprobs_k = body.get("logprobs")
        prompt_k = body.get("prompt_logprobs")
        plan = plan_turn(ids, params)

        if bool(body.get("stream", False)):
            options = body.get("stream_options") or {}
            continuous = bool(options.get("continuous_usage_stats", False))
            include_usage = bool(options.get("include_usage", False))
            return StreamingResponse(
                _stream(ids, plan, max_tokens, ignore_eos, continuous, include_usage),
                media_type="text/event-stream",
            )

        state.active += 1
        try:
            decoder = _Decoder(
                ids,
                plan,
                params,
                max_tokens=max_tokens,
                ignore_eos=ignore_eos,
                logprobs=logprobs_k,
            )
            first = True
            while True:
                token = decoder.step()
                if token is None:
                    break
                await asyncio.sleep(state.ttft_seconds(len(ids)) if first else state.itl_seconds())
                first = False
            choice: dict[str, Any] = {
                "index": 0,
                "text": decode(decoder.out),
                "finish_reason": decoder.finish_reason,
                "logprobs": None,
            }
            if logprobs_k is not None:
                choice["logprobs"] = _logprobs_payload(
                    decoder.logprob_rows, decoder.out, return_ids
                )
            prompt_fragment: str | None = None
            if prompt_k is not None:
                prompt_fragment = await asyncio.to_thread(
                    _prompt_logprobs_json, ids, plan, params, prompt_k
                )
                choice["prompt_logprobs"] = _PROMPT_SENTINEL
            if first:
                await asyncio.sleep(state.ttft_seconds(len(ids)))
            state.record_spec(len(decoder.out))
        finally:
            state.active -= 1
        usage = {
            "prompt_tokens": len(ids),
            "completion_tokens": len(decoder.out),
            "total_tokens": len(ids) + len(decoder.out),
        }
        payload = {
            "id": "cmpl-fake",
            "object": "text_completion",
            "created": 0,
            "model": config.model_name,
            "choices": [choice],
            "usage": usage,
        }
        if prompt_fragment is None:
            return JSONResponse(payload)
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        text = text.replace(f'"{_PROMPT_SENTINEL}"', prompt_fragment, 1)
        return Response(text, media_type="application/json")

    async def _stream(
        ids: list[int],
        plan: TurnPlan,
        max_tokens: int,
        ignore_eos: bool,
        continuous: bool,
        include_usage: bool,
    ) -> AsyncIterator[bytes]:
        """Yield server-sent events, one token per content chunk."""
        state.active += 1
        decoder = _Decoder(
            ids, plan, params, max_tokens=max_tokens, ignore_eos=ignore_eos, logprobs=None
        )
        first = True
        try:
            while True:
                token = decoder.step()
                if token is None:
                    break
                await asyncio.sleep(state.ttft_seconds(len(ids)) if first else state.itl_seconds())
                first = False
                chunk: dict[str, Any] = {
                    "id": "cmpl-fake",
                    "object": "text_completion",
                    "created": 0,
                    "model": config.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "text": token_text(token),
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                if continuous:
                    chunk["usage"] = _usage(ids, len(decoder.out))
                yield _sse(chunk)
            final: dict[str, Any] = {
                "id": "cmpl-fake",
                "object": "text_completion",
                "created": 0,
                "model": config.model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "logprobs": None,
                        "finish_reason": decoder.finish_reason,
                    }
                ],
            }
            yield _sse(final)
            if include_usage:
                yield _sse(
                    {
                        "id": "cmpl-fake",
                        "object": "text_completion",
                        "created": 0,
                        "model": config.model_name,
                        "choices": [],
                        "usage": _usage(ids, len(decoder.out)),
                    }
                )
            yield b"data: [DONE]\n\n"
            state.record_spec(len(decoder.out))
        finally:
            state.active -= 1

    async def metrics(request: Request) -> Response:
        """Expose the Prometheus text format with the fake counters."""
        del request
        label = f'{{model_name="{config.model_name}"}}'
        lines = [
            "# TYPE vllm:num_requests_running gauge",
            f"vllm:num_requests_running{label} {float(state.active)}",
            "# TYPE vllm:gpu_cache_usage_perc gauge",
            f"vllm:gpu_cache_usage_perc{label} 0.0",
        ]
        if config.spec_accept_rate is not None:
            lines += [
                "# TYPE vllm:spec_decode_num_drafts_total counter",
                f"vllm:spec_decode_num_drafts_total{label} {float(state.spec_drafts)}",
                "# TYPE vllm:spec_decode_num_draft_tokens_total counter",
                f"vllm:spec_decode_num_draft_tokens_total{label} {float(state.spec_draft_tokens)}",
                "# TYPE vllm:spec_decode_num_accepted_tokens_total counter",
                f"vllm:spec_decode_num_accepted_tokens_total{label} {float(state.spec_accepted)}",
            ]
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/version", version, methods=["GET"]),
        Route("/v1/models", models, methods=["GET"]),
        Route("/tokenize", tokenize, methods=["POST"]),
        Route("/v1/completions", completions, methods=["POST"]),
        Route("/metrics", metrics, methods=["GET"]),
    ]
    return Starlette(routes=routes)


def _usage(ids: list[int], completion_tokens: int) -> dict[str, int]:
    """Build a usage object.

    Args:
        ids: Prompt token ids.
        completion_tokens: Output tokens so far.

    Returns:
        The usage dictionary.
    """
    return {
        "prompt_tokens": len(ids),
        "completion_tokens": completion_tokens,
        "total_tokens": len(ids) + completion_tokens,
    }


def _sse(payload: dict[str, Any]) -> bytes:
    """Encode one server-sent event.

    Args:
        payload: JSON-compatible event body.

    Returns:
        The event as UTF-8 bytes.
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the fake server.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(prog="python -m day0.fake.server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", default="fake-model")
    parser.add_argument("--model-seed", type=int, default=7)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=11)
    parser.add_argument("--flip-prob", type=float, default=0.0)
    parser.add_argument("--flip-seed", type=int, default=13)
    parser.add_argument("--long-ctx-start", type=int, default=1_000_000)
    parser.add_argument("--long-ctx-factor", type=float, default=0.0)
    parser.add_argument("--speedup", type=float, default=1.0)
    parser.add_argument("--ttft-base-ms", type=float, default=20.0)
    parser.add_argument("--prefill-ms-per-1k", type=float, default=5.0)
    parser.add_argument("--itl-base-ms", type=float, default=2.0)
    parser.add_argument("--concurrency-penalty", type=float, default=0.02)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--spec-accept-rate", type=float, default=None)
    parser.add_argument("--max-logprobs", type=int, default=20)
    parser.add_argument("--startup-delay-s", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments and serve the fake model until interrupted.

    Args:
        argv: Command-line arguments without the program name. ``None`` reads ``sys.argv``.
    """
    args = build_parser().parse_args(argv)
    config = FakeServerConfig(
        model_name=args.model_name,
        max_model_len=args.max_model_len,
        max_logprobs=args.max_logprobs,
        startup_delay_s=args.startup_delay_s,
        speedup=args.speedup,
        ttft_base_ms=args.ttft_base_ms,
        prefill_ms_per_1k=args.prefill_ms_per_1k,
        itl_base_ms=args.itl_base_ms,
        concurrency_penalty=args.concurrency_penalty,
        spec_accept_rate=args.spec_accept_rate,
        params=ModelParams(
            model_seed=args.model_seed,
            noise=args.noise,
            noise_seed=args.noise_seed,
            flip_prob=args.flip_prob,
            flip_seed=args.flip_seed,
            long_ctx_start=args.long_ctx_start,
            long_ctx_factor=args.long_ctx_factor,
        ),
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
