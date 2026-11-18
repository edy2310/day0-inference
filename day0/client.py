"""Async HTTP client for OpenAI-compatible inference servers.

The client only speaks HTTP and returns typed results. It never computes metrics. Token ids are
always sent as prompts, so every configuration sees byte-identical inputs (§6.1).
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from day0.errors import ServerResponseError
from day0.parsing import TopK, aiter_sse_payloads, parse_completion, parse_prometheus

ERROR_BODY_CHARS = 500
_COMPLETIONS_PATH = "/v1/completions"


@dataclass(frozen=True)
class CompletionResult:
    """Result of one non-streaming completion.

    Attributes:
        output_ids: Generated token ids, or ``None`` when the server did not return them.
        output_text: Generated text.
        finish_reason: ``stop``, ``length``, or ``None``.
        gen_token_logprobs: Logprob of each generated token, or ``None``.
        gen_topk: Top-k distribution per generated position, or ``None``.
        prompt_topk: Top-k distribution per prompt position, index 0 is ``None``, or ``None``.
        prompt_tokens: Prompt length reported by the server.
        completion_tokens: Output length reported by the server.
        raw: The decoded response, kept only when the caller asked for it.
    """

    output_ids: list[int] | None
    output_text: str
    finish_reason: str | None
    gen_token_logprobs: list[float] | None
    gen_topk: list[TopK] | None
    prompt_topk: list[TopK | None] | None
    prompt_tokens: int
    completion_tokens: int
    raw: dict[str, Any] | None


@dataclass(frozen=True)
class StreamResult:
    """Timing and token counts of one streamed completion.

    Attributes:
        ok: Whether the request completed and the prompt length matched.
        error: Error message when ``ok`` is false.
        start_ns: Time the request was sent, from ``perf_counter_ns``.
        first_token_ns: Time the first content chunk was parsed, or ``None``.
        chunk_ns: Parse time of each content chunk.
        chunk_tokens: Output tokens carried by each content chunk.
        output_tokens: Output tokens from the final usage object, or the sum of chunk tokens.
        end_ns: Time the stream ended, from ``perf_counter_ns``.
        chunk_counts_exact: False when chunks carried no cumulative usage, so each chunk was
            counted as one token.
    """

    ok: bool
    error: str | None
    start_ns: int
    first_token_ns: int | None
    chunk_ns: list[int]
    chunk_tokens: list[int]
    output_tokens: int
    end_ns: int
    chunk_counts_exact: bool


class _FixtureRecorder:
    """Writes the first raw response of each request kind to a directory."""

    def __init__(self, directory: Path) -> None:
        """Create the recorder.

        Args:
            directory: Directory that receives one JSON file per request kind.
        """
        self._directory = directory
        self._seen: set[str] = set()

    def record(self, kind: str, payload: object) -> None:
        """Write ``payload`` if this kind has not been recorded yet.

        Args:
            kind: Request kind, used as the file name.
            payload: JSON-compatible response payload.
        """
        if kind in self._seen:
            return
        self._seen.add(kind)
        self._directory.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        (self._directory / f"{kind}.json").write_text(text, encoding="utf-8")


def _failed_stream(start_ns: int, error: str, *, exact: bool) -> StreamResult:
    """Build a failed :class:`StreamResult`.

    Args:
        start_ns: Time the request was sent.
        error: Error message.
        exact: Value for ``chunk_counts_exact``.

    Returns:
        A result with ``ok=False`` and no content chunks.
    """
    return StreamResult(
        ok=False,
        error=error,
        start_ns=start_ns,
        first_token_ns=None,
        chunk_ns=[],
        chunk_tokens=[],
        output_tokens=0,
        end_ns=time.perf_counter_ns(),
        chunk_counts_exact=exact,
    )


class InferenceClient:
    """Client for one OpenAI-compatible server.

    Each instance owns an :class:`httpx.AsyncClient` with a connection limit, so the concurrency of
    the caller is the concurrency seen by the server.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        max_connections: int,
        timeout_s: float = 600.0,
        capture_dir: Path | None = None,
    ) -> None:
        """Create a client.

        Args:
            base_url: Server root, for example ``http://127.0.0.1:8100``.
            model: Served model id used in request bodies.
            max_connections: Connection pool size, and keep-alive size.
            timeout_s: Read timeout in seconds.
            capture_dir: If set, the first raw response of each kind is written here.
        """
        self.model = model
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            limits=httpx.Limits(
                max_connections=max_connections, max_keepalive_connections=max_connections
            ),
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
        self._recorder = _FixtureRecorder(capture_dir) if capture_dir is not None else None

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._http.aclose()

    def _record(self, kind: str, payload: object) -> None:
        """Forward a payload to the fixture recorder, if one is configured.

        Args:
            kind: Request kind.
            payload: JSON-compatible payload.
        """
        if self._recorder is not None:
            self._recorder.record(kind, payload)

    async def _post(self, path: str, body: Mapping[str, Any]) -> httpx.Response:
        """POST a JSON body and wrap transport errors.

        Args:
            path: URL path, relative to the base URL.
            body: JSON body.

        Returns:
            The response, whatever its status code.

        Raises:
            ServerResponseError: If the request could not be sent or completed.
        """
        try:
            return await self._http.post(path, json=dict(body))
        except httpx.HTTPError as exc:
            raise ServerResponseError(f"POST {path} failed: {exc}") from exc

    async def _get(self, path: str) -> httpx.Response:
        """GET a path and wrap transport errors.

        Args:
            path: URL path, relative to the base URL.

        Returns:
            The response, whatever its status code.

        Raises:
            ServerResponseError: If the request could not be sent or completed.
        """
        try:
            return await self._http.get(path)
        except httpx.HTTPError as exc:
            raise ServerResponseError(f"GET {path} failed: {exc}") from exc

    @staticmethod
    def _check(response: httpx.Response, label: str) -> None:
        """Raise if the response status is an error.

        Args:
            response: Response to check.
            label: Request description for the message.

        Raises:
            ServerResponseError: If the status code is 400 or above.
        """
        if response.status_code >= 400:
            body = response.text[:ERROR_BODY_CHARS]
            raise ServerResponseError(
                f"{label} returned HTTP {response.status_code}: {body}",
                status=response.status_code,
            )

    @staticmethod
    def _json_object(response: httpx.Response, label: str) -> dict[str, Any]:
        """Decode a JSON object from a response.

        Args:
            response: Successful response.
            label: Request description for the message.

        Returns:
            The decoded object.

        Raises:
            ServerResponseError: If the body is not a JSON object.
        """
        try:
            payload: object = response.json()
        except ValueError as exc:
            raise ServerResponseError(f"{label} returned a body that is not JSON") from exc
        if not isinstance(payload, dict):
            raise ServerResponseError(f"{label} returned a JSON value that is not an object")
        return payload

    async def health(self) -> bool:
        """Check ``GET /health``.

        Returns:
            True when the server answers 200.
        """
        try:
            response = await self._http.get("/health")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def version(self) -> str | None:
        """Read ``GET /version``.

        Returns:
            The version string, or ``None`` when the endpoint does not exist.
        """
        response = await self._get("/version")
        if response.status_code == 404:
            return None
        self._check(response, "GET /version")
        payload = self._json_object(response, "GET /version")
        version = payload.get("version")
        return str(version) if version is not None else None

    async def tokenize_chat(self, messages: Sequence[Mapping[str, str]]) -> list[int]:
        """Tokenize chat messages with the generation prompt appended.

        Args:
            messages: OpenAI-style ``{"role", "content"}`` objects.

        Returns:
            The token ids.
        """
        body = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "add_generation_prompt": True,
        }
        return await self._tokenize(body, "chat")

    async def tokenize_text(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Tokenize plain text.

        Args:
            text: Text to tokenize.
            add_special_tokens: Whether the server adds special tokens such as BOS.

        Returns:
            The token ids.
        """
        body = {"model": self.model, "prompt": text, "add_special_tokens": add_special_tokens}
        return await self._tokenize(body, "text")

    async def _tokenize(self, body: Mapping[str, Any], kind: str) -> list[int]:
        """Send one ``POST /tokenize`` request.

        Args:
            body: Request body.
            kind: ``chat`` or ``text``, used for fixture names and messages.

        Returns:
            The token ids.

        Raises:
            ServerResponseError: If the response is malformed.
        """
        response = await self._post("/tokenize", body)
        label = f"POST /tokenize ({kind})"
        self._check(response, label)
        payload = self._json_object(response, label)
        self._record(f"tokenize_{kind}", payload)
        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
            raise ServerResponseError(f"{label} returned no integer 'tokens' list")
        return [int(token) for token in tokens]

    async def complete(
        self,
        prompt_ids: Sequence[int],
        *,
        max_tokens: int,
        logprobs: int | None,
        prompt_logprobs: int | None,
        seed: int,
        ignore_eos: bool = False,
        capture_raw: bool = False,
    ) -> CompletionResult:
        """Run one greedy, non-streaming completion on token ids.

        Asserts that the server counted exactly ``len(prompt_ids)`` prompt tokens (§6.1), which
        catches double BOS and similar template bugs.

        Args:
            prompt_ids: Prompt token ids.
            max_tokens: Output budget.
            logprobs: Top-k logprobs for generated tokens, or ``None``.
            prompt_logprobs: Top-k logprobs for prompt positions, or ``None``.
            seed: Request seed.
            ignore_eos: Whether generation continues past EOS.
            capture_raw: Whether to keep the decoded response in the result.

        Returns:
            The parsed result.

        Raises:
            ServerResponseError: On HTTP errors, malformed responses, or a prompt-length mismatch.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": list(prompt_ids),
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "seed": seed,
            "stream": False,
            "ignore_eos": ignore_eos,
        }
        if logprobs is not None:
            body["logprobs"] = logprobs
        if prompt_logprobs is not None:
            body["prompt_logprobs"] = prompt_logprobs
        if logprobs is not None or prompt_logprobs is not None:
            body["return_tokens_as_token_ids"] = True

        response = await self._post(_COMPLETIONS_PATH, body)
        label = f"POST {_COMPLETIONS_PATH}"
        self._check(response, label)
        payload = self._json_object(response, label)
        if prompt_logprobs is not None:
            kind = "completion_prompt_logprobs"
        elif logprobs is not None:
            kind = "completion_logprobs"
        else:
            kind = "completion_plain"
        self._record(kind, payload)

        parsed = parse_completion(payload)
        if parsed.prompt_tokens != len(prompt_ids):
            raise ServerResponseError(
                f"server counted {parsed.prompt_tokens} prompt tokens but {len(prompt_ids)} "
                f"token ids were sent (check for double BOS or template mismatches)"
            )
        return CompletionResult(
            output_ids=parsed.output_ids,
            output_text=parsed.text,
            finish_reason=parsed.finish_reason,
            gen_token_logprobs=parsed.gen_token_logprobs,
            gen_topk=parsed.gen_topk,
            prompt_topk=parsed.prompt_topk,
            prompt_tokens=parsed.prompt_tokens,
            completion_tokens=parsed.completion_tokens,
            raw=payload if capture_raw else None,
        )

    async def stream_complete(
        self,
        prompt_ids: Sequence[int],
        *,
        max_tokens: int,
        seed: int,
        ignore_eos: bool = True,
    ) -> StreamResult:
        """Run one streamed completion and record chunk timings.

        Errors are returned in the result (``ok=False``) rather than raised, so a load generator
        can count them.

        Args:
            prompt_ids: Prompt token ids.
            max_tokens: Output budget.
            seed: Request seed.
            ignore_eos: Whether generation continues past EOS.

        Returns:
            The timings and token counts.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": list(prompt_ids),
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "ignore_eos": ignore_eos,
        }
        start_ns = time.perf_counter_ns()
        try:
            async with self._http.stream("POST", _COMPLETIONS_PATH, json=body) as response:
                if response.status_code >= 400:
                    text = (await response.aread()).decode("utf-8", errors="replace")
                    error = f"HTTP {response.status_code}: {text[:ERROR_BODY_CHARS]}"
                    return _failed_stream(start_ns, error, exact=False)
                return await self._consume_stream(response, start_ns, len(prompt_ids))
        except (httpx.HTTPError, ServerResponseError, ValueError, KeyError) as exc:
            return _failed_stream(start_ns, f"{type(exc).__name__}: {exc}", exact=False)

    async def _consume_stream(
        self, response: httpx.Response, start_ns: int, prompt_len: int
    ) -> StreamResult:
        """Read a streamed response and compute chunk timings and counts.

        Args:
            response: Open streamed response.
            start_ns: Time the request was sent.
            prompt_len: Expected prompt length.

        Returns:
            The stream result.
        """
        chunk_ns: list[int] = []
        chunk_tokens: list[int] = []
        first_ns: int | None = None
        previous_cumulative = 0
        exact = True
        final_usage: Mapping[str, Any] | None = None
        async for payload_text in aiter_sse_payloads(response.aiter_lines()):
            now = time.perf_counter_ns()
            chunk = json.loads(payload_text)
            if not isinstance(chunk, dict):
                raise ServerResponseError("stream chunk is not a JSON object")
            if "error" in chunk:
                raise ServerResponseError(f"stream error: {chunk['error']}")
            choices = chunk.get("choices") or []
            text = str(choices[0].get("text", "")) if choices else ""
            usage = chunk.get("usage")
            if isinstance(usage, Mapping):
                final_usage = usage
            if text:
                if first_ns is None:
                    first_ns = now
                chunk_ns.append(now)
                if isinstance(usage, Mapping):
                    cumulative = int(usage["completion_tokens"])
                    chunk_tokens.append(max(1, cumulative - previous_cumulative))
                    previous_cumulative = cumulative
                else:
                    exact = False
                    chunk_tokens.append(1)
        end_ns = time.perf_counter_ns()

        if final_usage is not None:
            output_tokens = int(final_usage["completion_tokens"])
            if int(final_usage["prompt_tokens"]) != prompt_len:
                return StreamResult(
                    ok=False,
                    error=f"server counted {final_usage['prompt_tokens']} prompt tokens, "
                    f"expected {prompt_len}",
                    start_ns=start_ns,
                    first_token_ns=first_ns,
                    chunk_ns=chunk_ns,
                    chunk_tokens=chunk_tokens,
                    output_tokens=output_tokens,
                    end_ns=end_ns,
                    chunk_counts_exact=exact,
                )
        else:
            output_tokens = sum(chunk_tokens)
        return StreamResult(
            ok=True,
            error=None,
            start_ns=start_ns,
            first_token_ns=first_ns,
            chunk_ns=chunk_ns,
            chunk_tokens=chunk_tokens,
            output_tokens=output_tokens,
            end_ns=end_ns,
            chunk_counts_exact=exact,
        )

    async def metrics_text(self) -> str | None:
        """Read ``GET /metrics`` as text.

        Returns:
            The raw Prometheus text, or ``None`` when the endpoint does not exist.
        """
        response = await self._get("/metrics")
        if response.status_code == 404:
            return None
        self._check(response, "GET /metrics")
        self._record("metrics", {"text": response.text})
        return response.text

    async def metrics(self) -> dict[str, float] | None:
        """Read ``GET /metrics`` and parse it.

        Returns:
            The parsed samples, or ``None`` when the endpoint does not exist.
        """
        text = await self.metrics_text()
        return parse_prometheus(text) if text is not None else None
