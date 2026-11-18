"""GPU smoke test against a real vLLM server. Skipped unless ``DAY0_GPU_TESTS=1``.

It starts ``vllm serve`` with a small ungated model and checks the protocol assumptions that
day0 relies on (§19.3): token-id prompts are counted exactly, ``prompt_logprobs`` has the expected
shape and contains the actual token, ``return_tokens_as_token_ids`` parses, and streamed chunks
carry cumulative usage.

The server is started once per module. Each test creates its own client inside its own
``asyncio.run``, because an httpx client must not outlive the event loop that used it.

Run it on a GPU host:  DAY0_GPU_TESTS=1 .venv/bin/pytest -q tests/gpu
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest

from day0.backends import VLLMBackend
from day0.backends.base import BackendContext, ServerHandle
from day0.client import InferenceClient
from day0.config import BackendConfig, ServerConfig
from day0.parsing import parse_completion

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("DAY0_GPU_TESTS") != "1", reason="set DAY0_GPU_TESTS=1 on a GPU host"
    ),
    pytest.mark.skipif(shutil.which("vllm") is None, reason="vllm executable not on PATH"),
]
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TOP_K = 5
Action = Callable[[InferenceClient], Awaitable[None]]


def _free_port() -> int:
    """Return a TCP port that was free when queried.

    Returns:
        The port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _in_client(handle: ServerHandle, action: Action) -> None:
    """Run one action with a fresh client, inside its own event loop.

    Args:
        handle: Handle of the running server.
        action: Coroutine function that receives the client.
    """

    async def run() -> None:
        client = InferenceClient(handle.base_url, handle.served_model, max_connections=4)
        try:
            await action(client)
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.fixture(scope="module")
def vllm_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ServerHandle]:
    """Start one vLLM server for the module and stop it afterwards.

    Args:
        tmp_path_factory: Factory for module-scoped temporary directories.

    Yields:
        The handle of the running server.
    """
    cfg = BackendConfig.model_validate(
        {
            "kind": "vllm",
            "executable": "vllm",
            "port": _free_port(),
            "startup_timeout_s": 900,
            "cooldown_s": 0,
        }
    )
    context = BackendContext(
        model_id=SMOKE_MODEL, max_model_len=4096, seed=1234, max_logprobs=TOP_K
    )
    backend = VLLMBackend(cfg, context)
    log = tmp_path_factory.mktemp("vllm") / "server.log"
    handle = asyncio.run(backend.start(ServerConfig(name="smoke"), log))
    try:
        yield handle
    finally:
        asyncio.run(backend.stop(handle))


def test_endpoints_answer(vllm_server: ServerHandle) -> None:
    """Health, version, and metrics endpoints respond on the real server.

    Args:
        vllm_server: Handle of the smoke server.
    """

    async def check(client: InferenceClient) -> None:
        assert await client.health()
        assert await client.version() is not None
        assert await client.metrics_text() is not None

    _in_client(vllm_server, check)


def test_token_id_prompts_are_counted_exactly(vllm_server: ServerHandle) -> None:
    """The server counts exactly the token ids that were sent (no double BOS).

    Args:
        vllm_server: Handle of the smoke server.
    """

    async def check(client: InferenceClient) -> None:
        ids = await client.tokenize_chat([{"role": "user", "content": "Say hi."}])
        result = await client.complete(
            ids, max_tokens=4, logprobs=None, prompt_logprobs=None, seed=1
        )
        assert result.prompt_tokens == len(ids)

    _in_client(vllm_server, check)


def test_prompt_logprobs_shape_and_actual_token(vllm_server: ServerHandle) -> None:
    """Position 0 is null, each position has the actual token, and there are at most k + 1 entries.

    Args:
        vllm_server: Handle of the smoke server.
    """

    async def check(client: InferenceClient) -> None:
        ids = await client.tokenize_text("The quick brown fox.", add_special_tokens=True)
        result = await client.complete(
            ids, max_tokens=1, logprobs=None, prompt_logprobs=TOP_K, seed=1
        )
        rows = result.prompt_topk
        assert rows is not None and rows[0] is None
        assert len(rows) == len(ids)
        for position in range(1, len(ids)):
            row = rows[position]
            assert row is not None
            assert ids[position] in row
            assert len(row) <= TOP_K + 1

    _in_client(vllm_server, check)


def test_generation_top_k_and_token_ids_parse(vllm_server: ServerHandle) -> None:
    """With ``return_tokens_as_token_ids`` the output ids and top-k parse as integers.

    Args:
        vllm_server: Handle of the smoke server.
    """

    async def check(client: InferenceClient) -> None:
        ids = await client.tokenize_chat([{"role": "user", "content": "Name a colour."}])
        result = await client.complete(
            ids, max_tokens=6, logprobs=TOP_K, prompt_logprobs=None, seed=1
        )
        assert result.output_ids is not None
        assert result.gen_topk is not None
        assert len(result.gen_topk) == len(result.output_ids)

    _in_client(vllm_server, check)


def test_streaming_usage_is_cumulative(vllm_server: ServerHandle) -> None:
    """With ``continuous_usage_stats`` the streamed chunk counts are exact.

    Args:
        vllm_server: Handle of the smoke server.
    """

    async def check(client: InferenceClient) -> None:
        ids = await client.tokenize_text("Count to ten.", add_special_tokens=True)
        result = await client.stream_complete(ids, max_tokens=16, seed=1, ignore_eos=True)
        assert result.ok, result.error
        assert result.chunk_counts_exact
        assert sum(result.chunk_tokens) == result.output_tokens

    _in_client(vllm_server, check)


def test_fixtures_are_consistent_when_present(repo_root: Path) -> None:
    """Captured fixtures, when committed, parse to consistent lengths.

    Args:
        repo_root: Repository root.
    """
    fixtures = repo_root / "tests" / "fixtures" / "vllm"
    captured = sorted(fixtures.glob("completion_prompt_logprobs.json"))
    if not captured:
        pytest.skip("no captured fixtures in tests/fixtures/vllm (see docs/runbook_gpu.md)")
    payload = json.loads(captured[0].read_text(encoding="utf-8"))
    parsed = parse_completion(payload)
    assert parsed.prompt_topk is not None
    assert len(parsed.prompt_topk) == parsed.prompt_tokens
