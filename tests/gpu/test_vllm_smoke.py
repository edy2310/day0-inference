"""GPU smoke test against a real vLLM server. Skipped unless ``DAY0_GPU_TESTS=1``.

It starts ``vllm serve`` with a small ungated model and checks the protocol assumptions that
day0 relies on (§19.3): token-id prompts are counted exactly, ``prompt_logprobs`` has the expected
shape and contains the actual token, ``return_tokens_as_token_ids`` parses, and streamed chunks
carry cumulative usage.

Run it on a GPU host:  DAY0_GPU_TESTS=1 .venv/bin/pytest -q tests/gpu
"""

from __future__ import annotations

import os
import shutil
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest

from day0.backends import VLLMBackend
from day0.backends.base import BackendContext
from day0.client import InferenceClient
from day0.config import BackendConfig, ServerConfig

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("DAY0_GPU_TESTS") != "1", reason="set DAY0_GPU_TESTS=1 on a GPU host"
    ),
    pytest.mark.skipif(shutil.which("vllm") is None, reason="vllm executable not on PATH"),
]
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TOP_K = 5


def _free_port() -> int:
    """Return a TCP port that was free when queried.

    Returns:
        The port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="module")
def vllm_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[InferenceClient]:
    """Start a vLLM server for the module and yield a client for it.

    Args:
        tmp_path_factory: Factory for module-scoped temporary directories.

    Yields:
        A client connected to the server.
    """
    port = _free_port()
    cfg = BackendConfig.model_validate(
        {
            "kind": "vllm",
            "executable": "vllm",
            "port": port,
            "startup_timeout_s": 900,
            "cooldown_s": 0,
        }
    )
    context = BackendContext(
        model_id=SMOKE_MODEL, max_model_len=4096, seed=1234, max_logprobs=TOP_K
    )
    backend = VLLMBackend(cfg, context)
    log = tmp_path_factory.mktemp("vllm") / "server.log"
    import asyncio

    handle = asyncio.run(backend.start(ServerConfig(name="smoke"), log))
    client = InferenceClient(handle.base_url, handle.served_model, max_connections=4)
    try:
        yield client
    finally:
        asyncio.run(client.aclose())
        asyncio.run(backend.stop(handle))


def test_token_id_prompts_are_counted_exactly(vllm_client: InferenceClient) -> None:
    """The server counts exactly the token ids that were sent (no double BOS).

    Args:
        vllm_client: Client connected to the smoke server.
    """
    import asyncio

    async def run() -> None:
        ids = await vllm_client.tokenize_chat([{"role": "user", "content": "Say hi."}])
        result = await vllm_client.complete(
            ids, max_tokens=4, logprobs=None, prompt_logprobs=None, seed=1
        )
        assert result.prompt_tokens == len(ids)

    asyncio.run(run())


def test_prompt_logprobs_shape_and_actual_token(vllm_client: InferenceClient) -> None:
    """Position 0 is null, each position has the actual token, and there are at most k + 1 entries.

    Args:
        vllm_client: Client connected to the smoke server.
    """
    import asyncio

    async def run() -> None:
        ids = await vllm_client.tokenize_text("The quick brown fox.", add_special_tokens=True)
        result = await vllm_client.complete(
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

    asyncio.run(run())


def test_generation_top_k_and_token_ids_parse(vllm_client: InferenceClient) -> None:
    """With ``return_tokens_as_token_ids`` the output ids and top-k parse as integers.

    Args:
        vllm_client: Client connected to the smoke server.
    """
    import asyncio

    async def run() -> None:
        ids = await vllm_client.tokenize_chat([{"role": "user", "content": "Name a colour."}])
        result = await vllm_client.complete(
            ids, max_tokens=6, logprobs=TOP_K, prompt_logprobs=None, seed=1
        )
        assert result.output_ids is not None
        assert result.gen_topk is not None
        assert len(result.gen_topk) == len(result.output_ids)

    asyncio.run(run())


def test_streaming_usage_is_cumulative(vllm_client: InferenceClient) -> None:
    """With ``continuous_usage_stats`` the streamed chunk counts are exact.

    Args:
        vllm_client: Client connected to the smoke server.
    """
    import asyncio

    async def run() -> None:
        ids = await vllm_client.tokenize_text("Count to ten.", add_special_tokens=True)
        result = await vllm_client.stream_complete(ids, max_tokens=16, seed=1, ignore_eos=True)
        assert result.ok, result.error
        assert result.chunk_counts_exact
        assert sum(result.chunk_tokens) == result.output_tokens

    asyncio.run(run())


def test_fixtures_are_consistent_when_present(repo_root: Path) -> None:
    """Captured fixtures, when committed, parse to consistent lengths.

    Args:
        repo_root: Repository root.
    """
    import json

    from day0.parsing import parse_completion

    fixtures = repo_root / "tests" / "fixtures" / "vllm"
    captured = sorted(fixtures.glob("completion_prompt_logprobs.json"))
    if not captured:
        pytest.skip("no captured fixtures in tests/fixtures/vllm (see docs/runbook_gpu.md)")
    payload = json.loads(captured[0].read_text(encoding="utf-8"))
    parsed = parse_completion(payload)
    assert parsed.prompt_topk is not None
    assert len(parsed.prompt_topk) == parsed.prompt_tokens
