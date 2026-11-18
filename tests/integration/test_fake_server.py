"""Integration tests: run the fake server as a subprocess and talk to it over HTTP."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator

import httpx
import pytest

from day0.client import InferenceClient
from day0.fake.model import EOS_ID, ModelParams, decode, plan_turn

ServerStarter = Callable[..., str]


def _free_port() -> int:
    """Ask the OS for an unused TCP port.

    Returns:
        A port number that was free when queried.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture
def start_fake_server() -> Iterator[ServerStarter]:
    """Start fake servers with given arguments and stop them after the test.

    Yields:
        A function that takes server arguments and returns the base URL once ``/health`` is 200.
    """
    processes: list[subprocess.Popen[bytes]] = []

    def _start(*args: str) -> str:
        port = _free_port()
        command = [
            sys.executable,
            "-m",
            "day0.fake.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            *args,
        ]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(process)
        base_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"fake server exited early with code {process.returncode}")
            try:
                if httpx.get(f"{base_url}/health", timeout=1.0).status_code == 200:
                    return base_url
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise RuntimeError("fake server did not become healthy in time")

    yield _start
    for process in processes:
        process.terminate()
        process.wait(timeout=10)


def _client(base_url: str) -> InferenceClient:
    """Create a client for the fake model.

    Args:
        base_url: Server root.

    Returns:
        A client with a small connection pool.
    """
    return InferenceClient(base_url, "fake-model", max_connections=4)


async def test_basic_endpoints(start_fake_server: ServerStarter) -> None:
    """Health, version, models, tokenize, and metrics answer in the expected shapes.

    Args:
        start_fake_server: Fixture that starts servers.
    """
    base_url = start_fake_server()
    client = _client(base_url)
    try:
        assert await client.health()
        assert await client.version() == "fake-0.1"
        ids = await client.tokenize_text("abc", add_special_tokens=True)
        assert ids == [256, 97, 98, 99]
        chat = await client.tokenize_chat([{"role": "user", "content": "hi"}])
        assert chat[0] == 256
        assert decode(chat) == "<|user|>\nhi\n<|assistant|>\n"
        metrics = await client.metrics()
        assert metrics is not None
        assert any(key.startswith("vllm:num_requests_running") for key in metrics)
        assert not any("spec_decode" in key for key in metrics)
    finally:
        await client.aclose()


async def test_completion_prompt_tokens_match_request(start_fake_server: ServerStarter) -> None:
    """The server counts exactly the token ids that were sent.

    Args:
        start_fake_server: Fixture that starts servers.
    """
    client = _client(start_fake_server())
    try:
        prompt = await client.tokenize_text("hello world", add_special_tokens=True)
        result = await client.complete(
            prompt, max_tokens=8, logprobs=5, prompt_logprobs=None, seed=1
        )
        assert result.prompt_tokens == len(prompt)
        assert result.output_ids is not None
        assert len(result.output_ids) == len(result.gen_topk or [])
        assert all(len(row) <= 5 for row in result.gen_topk or [])
    finally:
        await client.aclose()


async def test_prompt_logprobs_have_null_first_and_actual_token(
    start_fake_server: ServerStarter,
) -> None:
    """Every prompt position contains the actual next token and at most k + 1 entries.

    Args:
        start_fake_server: Fixture that starts servers.
    """
    k = 4
    client = _client(start_fake_server())
    try:
        prompt = await client.tokenize_text("teacher forced", add_special_tokens=True)
        result = await client.complete(
            prompt, max_tokens=1, logprobs=None, prompt_logprobs=k, seed=1
        )
        rows = result.prompt_topk
        assert rows is not None
        assert rows[0] is None
        assert len(rows) == len(prompt)
        for position in range(1, len(prompt)):
            row = rows[position]
            assert row is not None
            assert prompt[position] in row
            assert len(row) <= k + 1
    finally:
        await client.aclose()


async def test_identical_config_gives_identical_logprobs(start_fake_server: ServerStarter) -> None:
    """With zero noise, two calls return bit-identical logprobs.

    Args:
        start_fake_server: Fixture that starts servers.
    """
    client = _client(start_fake_server("--noise", "0.0"))
    try:
        prompt = await client.tokenize_text("determinism", add_special_tokens=True)
        first = await client.complete(prompt, max_tokens=6, logprobs=5, prompt_logprobs=5, seed=3)
        second = await client.complete(prompt, max_tokens=6, logprobs=5, prompt_logprobs=5, seed=3)
        assert first.gen_token_logprobs == second.gen_token_logprobs
        assert first.prompt_topk == second.prompt_topk
    finally:
        await client.aclose()


async def test_rejects_sampling_and_logprob_limits(start_fake_server: ServerStarter) -> None:
    """Temperature above zero and logprobs above the limit are rejected with HTTP 400.

    Args:
        start_fake_server: Fixture that starts servers.
    """
    base_url = start_fake_server("--max-logprobs", "5")
    async with httpx.AsyncClient(base_url=base_url) as http:
        sampled = await http.post(
            "/v1/completions",
            json={"model": "fake-model", "prompt": [256, 97], "max_tokens": 2, "temperature": 0.7},
        )
        assert sampled.status_code == 400
        too_many = await http.post(
            "/v1/completions",
            json={"model": "fake-model", "prompt": [256, 97], "max_tokens": 2, "logprobs": 9},
        )
        assert too_many.status_code == 400
        assert "logprobs" in too_many.text


async def test_echo_marker_forces_answer(start_fake_server: ServerStarter) -> None:
    """The echo rule produces "The answer is N." and then stops at EOS.

    Args:
        start_fake_server: Fixture that starts servers.
    """
    client = _client(start_fake_server())
    try:
        messages = [{"role": "user", "content": "Count. <<ECHO:42>>"}]
        prompt = await client.tokenize_chat(messages)
        result = await client.complete(
            prompt, max_tokens=48, logprobs=3, prompt_logprobs=None, seed=1
        )
        assert result.output_text == "The answer is 42."
        assert result.finish_reason == "stop"
    finally:
        await client.aclose()


async def test_flip_probability_changes_echo_answer(start_fake_server: ServerStarter) -> None:
    """With flip probability 1 the echoed answer is always off by one.

    Args:
        start_fake_server: Fixture that starts servers.
    """
    client = _client(start_fake_server("--flip-prob", "1.0"))
    try:
        prompt = await client.tokenize_chat([{"role": "user", "content": "<<ECHO:7>>"}])
        result = await client.complete(
            prompt, max_tokens=48, logprobs=None, prompt_logprobs=None, seed=1
        )
        assert result.output_text == "The answer is 8."
    finally:
        await client.aclose()


async def test_streaming_counts_tokens_exactly(start_fake_server: ServerStarter) -> None:
    """Streamed chunks carry one token each, and the counts match the final usage.

    Args:
        start_fake_server: Fixture that starts servers.
    """
    client = _client(start_fake_server())
    try:
        prompt = await client.tokenize_text("stream me", add_special_tokens=True)
        result = await client.stream_complete(prompt, max_tokens=12, seed=1, ignore_eos=True)
        assert result.ok, result.error
        assert result.chunk_counts_exact
        assert sum(result.chunk_tokens) == result.output_tokens == 12
        assert len(result.chunk_ns) == 12
        assert result.first_token_ns is not None
        assert result.first_token_ns <= result.chunk_ns[-1] <= result.end_ns
    finally:
        await client.aclose()


async def test_stream_error_is_reported_not_raised(start_fake_server: ServerStarter) -> None:
    """A rejected streamed request returns ``ok=False`` with the server message.

    Args:
        start_fake_server: Fixture that starts servers.
    """
    client = _client(start_fake_server())
    try:
        too_long = [256] * 70000
        result = await client.stream_complete(too_long, max_tokens=4, seed=1)
        assert not result.ok
        assert result.error is not None
        assert "maximum context length" in result.error
    finally:
        await client.aclose()


async def test_speculative_counters_are_exposed(start_fake_server: ServerStarter) -> None:
    """With ``--spec-accept-rate`` the three counters appear on ``/metrics`` after generation.

    Args:
        start_fake_server: Fixture that starts servers.
    """
    client = _client(start_fake_server("--spec-accept-rate", "0.6"))
    try:
        prompt = await client.tokenize_text("spec", add_special_tokens=True)
        await client.complete(prompt, max_tokens=20, logprobs=None, prompt_logprobs=None, seed=1)
        metrics = await client.metrics()
        assert metrics is not None
        drafts = [v for k, v in metrics.items() if "num_drafts_total" in k]
        accepted = [v for k, v in metrics.items() if "num_accepted_tokens_total" in k]
        assert drafts and drafts[0] > 0
        assert accepted and accepted[0] > 0
    finally:
        await client.aclose()


def test_startup_delay_answers_503_then_200(start_fake_server: ServerStarter) -> None:
    """``/health`` is 503 for the configured delay. The fixture waits until it is 200.

    Args:
        start_fake_server: Fixture that starts servers.
    """
    base_url = start_fake_server("--startup-delay-s", "0.5")
    response = httpx.get(f"{base_url}/health", timeout=2.0)
    assert response.status_code == 200


async def test_stream_chunks_carry_the_simulated_timeline(start_fake_server: ServerStarter) -> None:
    """Each content chunk carries its simulated time: time to first token, then one gap per token.

    Args:
        start_fake_server: Fixture that starts servers.
    """
    base_url = start_fake_server(
        "--ttft-base-ms",
        "10",
        "--prefill-ms-per-1k",
        "0",
        "--itl-base-ms",
        "4",
        "--concurrency-penalty",
        "0",
        "--speedup",
        "1",
    )
    client = _client(base_url)
    try:
        prompt = await client.tokenize_text("timed", add_special_tokens=True)
        result = await client.stream_complete(
            prompt, max_tokens=5, seed=1, ignore_eos=True, concurrency=1
        )
        assert result.ok, result.error
        assert result.virtual_ns == [10_000_000 + k * 4_000_000 for k in range(5)]
    finally:
        await client.aclose()


def test_concurrency_header_must_be_a_positive_integer(start_fake_server: ServerStarter) -> None:
    """A malformed ``x-day0-concurrency`` header is rejected, not replaced by a default.

    Args:
        start_fake_server: Fixture that starts servers.
    """
    base_url = start_fake_server()
    body = {"model": "fake-model", "prompt": [72, 105], "max_tokens": 2}
    for value in ("0", "many"):
        response = httpx.post(
            f"{base_url}/v1/completions",
            json=body,
            headers={"x-day0-concurrency": value},
            timeout=5.0,
        )
        assert response.status_code == 400, value
    accepted = httpx.post(
        f"{base_url}/v1/completions", json=body, headers={"x-day0-concurrency": "4"}, timeout=5.0
    )
    assert accepted.status_code == 200


def test_echo_plan_is_consistent_for_prompt_and_output() -> None:
    """Planning on the prompt alone and on prompt plus output gives the same turn.

    The forced answer is a property of the prompt, so teacher-forced scoring of a generated output
    sees the same context as generation did.
    """
    params = ModelParams()
    prompt: list[int] = [256, *b"<|user|>\n<<ECHO:5>>\n<|assistant|>\n"]
    output: list[int] = [*b"The answer is 5.", EOS_ID]
    alone = plan_turn(prompt, params)
    combined = plan_turn([*prompt, *output], params)
    assert alone.assistant_end == combined.assistant_end
    assert alone.forced == combined.forced
