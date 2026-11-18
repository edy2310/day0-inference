"""The fake server's simulated timeline, and the virtual-clock load generator that runs on it."""

from __future__ import annotations

import pytest

from day0.client import StreamResult
from day0.errors import ServerResponseError
from day0.fake.server import FakeServerConfig, virtual_timing
from day0.perf.metrics import timing_clock
from day0.perf.runner import run_level
from day0.perf.workload import PerfRequest


def _stream(duration_ns: int, *, virtual: bool = True) -> StreamResult:
    """One successful one-token stream whose only token arrives after ``duration_ns``.

    Args:
        duration_ns: Simulated time of the token, from the start of the request.
        virtual: Whether the result carries a simulated timeline.

    Returns:
        The stream result, with wall-clock fields left at zero.
    """
    return StreamResult(
        ok=True,
        error=None,
        start_ns=0,
        first_token_ns=0,
        chunk_ns=[0],
        chunk_tokens=[1],
        output_tokens=1,
        end_ns=0,
        chunk_counts_exact=True,
        virtual_ns=[duration_ns] if virtual else None,
    )


class _SimulatedClient:
    """Stand-in for the client: each prompt length maps to one fixed simulated duration."""

    def __init__(self, durations_ms: dict[int, float], *, virtual: bool = True) -> None:
        self._durations_ns = {length: round(ms * 1e6) for length, ms in durations_ms.items()}
        self._virtual = virtual
        self.concurrency_seen: list[int | None] = []

    async def stream_complete(
        self,
        prompt_ids: list[int],
        *,
        max_tokens: int,
        seed: int,
        ignore_eos: bool = True,
        concurrency: int | None = None,
    ) -> StreamResult:
        """Return the fixed stream for this prompt length.

        Args:
            prompt_ids: Prompt token ids; only their count matters here.
            max_tokens: Unused.
            seed: Unused.
            ignore_eos: Unused.
            concurrency: Recorded, so the test can check the header value.

        Returns:
            A one-token stream.
        """
        del max_tokens, seed, ignore_eos
        self.concurrency_seen.append(concurrency)
        return _stream(self._durations_ns[len(prompt_ids)], virtual=self._virtual)

    async def metrics(self) -> None:
        """Report that the endpoint is absent, so the level has no speculative-decoding data."""


def _requests(
    lengths: list[int], *, warmup: bool = False, first_index: int = 0
) -> list[PerfRequest]:
    """Requests with the given prompt lengths, in order.

    Args:
        lengths: Prompt length of each request.
        warmup: Whether the requests are warmup requests.
        first_index: Index of the first request.

    Returns:
        The requests.
    """
    return [
        PerfRequest(level=2, index=first_index + i, warmup=warmup, prompt_ids=[0] * length)
        for i, length in enumerate(lengths)
    ]


def test_virtual_timing_follows_the_specified_formulas() -> None:
    """Time to first token and time between tokens are the §13.3 values, in whole nanoseconds."""
    config = FakeServerConfig(speedup=2.0, concurrency_penalty=0.0)
    # Prompt of 0 tokens: TTFT = 20 ms / 2, ITL = 2 ms / 2.
    assert virtual_timing(config, prompt_len=0, active=1) == (10_000_000, 1_000_000)
    # 1000 prompt tokens add 5 ms before scaling.
    assert virtual_timing(config, prompt_len=1000, active=1) == (12_500_000, 1_000_000)
    penalised = FakeServerConfig(speedup=1.0, concurrency_penalty=0.02)
    # Concurrency 11 gives a slowdown of 1 + 0.02 * 11 = 1.22.
    assert virtual_timing(penalised, prompt_len=0, active=11) == (24_400_000, 2_440_000)


def test_timing_clock_is_virtual_only_for_the_fake_server() -> None:
    """The fake server is timed on its simulated clock; every other backend on the wall clock."""
    assert timing_clock("fake") == "virtual"
    assert timing_clock("vllm") == "wall"
    assert timing_clock("external") == "wall"


async def test_requests_go_to_the_worker_that_is_free_first() -> None:
    """With two workers, each request starts on the worker that finished its predecessor first."""
    # Durations in ms, keyed by prompt length: the first request is long, the others short.
    client = _SimulatedClient({10: 3.0, 11: 1.0, 12: 1.0, 13: 2.0})
    run = await run_level(
        client,
        _requests([10, 11, 12, 13]),
        concurrency=2,
        max_tokens=1,
        seed=1,
        clock="virtual",
    )
    # Worker 0 runs request 0 over [0, 3]. Worker 1 runs requests 1, 2, and 3 back to back.
    starts = [result.start_ns for result in run.results]
    assert starts == [0, 0, 1_000_000, 2_000_000]
    assert run.window_s == pytest.approx(0.004)
    assert client.concurrency_seen == [2, 2, 2, 2]


async def test_measurement_starts_after_the_warmup_has_finished() -> None:
    """Warmup requests advance the simulated clock, and measurement begins when they all finish."""
    client = _SimulatedClient({13: 2.0, 10: 3.0, 11: 1.0, 12: 1.0})
    warmup = _requests([13], warmup=True, first_index=0)
    measured = _requests([10, 11, 12, 13], first_index=1)
    run = await run_level(
        client,
        warmup + measured,
        concurrency=2,
        max_tokens=1,
        seed=1,
        clock="virtual",
    )
    # Warmup ends at 2 ms. Measured request 0 then runs over [2, 5] on worker 0, and the window
    # spans [2, 6], four milliseconds long.
    assert min(result.start_ns for result in run.results) == 2_000_000
    assert run.window_s == pytest.approx(0.004)


async def test_virtual_clock_refuses_a_server_without_a_simulated_timeline() -> None:
    """A successful request with no simulated timeline is an error, not a silent wall-clock run."""
    client = _SimulatedClient({10: 1.0}, virtual=False)
    with pytest.raises(ServerResponseError, match="virtual clock"):
        await run_level(
            client,
            _requests([10]),
            concurrency=1,
            max_tokens=1,
            seed=1,
            clock="virtual",
        )
