"""Closed-loop load generator and perf sweep (§11.2).

For each concurrency level, ``C`` worker coroutines pull requests from a shared queue and send them
one at a time. Warmup requests run first at the same concurrency, and their results are discarded.

On the virtual clock (the fake server), the same workers run as a simulation on the server's
simulated timeline. See ``_simulate_queue``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from day0.client import InferenceClient, StreamResult
from day0.config import Day0Config
from day0.errors import ServerResponseError
from day0.perf.metrics import Clock, level_summary, timing_clock
from day0.perf.workload import PerfRequest
from day0.scrape import cache_usage, delta, snapshot, spec_decode_summary
from day0.seeds import derive_seed


@dataclass(frozen=True)
class LevelRun:
    """Raw outcome of one concurrency level.

    Attributes:
        concurrency: Concurrency of the level.
        results: Measured results, in request order.
        window_s: Measurement window in seconds.
        metrics_before: Server metrics before the level, or ``None``.
        metrics_after: Server metrics after the level, or ``None``.
    """

    concurrency: int
    results: list[StreamResult]
    window_s: float
    metrics_before: dict[str, float] | None
    metrics_after: dict[str, float] | None


@dataclass
class SweepResult:
    """Outcome of a full perf sweep.

    Attributes:
        levels: Level summaries, in configuration order.
        spec_decode: Speculative-decoding summary over the whole sweep, or ``None``.
        cache_usage: Cache gauges after the last level, or an empty dict.
        snapshots: Raw metric snapshots, one entry per level.
        clock: Clock that timed the requests.
    """

    levels: list[dict[str, object]] = field(default_factory=list)
    spec_decode: dict[str, float] | None = None
    cache_usage: dict[str, float] = field(default_factory=dict)
    snapshots: list[dict[str, object]] = field(default_factory=list)
    clock: Clock = "wall"


async def _run_queue(
    client: InferenceClient,
    requests: Sequence[PerfRequest],
    *,
    concurrency: int,
    max_tokens: int,
    seed: int,
) -> dict[tuple[bool, int], StreamResult]:
    """Send requests through ``concurrency`` workers pulling from one queue.

    Args:
        client: Server client.
        requests: Requests to send.
        concurrency: Number of workers.
        max_tokens: Output length, identical for every request.
        seed: Run seed.

    Returns:
        Results keyed by ``(warmup, index)``.
    """
    queue: asyncio.Queue[PerfRequest] = asyncio.Queue()
    for request in requests:
        queue.put_nowait(request)
    results: dict[tuple[bool, int], StreamResult] = {}

    async def worker() -> None:
        while True:
            try:
                request = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            label = "warmup" if request.warmup else "measured"
            results[(request.warmup, request.index)] = await client.stream_complete(
                request.prompt_ids,
                max_tokens=max_tokens,
                seed=derive_seed(seed, "perf-request", request.level, request.index, label),
                ignore_eos=True,
                concurrency=concurrency,
            )

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results


def _on_virtual_clock(result: StreamResult, start_ns: int) -> StreamResult:
    """Move one streamed result onto the simulated timeline of its worker.

    Args:
        result: The result as the client parsed it.
        start_ns: Simulated time at which the worker sent the request.

    Returns:
        The same result, with every timestamp on the simulated timeline.

    Raises:
        ServerResponseError: A successful request carried no simulated timeline. The virtual clock
            works only with the fake server.
    """
    if not result.ok:
        return replace(result, start_ns=start_ns, first_token_ns=None, chunk_ns=[], end_ns=start_ns)
    if result.virtual_ns is None:
        raise ServerResponseError(
            "the virtual clock needs a simulated timeline in every content chunk, and this server "
            "sent none; the virtual clock is for the fake server (backend.kind: fake)"
        )
    chunk_ns = [start_ns + offset for offset in result.virtual_ns]
    return replace(
        result,
        start_ns=start_ns,
        first_token_ns=chunk_ns[0] if chunk_ns else None,
        chunk_ns=chunk_ns,
        end_ns=chunk_ns[-1] if chunk_ns else start_ns,
    )


async def _simulate_queue(
    client: InferenceClient,
    requests: Sequence[PerfRequest],
    *,
    concurrency: int,
    max_tokens: int,
    seed: int,
    free_at_ns: list[int],
) -> dict[tuple[bool, int], StreamResult]:
    """Run requests on ``concurrency`` simulated workers, one request at a time.

    Each request goes to the worker that is free first on the simulated timeline (the lowest index
    on ties), in the order of ``requests``. A closed-loop generator with a shared queue produces
    this schedule when the server's latencies are known. The requests are sent one after another,
    because their simulated timeline does not depend on how they overlap in real time.

    Args:
        client: Server client.
        requests: Requests to send, in queue order.
        concurrency: Number of simulated workers.
        max_tokens: Output length, identical for every request.
        seed: Run seed.
        free_at_ns: Simulated time at which each worker is next free. Updated in place.

    Returns:
        Results keyed by ``(warmup, index)``, with timestamps on the simulated timeline.
    """
    results: dict[tuple[bool, int], StreamResult] = {}
    for request in requests:
        worker = min(range(concurrency), key=lambda index: (free_at_ns[index], index))
        label = "warmup" if request.warmup else "measured"
        raw = await client.stream_complete(
            request.prompt_ids,
            max_tokens=max_tokens,
            seed=derive_seed(seed, "perf-request", request.level, request.index, label),
            ignore_eos=True,
            concurrency=concurrency,
        )
        placed = _on_virtual_clock(raw, free_at_ns[worker])
        free_at_ns[worker] = placed.end_ns
        results[(request.warmup, request.index)] = placed
    return results


async def run_level(
    client: InferenceClient,
    requests: Sequence[PerfRequest],
    *,
    concurrency: int,
    max_tokens: int,
    seed: int,
    clock: Clock = "wall",
) -> LevelRun:
    """Run warmup and measured requests for one concurrency level.

    Args:
        client: Server client.
        requests: All requests of the level, warmup first.
        concurrency: Closed-loop concurrency.
        max_tokens: Output length.
        seed: Run seed.
        clock: ``"wall"`` runs the workers concurrently and times them with the wall clock.
            ``"virtual"`` runs them as a simulation on the server's simulated timeline.

    Returns:
        The measured results and the measurement window.
    """
    warmup = [r for r in requests if r.warmup]
    measured = [r for r in requests if not r.warmup]
    before = await snapshot(client)
    if clock == "virtual":
        free_at = [0] * concurrency
        await _simulate_queue(
            client,
            warmup,
            concurrency=concurrency,
            max_tokens=max_tokens,
            seed=seed,
            free_at_ns=free_at,
        )
        # Measurement starts once the last warmup request has finished, as in the wall-clock run.
        free_at = [max(free_at)] * concurrency
        by_key = await _simulate_queue(
            client,
            measured,
            concurrency=concurrency,
            max_tokens=max_tokens,
            seed=seed,
            free_at_ns=free_at,
        )
    else:
        await _run_queue(client, warmup, concurrency=concurrency, max_tokens=max_tokens, seed=seed)
        by_key = await _run_queue(
            client, measured, concurrency=concurrency, max_tokens=max_tokens, seed=seed
        )
    after = await snapshot(client)
    ordered = [by_key[(False, r.index)] for r in measured]
    if ordered:
        start = min(result.start_ns for result in ordered)
        end = max(result.end_ns for result in ordered)
        window_s = max(end - start, 0) / 1e9
    else:
        window_s = 0.0
    return LevelRun(
        concurrency=concurrency,
        results=ordered,
        window_s=window_s,
        metrics_before=before,
        metrics_after=after,
    )


async def run_sweep(
    client: InferenceClient,
    workload: Sequence[PerfRequest],
    cfg: Day0Config,
    *,
    seed: int,
) -> SweepResult:
    """Sweep every concurrency level in the configuration.

    Args:
        client: Server client.
        workload: Requests for every level, as built by ``build_workload`` or loaded from storage.
        cfg: Validated configuration.
        seed: Run seed.

    Returns:
        Level summaries, speculative-decoding summary, and raw snapshots.
    """
    clock = timing_clock(cfg.backend.kind)
    result = SweepResult(clock=clock)
    deltas: list[dict[str, float]] = []
    last_after: dict[str, float] | None = None
    for level in cfg.perf.concurrency_levels:
        requests = [r for r in workload if r.level == level]
        run = await run_level(
            client,
            requests,
            concurrency=level,
            max_tokens=cfg.perf.output_len,
            seed=seed,
            clock=clock,
        )
        summary = level_summary(
            concurrency=level,
            results=run.results,
            window_s=run.window_s,
            slo=cfg.slo,
            hardware=cfg.hardware,
            max_error_rate=cfg.perf.max_error_rate,
        )
        level_spec: dict[str, float] | None = None
        if run.metrics_before is not None and run.metrics_after is not None:
            level_delta = delta(run.metrics_before, run.metrics_after)
            deltas.append(level_delta)
            spec = spec_decode_summary(level_delta)
            level_spec = dict(spec.__dict__) if spec is not None else None
            last_after = run.metrics_after
        summary["spec_decode"] = level_spec
        result.levels.append(summary)
        result.snapshots.append(
            {
                "concurrency": level,
                "before": run.metrics_before,
                "after": run.metrics_after,
            }
        )

    if deltas:
        total: dict[str, float] = {}
        for item in deltas:
            for key, value in item.items():
                total[key] = total.get(key, 0.0) + value
        spec_total = spec_decode_summary(total)
        result.spec_decode = dict(spec_total.__dict__) if spec_total is not None else None
    if last_after is not None:
        result.cache_usage = cache_usage(last_after)
    return result


def perf_document(sweep: SweepResult, cfg: Day0Config) -> dict[str, object]:
    """Build the ``perf.json`` document for one configuration.

    The target summary is the level at ``perf.target_concurrency``. The maximum concurrency that
    meets the SLO is the largest level whose ``slo_met`` is true, or ``None``.

    Args:
        sweep: Outcome of :func:`run_sweep`.
        cfg: Validated configuration.

    Returns:
        A JSON-compatible document.
    """
    target: dict[str, object] | None = None
    meeting: list[int] = []
    for level in sweep.levels:
        if level["concurrency"] == cfg.perf.target_concurrency:
            target = level
        if level["slo_met"] is True:
            concurrency = level["concurrency"]
            if isinstance(concurrency, int):
                meeting.append(concurrency)
    return {
        "schema_version": 1,
        "target_concurrency": cfg.perf.target_concurrency,
        "levels": sweep.levels,
        "target": target,
        "max_concurrency_meeting_slo": max(meeting) if meeting else None,
        "spec_decode": sweep.spec_decode,
        "cache_usage": sweep.cache_usage,
        "warmup_requests": cfg.perf.warmup_requests,
        "requests_per_level": cfg.perf.requests_per_level,
        "clock": sweep.clock,
    }
