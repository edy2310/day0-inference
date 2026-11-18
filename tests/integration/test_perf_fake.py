"""Perf-only run against the fake server: plausible latencies and a spec-decode summary."""

from __future__ import annotations

import socket
from pathlib import Path

from day0.backends import FakeBackend
from day0.backends.base import BackendContext
from day0.client import InferenceClient
from day0.config import Day0Config, LengthDist, ServerConfig, load_config
from day0.perf.runner import run_sweep
from day0.perf.workload import build_workload, requests_from_rows, workload_rows

FAKE_TTFT_BASE_MS = 20.0


def _free_port() -> int:
    """Return a TCP port that was free when queried.

    Returns:
        The port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _small_config(examples_dir: Path) -> Day0Config:
    """Load the fake example and shrink the perf sweep so the test stays fast.

    Args:
        examples_dir: Directory with the example configurations.

    Returns:
        A configuration with a short sweep.
    """
    cfg = load_config(examples_dir / "fake.yaml")
    perf = cfg.perf.model_copy(
        update={
            "concurrency_levels": [1, 4],
            "target_concurrency": 4,
            "requests_per_level": 6,
            "warmup_requests": 2,
            "input_len": LengthDist(dist="uniform", min=64, max=128),
            "output_len": 8,
        }
    )
    return cfg.model_copy(update={"perf": perf})


async def test_perf_sweep_with_fake_server(examples_dir: Path, tmp_path: Path) -> None:
    """The sweep produces one summary per level, with TTFT at or above the fixed base latency.

    The workload is serialized and read back, which is how candidates reuse the reference prompts.

    Args:
        examples_dir: Directory with the example configurations.
        tmp_path: Pytest's per-test temporary directory.
    """
    cfg = _small_config(examples_dir)
    context = BackendContext(
        model_id=cfg.model.id,
        max_model_len=cfg.model.max_model_len,
        seed=cfg.run.seed,
        max_logprobs=cfg.quality.top_k_logprobs,
    )
    port = _free_port()
    backend = FakeBackend(cfg.backend.model_copy(update={"port": port}), context)
    handle = await backend.start(
        ServerConfig(name="spec", args=["--spec-accept-rate", "0.6"]),
        tmp_path / "server.log",
    )
    client = InferenceClient(handle.base_url, handle.served_model, max_connections=8)
    try:
        built = await build_workload(client, cfg, seed=cfg.run.seed)
        reloaded = requests_from_rows(workload_rows(built))
        assert len(reloaded) == len(built)
        sweep = await run_sweep(client, reloaded, cfg, seed=cfg.run.seed)
    finally:
        await client.aclose()
        await backend.stop(handle)

    assert [level["concurrency"] for level in sweep.levels] == [1, 4]
    for level in sweep.levels:
        assert level["n_ok"] == 6
        assert level["invalid"] is False
        ttft = level["ttft_ms"]
        assert isinstance(ttft, dict)
        # The fake server's fixed time to first token is 20 ms, so no measured p50 is below it.
        assert ttft["p50"] is not None and ttft["p50"] >= FAKE_TTFT_BASE_MS
        output_rate = level["output_tok_per_s"]
        assert isinstance(output_rate, float) and output_rate > 0.0
    assert sweep.spec_decode is not None
    assert 0.0 < sweep.spec_decode["acceptance_rate"] <= 1.0
