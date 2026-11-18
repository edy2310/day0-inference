"""Quality run against the fake server: reference and candidate records, and their artifacts."""

from __future__ import annotations

import math
import socket
from pathlib import Path

import pytest

from day0.artifacts import (
    decode_candidate,
    decode_reference,
    encode_candidate,
    encode_reference,
    read_jsonl_gz,
    write_jsonl_gz,
)
from day0.backends import FakeBackend
from day0.backends.base import BackendContext
from day0.client import InferenceClient
from day0.config import ServerConfig, load_config, resolve_path
from day0.quality.runner import (
    build_prompt_items,
    run_candidate,
    run_reference,
    tokenize_prompts,
)
from day0.suites import load_suites

ITEM_LIMIT = 16


def _free_port() -> int:
    """Return a TCP port that was free when queried.

    Returns:
        The port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.mark.parametrize("candidate_noise", ["0.0", "0.8"])
async def test_quality_run_with_fake_servers(
    examples_dir: Path, tmp_path: Path, candidate_noise: str
) -> None:
    """A quality run produces reference and candidate records that round-trip through artifacts.

    With zero candidate noise every output matches the reference. With large noise, the candidate
    diverges somewhere.

    Args:
        examples_dir: Directory with the example configurations.
        tmp_path: Pytest's per-test temporary directory.
        candidate_noise: Noise scale passed to the candidate fake server.
    """
    cfg = load_config(examples_dir / "fake.yaml")
    suite_paths = [resolve_path(cfg, suite.path) for suite in cfg.quality.suites]
    items = load_suites(suite_paths)[:ITEM_LIMIT]
    context = BackendContext(
        model_id=cfg.model.id,
        max_model_len=cfg.model.max_model_len,
        seed=cfg.run.seed,
        max_logprobs=cfg.quality.top_k_logprobs,
    )
    ref_port = _free_port()
    cand_port = _free_port()
    ref_backend = FakeBackend(cfg.backend.model_copy(update={"port": ref_port}), context)
    cand_backend = FakeBackend(cfg.backend.model_copy(update={"port": cand_port}), context)
    ref_handle = await ref_backend.start(cfg.reference, tmp_path / "reference" / "server.log")
    cand_handle = await cand_backend.start(
        ServerConfig(name="cand", args=["--noise", candidate_noise, "--speedup", "1.0"]),
        tmp_path / "candidates" / "cand" / "server.log",
    )
    ref_client = InferenceClient(ref_handle.base_url, ref_handle.served_model, max_connections=8)
    cand_client = InferenceClient(cand_handle.base_url, cand_handle.served_model, max_connections=8)
    try:
        prompts = await tokenize_prompts(ref_client, items, concurrency=8)
        prompt_items = build_prompt_items(cfg, items, prompts)
        references = await run_reference(
            ref_client, prompt_items, cfg, logprobs_k=cfg.quality.top_k_logprobs, seed=cfg.run.seed
        )
        candidate = await run_candidate(
            cand_client,
            references,
            cfg,
            logprobs_k=cfg.quality.top_k_logprobs,
            seed=cfg.run.seed,
        )
    finally:
        await ref_client.aclose()
        await cand_client.aclose()
        await cand_backend.stop(cand_handle)
        await ref_backend.stop(ref_handle)

    assert len(references) == len(items) == ITEM_LIMIT
    for record in references:
        assert record.gen_topk is not None
        assert record.prefill_topk is not None
        assert len(record.prefill_topk) == len(record.output_ids)
        assert record.prefill_token_lp is not None
        for position, token in enumerate(record.output_ids):
            assert token in record.prefill_topk[position]
    assert candidate.generation_logprobs_available is True

    ref_path = tmp_path / "reference" / "quality.jsonl.gz"
    cand_path = tmp_path / "candidates" / "cand" / "quality.jsonl.gz"
    write_jsonl_gz(ref_path, (encode_reference(r) for r in references))
    write_jsonl_gz(cand_path, (encode_candidate(r) for r in candidate.records))
    reloaded_refs = [decode_reference(row) for row in read_jsonl_gz(ref_path)]
    reloaded_cands = [decode_candidate(row) for row in read_jsonl_gz(cand_path)]
    assert [r.item_id for r in reloaded_refs] == [r.item_id for r in references]
    assert len(reloaded_cands) == len(candidate.records)

    if candidate_noise == "0.0":
        assert all(record.metrics.exact_match for record in candidate.records)
        assert all(
            record.metrics.prefill_kl_mean is not None and record.metrics.prefill_kl_mean < 1e-6
            for record in candidate.records
        )
    else:
        kls = [
            record.metrics.prefill_kl_mean
            for record in candidate.records
            if record.metrics.prefill_kl_mean is not None
        ]
        assert kls and max(kls) > 0.0
        assert all(not math.isnan(value) for value in kls)
