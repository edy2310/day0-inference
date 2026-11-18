"""Synthetic perf workload: deterministic prompts, tokenized once on the reference server.

The reference server tokenizes every prompt. The token ids are stored and reused for every
candidate, so all configurations do identical work (§11.1).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from day0.client import InferenceClient
from day0.config import Day0Config, LengthDist
from day0.corpus import filler_text
from day0.seeds import derive_seed


@dataclass(frozen=True)
class PerfRequest:
    """One prompt of the perf workload.

    Attributes:
        level: Concurrency level the prompt belongs to.
        index: Position within the level, counting warmup requests first.
        warmup: Whether the request is a warmup request, whose result is discarded.
        prompt_ids: Prompt token ids.
    """

    level: int
    index: int
    warmup: bool
    prompt_ids: list[int]


def sample_length(dist: LengthDist, rng: np.random.Generator) -> int:
    """Draw one target length in tokens.

    Args:
        dist: Length distribution.
        rng: Seeded generator.

    Returns:
        The sampled length.
    """
    if dist.dist == "fixed":
        return int(dist.value or 0)
    return int(rng.integers(int(dist.min or 1), int(dist.max or 1) + 1))


async def _tokens_of_length(
    client: InferenceClient,
    rng: np.random.Generator,
    target: int,
    *,
    add_special_tokens: bool,
) -> list[int]:
    """Build token ids of exactly ``target`` length from filler text.

    Text is extended and re-tokenized until it is long enough, then truncated. Truncation can split
    a word, which is acceptable because only token ids matter here.

    Args:
        client: Reference client.
        rng: Seeded generator.
        target: Required length in tokens.
        add_special_tokens: Whether the tokenizer adds special tokens.

    Returns:
        Exactly ``target`` token ids.
    """
    text = filler_text(rng, min_chars=max(target, 1))
    ids = await client.tokenize_text(text, add_special_tokens=add_special_tokens)
    while len(ids) < target:
        text += " " + filler_text(rng, min_chars=max(64, (target - len(ids)) * 4))
        ids = await client.tokenize_text(text, add_special_tokens=add_special_tokens)
    return ids[:target]


async def build_workload(
    client: InferenceClient, cfg: Day0Config, *, seed: int
) -> list[PerfRequest]:
    """Build every perf prompt for every concurrency level.

    Each level has its own seeded generator, so adding a level does not change the others.

    Args:
        client: Reference client.
        cfg: Validated configuration.
        seed: Run seed.

    Returns:
        Requests for every level, warmup requests first within each level.
    """
    perf = cfg.perf
    prefix: list[int] = []
    if perf.shared_prefix_len > 0:
        prefix_rng = np.random.default_rng(derive_seed(seed, "perf", "prefix"))
        prefix = await _tokens_of_length(
            client, prefix_rng, perf.shared_prefix_len, add_special_tokens=True
        )

    requests: list[PerfRequest] = []
    for level in perf.concurrency_levels:
        rng = np.random.default_rng(derive_seed(seed, "perf", level))
        total = perf.warmup_requests + perf.requests_per_level
        for index in range(total):
            body_len = sample_length(perf.input_len, rng)
            if prefix:
                body = await _tokens_of_length(client, rng, body_len, add_special_tokens=False)
                prompt = [*prefix, *body]
            else:
                prompt = await _tokens_of_length(client, rng, body_len, add_special_tokens=True)
            requests.append(
                PerfRequest(
                    level=level,
                    index=index,
                    warmup=index < perf.warmup_requests,
                    prompt_ids=prompt,
                )
            )
    return requests


def workload_rows(requests: Iterable[PerfRequest]) -> list[dict[str, Any]]:
    """Encode requests for ``perf_workload.jsonl.gz``.

    Args:
        requests: The requests.

    Returns:
        One JSON-compatible row per request.
    """
    return [
        {"level": r.level, "index": r.index, "warmup": r.warmup, "prompt_ids": r.prompt_ids}
        for r in requests
    ]


def requests_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[PerfRequest]:
    """Decode rows written by :func:`workload_rows`.

    Args:
        rows: Stored rows.

    Returns:
        The requests, in stored order.
    """
    return [
        PerfRequest(
            level=int(row["level"]),
            index=int(row["index"]),
            warmup=bool(row["warmup"]),
            prompt_ids=[int(t) for t in row["prompt_ids"]],
        )
        for row in rows
    ]
