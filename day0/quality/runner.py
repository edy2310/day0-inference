"""Quality runs: reference and candidate, on the generation and teacher-forced paths.

Reference (§12.1): greedy generation with top-k logprobs, then a teacher-forced pass over
``prompt + reference_output`` that yields the reference prefill distributions.

Candidate (§12.2): the same prompts (token ids from the reference), greedy generation, and a
teacher-forced pass over the reference output. Metrics are computed per item.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from day0.client import CompletionResult, InferenceClient
from day0.config import Day0Config, resolve_path
from day0.errors import ConfigError, ServerResponseError
from day0.parsing import TopK
from day0.quality.metrics import CandidateRecord, ReferenceRecord, compute_item_metrics
from day0.seeds import derive_seed
from day0.suites import SuiteItem, assign_slice, check_answer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptItem:
    """A suite item with its tokenized prompt and slice.

    Attributes:
        item: The suite item.
        prompt_ids: Prompt token ids produced by the reference tokenizer.
        slice_index: Context-length slice of the prompt.
        max_tokens: Generation budget from the suite configuration.
    """

    item: SuiteItem
    prompt_ids: list[int]
    slice_index: int
    max_tokens: int


@dataclass(frozen=True)
class GenRequest:
    """One generation request: item id, prompt ids, and budget.

    Attributes:
        item_id: Item id, used to derive the request seed.
        prompt_ids: Prompt token ids.
        max_tokens: Generation budget.
    """

    item_id: str
    prompt_ids: list[int]
    max_tokens: int


@dataclass
class GenerationMode:
    """Whether generation requests ask for logprobs, and whether that may fall back.

    Attributes:
        allow_fallback: Candidates may retry without logprobs when the server rejects them.
            The reference never falls back, because its top-k is the baseline for every metric.
        logprobs_enabled: Becomes ``False`` after the first rejection when fallback is allowed.
    """

    allow_fallback: bool = False
    logprobs_enabled: bool = True


@dataclass(frozen=True)
class CandidateQualityRun:
    """Result of a candidate quality run.

    Attributes:
        records: One record per reference item, in reference order.
        generation_logprobs_available: False if the server rejected logprobs for generation.
    """

    records: list[CandidateRecord]
    generation_logprobs_available: bool


def max_tokens_by_suite(cfg: Day0Config) -> dict[str, int]:
    """Map each suite name to its generation budget.

    Args:
        cfg: Validated configuration.

    Returns:
        Suite name (file stem) to ``max_tokens``.
    """
    return {resolve_path(cfg, suite.path).stem: suite.max_tokens for suite in cfg.quality.suites}


def build_prompt_items(
    cfg: Day0Config, items: Sequence[SuiteItem], prompts: Sequence[Sequence[int]]
) -> list[PromptItem]:
    """Attach tokenized prompts, slices, and budgets to suite items.

    Args:
        cfg: Validated configuration.
        items: Suite items.
        prompts: Prompt token ids, one list per item, in the same order.

    Returns:
        The prompt items.

    Raises:
        ConfigError: If a prompt is longer than the last slice boundary.
    """
    budgets = max_tokens_by_suite(cfg)
    result: list[PromptItem] = []
    for item, ids in zip(items, prompts, strict=True):
        try:
            slice_index = assign_slice(len(ids), cfg.quality.slices)
        except ValueError as exc:
            raise ConfigError(f"item {item.id!r}: {exc}") from exc
        result.append(
            PromptItem(
                item=item,
                prompt_ids=list(ids),
                slice_index=slice_index,
                max_tokens=budgets[item.suite],
            )
        )
    return result


async def tokenize_prompts(
    client: InferenceClient, items: Sequence[SuiteItem], *, concurrency: int
) -> list[list[int]]:
    """Tokenize every item's chat messages on the reference server.

    Args:
        client: Reference client.
        items: Suite items.
        concurrency: Concurrent tokenize requests.

    Returns:
        Token ids per item, in input order.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def one(item: SuiteItem) -> list[int]:
        async with semaphore:
            return await client.tokenize_chat(item.messages)

    return list(await asyncio.gather(*(one(item) for item in items)))


def _is_logprobs_rejection(exc: ServerResponseError) -> bool:
    """Whether a server error means "this server does not accept logprobs".

    Args:
        exc: The error raised by the client.

    Returns:
        True for HTTP 400 responses that mention logprobs.
    """
    return exc.status == 400 and "logprob" in str(exc).lower()


async def _generate_one(
    client: InferenceClient,
    request: GenRequest,
    *,
    logprobs_k: int,
    seed: int,
    mode: GenerationMode,
) -> CompletionResult:
    """Run one greedy generation, with the logprobs fallback when the mode allows it.

    Args:
        client: Server client.
        request: The generation request.
        logprobs_k: Top-k logprobs to request.
        seed: Run seed.
        mode: Shared generation mode.

    Returns:
        The completion.
    """
    item_seed = derive_seed(seed, "generate", request.item_id)
    if mode.logprobs_enabled:
        try:
            return await client.complete(
                request.prompt_ids,
                max_tokens=request.max_tokens,
                logprobs=logprobs_k,
                prompt_logprobs=None,
                seed=item_seed,
            )
        except ServerResponseError as exc:
            if not (mode.allow_fallback and _is_logprobs_rejection(exc)):
                raise
            mode.logprobs_enabled = False
            logger.warning(
                "server rejected logprobs for generation; continuing without them: %s", exc
            )
    return await client.complete(
        request.prompt_ids,
        max_tokens=request.max_tokens,
        logprobs=None,
        prompt_logprobs=None,
        seed=item_seed,
    )


async def generate_outputs(
    client: InferenceClient,
    requests: Sequence[GenRequest],
    *,
    concurrency: int,
    logprobs_k: int,
    seed: int,
    mode: GenerationMode,
    on_done: Callable[[], None] | None = None,
) -> list[CompletionResult]:
    """Run greedy generation for many requests with bounded concurrency.

    Args:
        client: Server client.
        requests: Generation requests.
        concurrency: Maximum requests in flight.
        logprobs_k: Top-k logprobs to request.
        seed: Run seed.
        mode: Shared generation mode.
        on_done: Called once per finished request, for progress display.

    Returns:
        Completions in request order.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def one(request: GenRequest) -> CompletionResult:
        async with semaphore:
            result = await _generate_one(
                client, request, logprobs_k=logprobs_k, seed=seed, mode=mode
            )
        if on_done is not None:
            on_done()
        return result

    return list(await asyncio.gather(*(one(request) for request in requests)))


async def _prefill_one(
    client: InferenceClient,
    prompt_ids: Sequence[int],
    output_ids: Sequence[int],
    *,
    logprobs_k: int,
    seed_label: str,
    seed: int,
) -> tuple[list[TopK], list[float]]:
    """Teacher-force ``prompt + output`` and read the distributions at the output positions.

    Position ``m + j`` (``m`` is the prompt length) holds the distribution that predicts
    ``output_ids[j]``. The actual token is always present in that position.

    Args:
        client: Server client.
        prompt_ids: Prompt token ids.
        output_ids: Output token ids to score.
        logprobs_k: Top-k logprobs to request.
        seed_label: Label used to derive the request seed.
        seed: Run seed.

    Returns:
        ``(topk, token_logprobs)``, one entry per output token.

    Raises:
        ServerResponseError: If the server omits the prompt logprobs or the actual token.
    """
    teacher = [*prompt_ids, *output_ids]
    result = await client.complete(
        teacher,
        max_tokens=1,
        logprobs=None,
        prompt_logprobs=logprobs_k,
        seed=derive_seed(seed, "prefill", seed_label),
    )
    rows = result.prompt_topk
    m = len(prompt_ids)
    if rows is None or len(rows) < m + len(output_ids):
        raise ServerResponseError(f"server returned no prompt logprobs for {seed_label}")
    topk: list[TopK] = []
    token_lp: list[float] = []
    for j, token in enumerate(output_ids):
        row = rows[m + j]
        if row is None or token not in row:
            raise ServerResponseError(
                f"prompt logprobs for {seed_label} do not include the actual token at output "
                f"position {j}"
            )
        topk.append(row)
        token_lp.append(row[token])
    return topk, token_lp


async def run_reference(
    client: InferenceClient,
    prompt_items: Sequence[PromptItem],
    cfg: Day0Config,
    *,
    logprobs_k: int,
    seed: int,
    on_done: Callable[[], None] | None = None,
) -> list[ReferenceRecord]:
    """Run the reference quality measurement (§12.1).

    Args:
        client: Reference client.
        prompt_items: Tokenized suite items.
        cfg: Validated configuration.
        logprobs_k: Top-k logprobs to request.
        seed: Run seed.
        on_done: Progress callback, called once per generation request.

    Returns:
        One record per item, in input order.

    Raises:
        ServerResponseError: If the reference server does not return ids or top-k logprobs.
    """
    requests = [GenRequest(p.item.id, p.prompt_ids, p.max_tokens) for p in prompt_items]
    completions = await generate_outputs(
        client,
        requests,
        concurrency=cfg.quality.concurrency,
        logprobs_k=logprobs_k,
        seed=seed,
        mode=GenerationMode(allow_fallback=False),
        on_done=on_done,
    )

    generated: list[tuple[list[int], list[TopK]]] = []
    for prompt, completion in zip(prompt_items, completions, strict=True):
        if completion.output_ids is None or completion.gen_topk is None:
            raise ServerResponseError(
                f"reference server returned no token ids or top-k logprobs for {prompt.item.id!r}; "
                f"check --logprobs-mode raw_logprobs and the tokens-as-ids format"
            )
        generated.append((completion.output_ids, completion.gen_topk))

    prefill_targets = [
        index
        for index, prompt in enumerate(prompt_items)
        if len(prompt.prompt_ids) <= cfg.quality.prefill_max_prompt_tokens and generated[index][0]
    ]
    prefill_semaphore = asyncio.Semaphore(cfg.quality.prefill_concurrency)

    async def run_prefill(index: int) -> tuple[list[TopK], list[float]]:
        async with prefill_semaphore:
            prompt = prompt_items[index]
            return await _prefill_one(
                client,
                prompt.prompt_ids,
                generated[index][0],
                logprobs_k=logprobs_k,
                seed_label=prompt.item.id,
                seed=seed,
            )

    prefill_results = await asyncio.gather(*(run_prefill(index) for index in prefill_targets))
    prefill_by_index = dict(zip(prefill_targets, prefill_results, strict=True))

    records: list[ReferenceRecord] = []
    for index, prompt in enumerate(prompt_items):
        output_ids, gen_topk = generated[index]
        output_text = completions[index].output_text
        prefill = prefill_by_index.get(index)
        records.append(
            ReferenceRecord(
                item_id=prompt.item.id,
                suite=prompt.item.suite,
                slice_index=prompt.slice_index,
                prompt_ids=prompt.prompt_ids,
                output_ids=output_ids,
                output_text=output_text,
                gold=prompt.item.gold,
                answer_type=prompt.item.answer_type,
                correct=check_answer(output_text, prompt.item.gold, prompt.item.answer_type),
                gen_topk=gen_topk,
                prefill_topk=prefill[0] if prefill else None,
                prefill_token_lp=prefill[1] if prefill else None,
            )
        )
    return records


async def run_candidate(
    client: InferenceClient,
    references: Sequence[ReferenceRecord],
    cfg: Day0Config,
    *,
    logprobs_k: int,
    seed: int,
    on_done: Callable[[], None] | None = None,
) -> CandidateQualityRun:
    """Run the candidate quality measurement and compute per-item metrics (§12.2).

    Args:
        client: Candidate client.
        references: Reference records, which supply prompts, gold labels, and reference outputs.
        cfg: Validated configuration.
        logprobs_k: Top-k logprobs to request.
        seed: Run seed.
        on_done: Progress callback, called once per generation request.

    Returns:
        The candidate records and whether generation logprobs were available.
    """
    mode = GenerationMode(allow_fallback=True)
    budgets = max_tokens_by_suite(cfg)
    requests = [GenRequest(ref.item_id, ref.prompt_ids, budgets[ref.suite]) for ref in references]
    completions = await generate_outputs(
        client,
        requests,
        concurrency=cfg.quality.concurrency,
        logprobs_k=logprobs_k,
        seed=seed,
        mode=mode,
        on_done=on_done,
    )

    prefill_targets = [
        index
        for index, ref in enumerate(references)
        if ref.prefill_topk is not None and ref.output_ids
    ]
    prefill_semaphore = asyncio.Semaphore(cfg.quality.prefill_concurrency)

    async def run_prefill(index: int) -> tuple[list[TopK], list[float]]:
        async with prefill_semaphore:
            ref = references[index]
            return await _prefill_one(
                client,
                ref.prompt_ids,
                ref.output_ids,
                logprobs_k=logprobs_k,
                seed_label=ref.item_id,
                seed=seed,
            )

    prefill_results = await asyncio.gather(*(run_prefill(index) for index in prefill_targets))
    prefill_by_index = dict(zip(prefill_targets, prefill_results, strict=True))

    records: list[CandidateRecord] = []
    for index, ref in enumerate(references):
        completion = completions[index]
        cand_correct = check_answer(completion.output_text, ref.gold, ref.answer_type)
        prefill = prefill_by_index.get(index)
        metrics = compute_item_metrics(
            ref=ref,
            cand_output_ids=completion.output_ids,
            cand_output_text=completion.output_text,
            cand_gen_topk=completion.gen_topk,
            cand_prefill_topk=prefill[0] if prefill else None,
            cand_prefill_token_lp=prefill[1] if prefill else None,
            cand_correct=cand_correct,
        )
        records.append(
            CandidateRecord(
                item_id=ref.item_id,
                suite=ref.suite,
                slice_index=ref.slice_index,
                output_ids=completion.output_ids,
                output_text=completion.output_text,
                correct=cand_correct,
                gen_topk=completion.gen_topk,
                prefill_topk=prefill[0] if prefill else None,
                prefill_token_lp=prefill[1] if prefill else None,
                metrics=metrics,
            )
        )
    return CandidateQualityRun(records=records, generation_logprobs_available=mode.logprobs_enabled)
