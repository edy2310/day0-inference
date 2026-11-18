"""End-to-end run flow (§5): reference phase, one phase per candidate, then the report.

Error isolation (§5.2): a failing candidate is recorded as ``ERROR`` and the run continues. A
failing reference aborts the run, because every metric is relative to it.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from day0 import __version__
from day0.artifacts import (
    LoadedReference,
    atomic_write_bytes,
    encode_candidate,
    encode_reference,
    load_candidate,
    load_reference,
    read_json,
    read_jsonl_gz,
    write_json,
    write_jsonl_gz,
)
from day0.backends import Backend, get_backend
from day0.backends.base import backend_context_from
from day0.client import InferenceClient
from day0.config import (
    Day0Config,
    ServerConfig,
    config_hash,
    reference_fingerprint,
    resolve_path,
)
from day0.env import capture_env, git_sha
from day0.errors import ConfigError, Day0Error, ReferenceMismatchError, TokenizerMismatchError
from day0.gate import STATUS_PASS, evaluate_candidate
from day0.perf.runner import perf_document, run_sweep
from day0.perf.workload import build_workload, requests_from_rows, workload_rows
from day0.quality.determinism import run_consistency_check
from day0.quality.runner import (
    GenRequest,
    PromptItem,
    build_prompt_items,
    max_tokens_by_suite,
    run_candidate,
    run_reference,
    tokenize_prompts,
)
from day0.report import render_run
from day0.suites import TOKENIZER_PROBES, SuiteItem, load_suites, suites_hash, tokenizer_fingerprint

logger = logging.getLogger(__name__)
console = Console(stderr=True)

RUN_JSON = "run.json"
ENV_JSON = "env.json"
RESOLVED_CONFIG = "config.resolved.yaml"
TOKENIZER_FILE = "tokenizer_fingerprint.txt"
REFERENCE_DIR = "reference"
CANDIDATES_DIR = "candidates"


class RunAborted(Day0Error):
    """The reference failed, so the run cannot produce a report."""


@dataclass(frozen=True)
class RunOptions:
    """Options of one ``day0 run``.

    Attributes:
        out_dir: Parent directory for the run directory. Defaults to ``run.output_dir``.
        only: Candidate names to run. Empty means all enabled candidates.
        reuse_reference: Stored reference directory to reuse instead of running the reference.
        skip_perf: Skip the performance phases.
        skip_quality: Skip the quality phases.
        capture_fixtures: Save the first raw response of each request kind.
        command: Command line, recorded in ``run.json`` for reproduction.
    """

    out_dir: Path | None = None
    only: tuple[str, ...] = ()
    reuse_reference: Path | None = None
    skip_perf: bool = False
    skip_quality: bool = False
    capture_fixtures: bool = False
    command: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunResult:
    """Outcome of a run.

    Attributes:
        run_dir: The run directory.
        recommendation: Recommended configuration, or ``None``.
        exit_code: 0 when a configuration is recommended, otherwise 1.
    """

    run_dir: Path
    recommendation: str | None
    exit_code: int


def utc_now() -> str:
    """Current UTC time in ISO 8601 with a ``Z`` suffix.

    Returns:
        The timestamp.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _timestamp_slug() -> str:
    """Current UTC time as ``YYYYmmddTHHMMSSZ``.

    Returns:
        The slug.
    """
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _make_run_dir(cfg: Day0Config, out_dir: Path | None) -> Path:
    """Create a new, unused run directory.

    Args:
        cfg: Validated configuration.
        out_dir: Parent directory, or ``None`` for ``run.output_dir``.

    Returns:
        The created directory.
    """
    parent = out_dir.resolve() if out_dir is not None else resolve_path(cfg, cfg.run.output_dir)
    base = f"{cfg.run.name}-{_timestamp_slug()}"
    candidate = parent / base
    counter = 1
    while candidate.exists():
        counter += 1
        candidate = parent / f"{base}-{counter}"
    candidate.mkdir(parents=True)
    return candidate


def _write_resolved_config(cfg: Day0Config, run_dir: Path) -> None:
    """Write the configuration with every path made absolute, so the run can be read later.

    Args:
        cfg: Validated configuration.
        run_dir: Run directory.
    """
    payload = cfg.model_dump(mode="json")
    payload["run"]["output_dir"] = str(resolve_path(cfg, cfg.run.output_dir))
    for suite, original in zip(payload["quality"]["suites"], cfg.quality.suites, strict=True):
        suite["path"] = str(resolve_path(cfg, original.path))
    text = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
    atomic_write_bytes(run_dir / RESOLVED_CONFIG, text.encode("utf-8"))


def _client_connections(cfg: Day0Config) -> int:
    """Connection pool size large enough for the highest concurrency used.

    Args:
        cfg: Validated configuration.

    Returns:
        The pool size.
    """
    levels = cfg.perf.concurrency_levels
    values = [
        cfg.quality.concurrency,
        cfg.quality.prefill_concurrency,
        cfg.quality.noise_floor.concurrency_b,
        cfg.quality.determinism.concurrency_high,
        max(levels),
    ]
    return max(values) + 4


@contextmanager
def _progress(description: str, total: int) -> Iterator[Callable[[], None]]:
    """Show a progress bar for a phase and yield the callback that advances it.

    Args:
        description: Label shown next to the bar.
        total: Number of steps.

    Yields:
        A function that advances the bar by one step.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(description, total=max(total, 1))

        def advance() -> None:
            progress.advance(task)

        yield advance


def _run_json_base(
    cfg: Day0Config,
    opts: RunOptions,
    run_dir: Path,
    digest: str,
    fingerprint: str,
    candidate_order: Sequence[str],
) -> dict[str, Any]:
    """Build the initial ``run.json`` document.

    Args:
        cfg: Validated configuration.
        opts: Run options.
        run_dir: Run directory.
        digest: Config hash.
        fingerprint: Reference fingerprint.
        candidate_order: Candidates that will run, in configuration order.

    Returns:
        The document.
    """
    return {
        "schema_version": 1,
        "day0_version": __version__,
        "git_sha": git_sha(),
        "started_utc": utc_now(),
        "ended_utc": None,
        "run_name": cfg.run.name,
        "run_dir": str(run_dir),
        "config_hash": digest,
        "reference_fingerprint": fingerprint,
        "command": list(opts.command),
        "backend_kind": cfg.backend.kind,
        "candidate_order": list(candidate_order),
        "reused_reference": opts.reuse_reference is not None,
        "statuses": {},
    }


async def _reference_phase(
    cfg: Day0Config,
    backend: Backend,
    run_dir: Path,
    items: Sequence[SuiteItem],
    opts: RunOptions,
    fingerprint: str,
    suite_digest: str,
) -> None:
    """Run the reference: quality, noise floor, perf sweep, and their artifacts (§5.1).

    Args:
        cfg: Validated configuration.
        backend: Backend for the reference server.
        run_dir: Run directory.
        items: Suite items, already loaded.
        opts: Run options.
        fingerprint: Reference fingerprint.
        suite_digest: Hash of the suite files.

    Raises:
        RunAborted: Wrapping any error, after its status has been recorded.
    """
    ref_dir = run_dir / REFERENCE_DIR
    ref_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_utc = utc_now()
    status: dict[str, Any] = {"status": "COMPLETED", "started_utc": started_utc}
    logger.info("starting the reference server")
    handle = await backend.start(cfg.reference, ref_dir / "server.log")
    capture = run_dir / "fixtures" / "reference" if opts.capture_fixtures else None
    client = InferenceClient(
        handle.base_url,
        handle.served_model,
        max_connections=_client_connections(cfg),
        capture_dir=capture,
    )
    k = cfg.quality.top_k_logprobs
    try:
        env = await capture_env(
            client, backend_kind=cfg.backend.kind, executable=cfg.backend.executable
        )
        write_json(run_dir / ENV_JSON, env)

        probe_ids = [await client.tokenize_chat(list(probe)) for probe in TOKENIZER_PROBES]
        probe_fingerprint = tokenizer_fingerprint(probe_ids)
        atomic_write_bytes(ref_dir / TOKENIZER_FILE, (probe_fingerprint + "\n").encode("utf-8"))

        if not opts.skip_quality:
            prompts = await tokenize_prompts(client, items, concurrency=cfg.quality.concurrency)
            prompt_items = build_prompt_items(cfg, items, prompts)
            with _progress("reference quality", len(prompt_items)) as tick:
                records = await run_reference(
                    client, prompt_items, cfg, logprobs_k=k, seed=cfg.run.seed, on_done=tick
                )
            write_jsonl_gz(
                ref_dir / "quality.jsonl.gz", (encode_reference(record) for record in records)
            )
            if cfg.quality.noise_floor.enabled:
                await _noise_floor(cfg, client, prompt_items, ref_dir, k)

        if not opts.skip_perf:
            logger.info("building the perf workload")
            workload = await build_workload(client, cfg, seed=cfg.run.seed)
            write_jsonl_gz(ref_dir / "perf_workload.jsonl.gz", workload_rows(workload))
            with _progress("reference perf", len(cfg.perf.concurrency_levels)) as tick:
                sweep = await _sweep_with_progress(client, workload, cfg, tick)
            write_json(ref_dir / "perf.json", perf_document(sweep, cfg))
            write_json(ref_dir / "metrics_snapshots.json", sweep.snapshots)

        write_json(
            ref_dir / "fingerprint.json",
            {
                "reference_fingerprint": fingerprint,
                "suites_hash": suite_digest,
                "served_model": handle.served_model,
                "config_hash": config_hash(cfg),
                "day0_version": __version__,
            },
        )
    except Exception as exc:
        status.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        raise RunAborted(f"reference failed: {exc}") from exc
    finally:
        status["duration_s"] = time.monotonic() - started
        status["ended_utc"] = utc_now()
        status["served_model"] = handle.served_model
        write_json(ref_dir / "status.json", status)
        await client.aclose()
        await backend.stop(handle)


async def _sweep_with_progress(
    client: InferenceClient, workload: Sequence[Any], cfg: Day0Config, tick: Callable[[], None]
) -> Any:
    """Run the perf sweep, advancing the progress bar once per level.

    Args:
        client: Server client.
        workload: Perf requests.
        cfg: Validated configuration.
        tick: Progress callback.

    Returns:
        The sweep result.
    """
    sweep = await run_sweep(client, workload, cfg, seed=cfg.run.seed)
    for _ in cfg.perf.concurrency_levels:
        tick()
    return sweep


async def _noise_floor(
    cfg: Day0Config,
    client: InferenceClient,
    prompt_items: Sequence[PromptItem],
    ref_dir: Path,
    k: int,
) -> None:
    """Measure the reference against itself at two batch concurrencies (§12.5).

    Args:
        cfg: Validated configuration.
        client: Reference client.
        prompt_items: Tokenized suite items.
        ref_dir: Reference directory.
        k: Top-k logprobs.
    """
    nf = cfg.quality.noise_floor
    subset = sorted(prompt_items, key=lambda p: p.item.id)[: nf.num_prompts]
    requests = [GenRequest(p.item.id, p.prompt_ids, p.max_tokens) for p in subset]
    golds = [(p.item.gold, p.item.answer_type) for p in subset]
    report = await run_consistency_check(
        client,
        requests,
        golds,
        concurrency_a=nf.concurrency_a,
        concurrency_b=nf.concurrency_b,
        logprobs_k=k,
        seed=cfg.run.seed,
        confidence=cfg.budget.confidence,
    )
    write_json(ref_dir / "noise_floor.json", {"enabled": True, **report.__dict__})


def _reuse_reference(reuse: Path, run_dir: Path, fingerprint: str) -> None:
    """Copy a stored reference into the run after checking that it matches the configuration.

    Args:
        cfg: Validated configuration.
        reuse: Stored reference directory.
        run_dir: Run directory.
        fingerprint: Fingerprint of the current configuration.

    Raises:
        ReferenceMismatchError: If the stored fingerprint differs, or is missing.
    """
    stored_path = reuse / "fingerprint.json"
    if not stored_path.is_file():
        raise ReferenceMismatchError(f"{reuse} has no fingerprint.json; it is not a reference run")
    stored = read_json(stored_path)
    if stored.get("reference_fingerprint") != fingerprint:
        raise ReferenceMismatchError(
            f"stored reference in {reuse} was produced by a different configuration "
            f"(model, reference args, suites, quality or perf parameters, or seed differ)"
        )
    shutil.copytree(reuse, run_dir / REFERENCE_DIR)
    parent_env = reuse.parent / ENV_JSON
    if parent_env.is_file() and not (run_dir / ENV_JSON).exists():
        shutil.copy(parent_env, run_dir / ENV_JSON)


def _write_run_json(run_dir: Path, document: dict[str, Any]) -> None:
    """Write ``run.json``.

    Args:
        run_dir: Run directory.
        document: Document to write.
    """
    write_json(run_dir / RUN_JSON, document)


async def _candidate_phase(
    cfg: Day0Config,
    server: ServerConfig,
    backend: Backend,
    run_dir: Path,
    reference_dir: Path,
    reference: LoadedReference,
    probe_fingerprint: str,
    opts: RunOptions,
) -> dict[str, Any]:
    """Run one candidate: quality, determinism, perf. Errors are recorded, not raised (§5.2).

    Args:
        cfg: Validated configuration.
        server: The candidate configuration.
        backend: Backend for the candidate server.
        run_dir: Run directory.
        reference: Loaded reference artifacts.
        probe_fingerprint: Tokenizer fingerprint of the reference.
        opts: Run options.

    Returns:
        The candidate's status document.
    """
    cand_dir = run_dir / CANDIDATES_DIR / server.name
    cand_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status: dict[str, Any] = {"status": "COMPLETED", "started_utc": utc_now()}
    handle = None
    client: InferenceClient | None = None
    k = cfg.quality.top_k_logprobs
    try:
        logger.info("candidate %s: starting server", server.name)
        handle = await backend.start(server, cand_dir / "server.log")
        capture = run_dir / "fixtures" / server.name if opts.capture_fixtures else None
        client = InferenceClient(
            handle.base_url,
            handle.served_model,
            max_connections=_client_connections(cfg),
            capture_dir=capture,
        )
        probes = [await client.tokenize_chat(list(probe)) for probe in TOKENIZER_PROBES]
        if tokenizer_fingerprint(probes) != probe_fingerprint:
            raise TokenizerMismatchError(
                f"candidate {server.name} tokenizes the probe set differently from the reference"
            )

        if not opts.skip_quality and reference.records:
            with _progress(f"{server.name} quality", len(reference.records)) as tick:
                quality = await run_candidate(
                    client, reference.records, cfg, logprobs_k=k, seed=cfg.run.seed, on_done=tick
                )
            write_jsonl_gz(
                cand_dir / "quality.jsonl.gz",
                (encode_candidate(record) for record in quality.records),
            )
            status["generation_logprobs_available"] = quality.generation_logprobs_available
            if cfg.quality.determinism.enabled:
                await _determinism(cfg, client, reference, cand_dir, k)

        if not opts.skip_perf:
            workload = requests_from_rows(read_jsonl_gz(reference_dir / "perf_workload.jsonl.gz"))
            with _progress(f"{server.name} perf", len(cfg.perf.concurrency_levels)) as tick:
                sweep = await _sweep_with_progress(client, workload, cfg, tick)
            write_json(cand_dir / "perf.json", perf_document(sweep, cfg))
            write_json(cand_dir / "metrics_snapshots.json", sweep.snapshots)
    except Exception as exc:
        logger.error("candidate %s failed: %s", server.name, exc)
        status.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
    finally:
        if client is not None:
            await client.aclose()
        if handle is not None:
            await backend.stop(handle)
        status["duration_s"] = time.monotonic() - started
        status["ended_utc"] = utc_now()
        if handle is not None:
            status["served_model"] = handle.served_model
        write_json(cand_dir / "status.json", status)
    return status


async def _determinism(
    cfg: Day0Config,
    client: InferenceClient,
    reference: LoadedReference,
    cand_dir: Path,
    k: int,
) -> None:
    """Measure a candidate against itself at two batch concurrencies (§12.5).

    Args:
        cfg: Validated configuration.
        client: Candidate client.
        reference: Loaded reference artifacts, which provide prompts and labels.
        cand_dir: Candidate directory.
        k: Top-k logprobs.
    """
    det = cfg.quality.determinism
    budgets = max_tokens_by_suite(cfg)
    subset = sorted(reference.records, key=lambda r: r.item_id)[: det.num_prompts]
    requests = [GenRequest(r.item_id, r.prompt_ids, budgets[r.suite]) for r in subset]
    golds = [(r.gold, r.answer_type) for r in subset]
    report = await run_consistency_check(
        client,
        requests,
        golds,
        concurrency_a=det.concurrency_low,
        concurrency_b=det.concurrency_high,
        logprobs_k=k,
        seed=cfg.run.seed,
        confidence=cfg.budget.confidence,
    )
    write_json(cand_dir / "determinism.json", {"enabled": True, **report.__dict__})


def _selected_candidates(cfg: Day0Config, only: Sequence[str]) -> list[ServerConfig]:
    """Enabled candidates, filtered by ``--only``, in configuration order.

    Args:
        cfg: Validated configuration.
        only: Names requested with ``--only``.

    Returns:
        The candidates to run.

    Raises:
        ConfigError: If a name in ``only`` is not a candidate of the configuration.
    """
    known = {server.name for server in cfg.candidates}
    unknown = [name for name in only if name not in known]
    if unknown:
        raise ConfigError(f"--only names unknown candidates: {unknown}")
    return [
        server for server in cfg.candidates if server.enabled and (not only or server.name in only)
    ]


async def run_experiment(cfg: Day0Config, opts: RunOptions) -> RunResult:
    """Run the reference and every selected candidate, then write the report.

    Args:
        cfg: Validated configuration.
        opts: Run options.

    Returns:
        The run directory, the recommendation, and the exit code.

    Raises:
        RunAborted: If the reference fails.
        ReferenceMismatchError: If the reused reference does not match the configuration.
    """
    selected = _selected_candidates(cfg, opts.only)
    suite_paths = [resolve_path(cfg, suite.path) for suite in cfg.quality.suites]
    items: list[SuiteItem] = load_suites(suite_paths) if not opts.skip_quality else []
    digest = suites_hash(suite_paths)
    fingerprint = reference_fingerprint(cfg, digest)
    run_dir = _make_run_dir(cfg, opts.out_dir)
    _write_resolved_config(cfg, run_dir)
    document = _run_json_base(
        cfg, opts, run_dir, config_hash(cfg), fingerprint, [s.name for s in selected]
    )
    _write_run_json(run_dir, document)
    logger.info("run directory: %s", run_dir)

    backend = get_backend(cfg.backend.kind, cfg.backend, backend_context_from(cfg))
    if opts.reuse_reference is not None:
        _reuse_reference(opts.reuse_reference, run_dir, fingerprint)
    else:
        await _reference_phase(cfg, backend, run_dir, items, opts, fingerprint, digest)

    reference = load_reference(run_dir / REFERENCE_DIR)
    probe_fingerprint = (
        (run_dir / REFERENCE_DIR / TOKENIZER_FILE).read_text(encoding="utf-8").strip()
    )
    statuses: dict[str, Any] = {"reference": reference.status.get("status", "COMPLETED")}

    candidate_statuses: dict[str, str] = {}
    for server in selected:
        status = await _candidate_phase(
            cfg,
            server,
            backend,
            run_dir,
            run_dir / REFERENCE_DIR,
            reference,
            probe_fingerprint,
            opts,
        )
        candidate_statuses[server.name] = str(status.get("status"))
    statuses["candidates"] = candidate_statuses

    document["ended_utc"] = utc_now()
    document["statuses"] = statuses
    _write_run_json(run_dir, document)
    rendered = render_run(run_dir)
    recommendation = rendered["recommendation"]["name"]
    exit_code = 0 if recommendation is not None else 1
    return RunResult(run_dir=run_dir, recommendation=recommendation, exit_code=exit_code)


async def run_check(
    cfg: Day0Config,
    *,
    candidate_name: str,
    reference_dir: Path,
    out_dir: Path | None,
    command: Sequence[str],
) -> tuple[Path, str, int]:
    """Run one candidate against stored reference artifacts and gate it (§15.1).

    Args:
        cfg: Validated configuration.
        candidate_name: Name of the candidate to run.
        reference_dir: Stored reference directory.
        out_dir: Parent directory for the check's run directory.
        command: Command line, recorded for reproduction.

    Returns:
        The check's run directory, the gate status, and the exit code (0 PASS, 1 otherwise, 2 for
        errors).

    Raises:
        ConfigError: If the candidate is not part of the configuration.
        ReferenceMismatchError: If the stored reference does not match the configuration.
    """
    matches = [s for s in cfg.candidates if s.name == candidate_name]
    if not matches:
        raise ConfigError(f"no candidate named {candidate_name!r} in the configuration")
    server = matches[0]
    suite_paths = [resolve_path(cfg, suite.path) for suite in cfg.quality.suites]
    digest = suites_hash(suite_paths)
    fingerprint = reference_fingerprint(cfg, digest)
    stored = (
        read_json(reference_dir / "fingerprint.json")
        if (reference_dir / "fingerprint.json").is_file()
        else {}
    )
    if stored.get("reference_fingerprint") != fingerprint:
        raise ReferenceMismatchError(
            f"stored reference in {reference_dir} does not match this configuration"
        )

    run_dir = _make_run_dir(cfg, out_dir)
    _write_resolved_config(cfg, run_dir)
    opts = RunOptions(command=tuple(command))
    document = _run_json_base(cfg, opts, run_dir, config_hash(cfg), fingerprint, [candidate_name])
    _write_run_json(run_dir, document)

    reference = load_reference(reference_dir)
    probe_fingerprint = (reference_dir / TOKENIZER_FILE).read_text(encoding="utf-8").strip()
    backend = get_backend(cfg.backend.kind, cfg.backend, backend_context_from(cfg))
    status = await _candidate_phase(
        cfg,
        server,
        backend,
        run_dir,
        reference_dir,
        reference,
        probe_fingerprint,
        opts,
    )

    loaded = load_candidate(candidate_name, run_dir / CANDIDATES_DIR / candidate_name)
    ref_correct = {r.item_id: r.correct for r in reference.records}
    evaluation = evaluate_candidate(
        name=candidate_name,
        candidate=loaded,
        ref_correct=ref_correct,
        budget=cfg.budget,
        slices=cfg.quality.slices,
        run_seed=cfg.run.seed,
    )
    write_json(
        run_dir / CANDIDATES_DIR / candidate_name / "check.json",
        {
            "status": evaluation.status,
            "reasons": evaluation.reasons,
            "error": evaluation.error,
            "target": evaluation.target,
            "overall": evaluation.overall,
            "slices": evaluation.slices,
            "failures": evaluation.failures,
            "not_covered": evaluation.not_covered,
            "insufficient": evaluation.insufficient,
        },
    )
    document["ended_utc"] = utc_now()
    document["statuses"] = {"candidates": {candidate_name: str(status.get("status"))}}
    _write_run_json(run_dir, document)
    if evaluation.status == STATUS_PASS:
        exit_code = 0
    elif evaluation.status == "ERROR":
        exit_code = 2
    else:
        exit_code = 1
    return run_dir, evaluation.status, exit_code


def run_report(run_dir: Path) -> dict[str, Any]:
    """Rebuild ``card.md``, ``card.json``, and ``pareto.png`` from a run directory.

    Args:
        run_dir: Run directory.

    Returns:
        The report data.
    """
    return render_run(run_dir)


__all__ = [
    "RunAborted",
    "RunOptions",
    "RunResult",
    "run_check",
    "run_experiment",
    "run_report",
    "PromptItem",
]
