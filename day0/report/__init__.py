"""Report generation: build the card data from run artifacts, then write card.md, card.json,
pareto.png, and one summary per candidate.

The report is a pure function of the run directory. ``day0 run`` and ``day0 report`` call the same
code, so the card does not depend on how it was produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from day0.artifacts import (
    LoadedCandidate,
    LoadedReference,
    atomic_write_bytes,
    load_candidate,
    load_reference,
    read_json,
    write_json,
)
from day0.config import Day0Config, load_config
from day0.errors import ConfigError
from day0.gate import (
    STATUS_FAIL_SLO,
    STATUS_INSUFFICIENT,
    STATUS_PASS,
    CandidateEvaluation,
    ParetoPoint,
    evaluate_candidate,
    pareto_front,
    recommend,
)
from day0.perf.metrics import timing_clock
from day0.quality.metrics import ReferenceRecord, topk_kl
from day0.report.card import render_card
from day0.report.pareto import render_pareto

RUN_JSON = "run.json"
ENV_JSON = "env.json"
RESOLVED_CONFIG = "config.resolved.yaml"
CARD_MD = "card.md"
CARD_JSON = "card.json"
PARETO_PNG = "pareto.png"
CARD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunArtifacts:
    """Everything a report needs, read from one run directory.

    Attributes:
        run_dir: Run directory.
        run: Contents of ``run.json``.
        config: The resolved configuration.
        env: Contents of ``env.json``.
        reference: Reference artifacts.
        candidates: Candidate artifacts, in configuration order.
    """

    run_dir: Path
    run: dict[str, Any]
    config: Day0Config
    env: dict[str, Any]
    reference: LoadedReference
    candidates: list[LoadedCandidate]


def load_run(run_dir: Path) -> RunArtifacts:
    """Read a run directory.

    Args:
        run_dir: Run directory created by ``day0 run``.

    Returns:
        The loaded artifacts.

    Raises:
        ConfigError: If the directory is not a run directory.
    """
    if not (run_dir / RUN_JSON).is_file():
        raise ConfigError(f"{run_dir} is not a day0 run directory (run.json is missing)")
    run = read_json(run_dir / RUN_JSON)
    config = load_config(run_dir / RESOLVED_CONFIG, check_suites=False)
    env_path = run_dir / ENV_JSON
    env = read_json(env_path) if env_path.is_file() else {}
    reference = load_reference(run_dir / "reference")
    names = [str(name) for name in run.get("candidate_order", [])]
    candidates = [load_candidate(name, run_dir / "candidates" / name) for name in names]
    return RunArtifacts(
        run_dir=run_dir,
        run=run,
        config=config,
        env=env,
        reference=reference,
        candidates=candidates,
    )


def reference_self_consistency(records: list[ReferenceRecord]) -> float | None:
    """Mean over prompts of the KL between the reference's own decode and prefill distributions.

    This is the numerical gap inside the reference, and the card reports it as context.

    Args:
        records: Reference records.

    Returns:
        The mean, or ``None`` when no record has both paths.
    """
    per_prompt: list[float] = []
    for record in records:
        if record.gen_topk is None or record.prefill_topk is None:
            continue
        pairs = list(zip(record.gen_topk, record.prefill_topk, strict=False))
        if pairs:
            per_prompt.append(sum(topk_kl(g, p) for g, p in pairs) / len(pairs))
    return sum(per_prompt) / len(per_prompt) if per_prompt else None


def _finite_or_none(value: Any) -> Any:
    """Replace non-finite floats with ``None``.

    Args:
        value: Any value.

    Returns:
        The value, or ``None`` for NaN or infinity.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _ratio(target: dict[str, Any] | None, reference_tok: float | None) -> float | None:
    """Throughput of a target summary relative to the reference.

    Args:
        target: Target-concurrency summary.
        reference_tok: Reference output tokens per second.

    Returns:
        The ratio, or ``None`` when either side is missing or zero.
    """
    if target is None or reference_tok is None or reference_tok <= 0:
        return None
    tok = target.get("output_tok_per_s")
    if tok is None:
        return None
    return float(tok) / reference_tok


def _candidate_entry(
    evaluation: CandidateEvaluation,
    candidate: LoadedCandidate,
    description: str | None,
    args: list[str],
    reference_tok: float | None,
    num_gpus: int,
) -> dict[str, Any]:
    """Build the report entry of one candidate.

    Args:
        evaluation: Gate result.
        candidate: Loaded artifacts.
        description: Configuration description.
        args: Server arguments from the configuration.
        reference_tok: Reference output tokens per second.
        num_gpus: GPUs per server.

    Returns:
        The entry.
    """
    perf = candidate.perf or {}
    duration = candidate.status.get("duration_s")
    return {
        "name": candidate.name,
        "role": "candidate",
        "description": description,
        "args": args,
        "status": evaluation.status,
        "reasons": evaluation.reasons,
        "error": evaluation.error,
        "target": evaluation.target,
        "throughput_vs_reference": _finite_or_none(_ratio(evaluation.target, reference_tok)),
        "max_concurrency_meeting_slo": perf.get("max_concurrency_meeting_slo"),
        "levels": perf.get("levels", []),
        "spec_decode": perf.get("spec_decode"),
        "overall": evaluation.overall,
        "slices": evaluation.slices,
        "failures": evaluation.failures,
        "insufficient": evaluation.insufficient,
        "not_covered": evaluation.not_covered,
        "determinism": candidate.determinism,
        "generation_logprobs_available": candidate.status.get("generation_logprobs_available"),
        "duration_s": duration,
        "gpu_seconds": (float(duration) * num_gpus) if duration is not None else None,
    }


def build_report(art: RunArtifacts) -> dict[str, Any]:
    """Build the card data from loaded artifacts.

    Args:
        art: The loaded run.

    Returns:
        A JSON-compatible dictionary: the content of ``card.json``.
    """
    cfg = art.config
    ref_records = art.reference.records
    ref_correct = {record.item_id: record.correct for record in ref_records}
    ref_perf = art.reference.perf or {}
    ref_target = ref_perf.get("target") if isinstance(ref_perf.get("target"), dict) else None
    ref_tok = float(ref_target["output_tok_per_s"]) if ref_target else None
    ref_status = (
        STATUS_INSUFFICIENT
        if ref_target is None
        else (STATUS_PASS if ref_target.get("slo_met") is True else STATUS_FAIL_SLO)
    )
    descriptions = {s.name: (s.description, s.args) for s in cfg.candidates}

    evaluations = [
        evaluate_candidate(
            name=candidate.name,
            candidate=candidate,
            ref_correct=ref_correct,
            budget=cfg.budget,
            slices=cfg.quality.slices,
            run_seed=cfg.run.seed,
        )
        for candidate in art.candidates
    ]
    by_name = {e.name: e for e in evaluations}
    recommendation = recommend(
        reference_name=cfg.reference.name,
        reference_target=ref_target,
        evaluations=evaluations,
    )

    candidate_entries = [
        _candidate_entry(
            by_name[candidate.name],
            candidate,
            descriptions.get(candidate.name, (None, []))[0],
            list(descriptions.get(candidate.name, (None, []))[1]),
            ref_tok,
            cfg.hardware.num_gpus,
        )
        for candidate in art.candidates
    ]
    ref_duration = art.reference.status.get("duration_s")
    reference_entry: dict[str, Any] = {
        "name": cfg.reference.name,
        "role": "reference",
        "description": cfg.reference.description,
        "args": cfg.reference.args,
        "status": ref_status,
        "reasons": [] if ref_status == STATUS_PASS else ["no SLO-compliant target summary"],
        "error": None,
        "target": ref_target,
        "throughput_vs_reference": 1.0 if ref_tok else None,
        "max_concurrency_meeting_slo": ref_perf.get("max_concurrency_meeting_slo"),
        "levels": ref_perf.get("levels", []),
        "spec_decode": ref_perf.get("spec_decode"),
        "overall": None,
        "slices": [],
        "failures": [],
        "insufficient": [],
        "not_covered": [],
        "determinism": None,
        "generation_logprobs_available": True,
        "duration_s": ref_duration,
        "gpu_seconds": (float(ref_duration) * cfg.hardware.num_gpus)
        if ref_duration is not None
        else None,
    }
    configs = [reference_entry, *candidate_entries]

    points: list[ParetoPoint] = []
    point_rows: list[dict[str, Any]] = []
    if ref_target is not None and ref_target.get("invalid") is not True:
        points.append(ParetoPoint(cfg.reference.name, 1.0, 0.0, 0.0))
        point_rows.append(
            {
                "name": cfg.reference.name,
                "x": 1.0,
                "flip_rate": 0.0,
                "flip_lower": 0.0,
                "flip_upper": 0.0,
                "status": ref_status,
            }
        )
    for entry in candidate_entries:
        target = entry["target"]
        flip = entry["overall"].get("flip_rate") if entry["overall"] else None
        ratio = entry["throughput_vs_reference"]
        if (
            target is None
            or target.get("invalid") is True
            or flip is None
            or flip.get("value") is None
            or ratio is None
        ):
            continue
        points.append(
            ParetoPoint(entry["name"], float(ratio), float(flip["value"]), float(flip["upper"]))
        )
        point_rows.append(
            {
                "name": entry["name"],
                "x": float(ratio),
                "flip_rate": flip["value"],
                "flip_lower": flip["lower"],
                "flip_upper": flip["upper"],
                "status": entry["status"],
            }
        )
    front = pareto_front(points)

    recommended = None
    if recommendation.name is not None:
        recommended = next(c for c in configs if c["name"] == recommendation.name)
    rec_target = recommended["target"] if recommended else None
    rec_flip = (
        recommended["overall"].get("flip_rate") if recommended and recommended["overall"] else None
    )
    rec_kl = (
        recommended["overall"].get("prefill_kl_mean")
        if recommended and recommended["overall"]
        else None
    )
    recommendation_data: dict[str, Any] = {
        "name": recommendation.name,
        "reason": recommendation.reason,
        "throughput_vs_reference": recommended["throughput_vs_reference"] if recommended else None,
        "p99_ttft_ms": (rec_target or {}).get("ttft_ms", {}).get("p99") if rec_target else None,
        "p99_tpot_ms": (rec_target or {}).get("tpot_ms", {}).get("p99") if rec_target else None,
        "cost_per_1m_output_tokens_usd": (rec_target or {}).get("cost_per_1m_output_tokens_usd")
        if rec_target
        else None,
        "flip_rate": rec_flip,
        "prefill_kl_mean": rec_kl,
    }

    run = art.run
    hardware = cfg.hardware
    data: dict[str, Any] = {
        "schema_version": CARD_SCHEMA_VERSION,
        "run": {
            "run_name": cfg.run.name,
            "model_id": cfg.model.id,
            "day0_version": run.get("day0_version"),
            "git_sha": run.get("git_sha"),
            "started_utc": run.get("started_utc"),
            "ended_utc": run.get("ended_utc"),
            "command": run.get("command", []),
            "config_hash": run.get("config_hash"),
            "reference_fingerprint": run.get("reference_fingerprint"),
            "target_concurrency": cfg.perf.target_concurrency,
            "hardware_label": hardware.label,
            "num_gpus": hardware.num_gpus,
            "gpu_hour_usd": hardware.gpu_hour_usd,
            "backend_kind": cfg.backend.kind,
            "timing_clock": timing_clock(cfg.backend.kind),
            "seed": cfg.run.seed,
        },
        "budget": cfg.budget.model_dump(mode="json"),
        "slo": cfg.slo.model_dump(mode="json"),
        "slices": list(cfg.quality.slices),
        "recommendation": recommendation_data,
        "configs": configs,
        "noise_floor": art.reference.noise_floor,
        "reference_self_consistency_kl": reference_self_consistency(ref_records),
        "pareto": {"points": point_rows, "front": front},
        "environment": art.env,
    }
    return data


def render_run(run_dir: Path) -> dict[str, Any]:
    """Build the report of a run directory and write every report file.

    Writes ``card.md``, ``card.json``, ``pareto.png``, and ``candidates/<name>/summary.json``.

    Args:
        run_dir: Run directory.

    Returns:
        The report data.
    """
    art = load_run(run_dir)
    data = build_report(art)
    write_json(run_dir / CARD_JSON, data)
    atomic_write_bytes(run_dir / CARD_MD, render_card(data).encode("utf-8"))
    for config in data["configs"]:
        if config["role"] == "candidate":
            write_json(run_dir / "candidates" / config["name"] / "summary.json", config)
    pareto = data["pareto"]
    recommended_name = data["recommendation"]["name"]
    render_pareto(
        run_dir / PARETO_PNG,
        points=pareto["points"],
        front=pareto["front"],
        recommended=recommended_name,
        flip_rate_max=art.config.budget.flip_rate_max,
        title=(
            f"{art.config.model.id} on "
            f"{(art.env.get('gpus') or [{'name': art.config.hardware.label}])[0]['name']} · "
            f"target concurrency {art.config.perf.target_concurrency}"
        ),
    )
    return data
