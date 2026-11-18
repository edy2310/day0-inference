"""Read and write run artifacts: atomic JSON, deterministic gzip JSONL, and quality records.

Every write goes to ``<name>.tmp`` first and is then moved into place with ``os.replace``, so an
interrupted run never leaves a half-written artifact. Gzip output has ``mtime=0``, so the same
records always produce the same bytes.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from day0.parsing import TopK
from day0.quality.metrics import CandidateRecord, ItemMetrics, ReferenceRecord

LOGPROB_DECIMALS = 6
FORMAT_VERSION = 1


def _sanitize(value: Any) -> Any:
    """Replace NaN and infinities with ``None`` so that the output is strict JSON.

    Args:
        value: Any JSON-compatible value.

    Returns:
        The same structure with non-finite floats replaced by ``None``.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize(item) for item in value]
    return value


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to a file atomically.

    Args:
        path: Destination file. Parent directories are created.
        data: Content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON file with sorted keys, two-space indentation, and UTF-8 encoding.

    Args:
        path: Destination file.
        payload: JSON-compatible value. NaN and infinities become ``null``.
    """
    text = json.dumps(
        _sanitize(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    )
    atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def read_json(path: Path) -> Any:
    """Read a JSON file.

    Args:
        path: File to read.

    Returns:
        The decoded value.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write one JSON object per line, gzip-compressed, with a fixed gzip timestamp.

    Args:
        path: Destination file.
        rows: Objects to write, in order.
    """
    lines = [
        json.dumps(_sanitize(row), sort_keys=True, ensure_ascii=False, allow_nan=False)
        for row in rows
    ]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    compressed = gzip.compress(payload, mtime=0)
    atomic_write_bytes(path, compressed)


def read_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    """Read a gzip-compressed JSONL file.

    Args:
        path: File to read.

    Yields:
        One decoded object per line.
    """
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def encode_topk(distributions: Iterable[TopK] | None) -> list[list[list[float | int]]] | None:
    """Encode per-position top-k distributions as ``[token_id, logprob]`` pairs.

    JSON objects cannot have integer keys, so pairs are used. Logprobs are rounded.

    Args:
        distributions: One distribution per position, or ``None``.

    Returns:
        Nested lists, or ``None``.
    """
    if distributions is None:
        return None
    return [
        [[token, round(logprob, LOGPROB_DECIMALS)] for token, logprob in sorted(dist.items())]
        for dist in distributions
    ]


def decode_topk(encoded: list[list[list[float | int]]] | None) -> list[TopK] | None:
    """Decode the output of :func:`encode_topk`.

    Args:
        encoded: Nested lists, or ``None``.

    Returns:
        Dictionaries, or ``None``.
    """
    if encoded is None:
        return None
    return [{int(pair[0]): float(pair[1]) for pair in position} for position in encoded]


def encode_reference(record: ReferenceRecord) -> dict[str, Any]:
    """Encode a reference record as one JSON object.

    Args:
        record: The reference record.

    Returns:
        The JSON-compatible object.
    """
    return {
        "item_id": record.item_id,
        "suite": record.suite,
        "slice_index": record.slice_index,
        "prompt_ids": record.prompt_ids,
        "output_ids": record.output_ids,
        "output_text": record.output_text,
        "gold": record.gold,
        "answer_type": record.answer_type,
        "correct": record.correct,
        "gen_topk": encode_topk(record.gen_topk),
        "prefill_topk": encode_topk(record.prefill_topk),
        "prefill_token_lp": (
            [round(value, LOGPROB_DECIMALS) for value in record.prefill_token_lp]
            if record.prefill_token_lp is not None
            else None
        ),
    }


def decode_reference(row: Mapping[str, Any]) -> ReferenceRecord:
    """Decode a reference record.

    Args:
        row: The stored JSON object.

    Returns:
        The reference record.
    """
    return ReferenceRecord(
        item_id=str(row["item_id"]),
        suite=str(row["suite"]),
        slice_index=int(row["slice_index"]),
        prompt_ids=[int(t) for t in row["prompt_ids"]],
        output_ids=[int(t) for t in row["output_ids"]],
        output_text=str(row["output_text"]),
        gold=row["gold"],
        answer_type=row["answer_type"],
        correct=row["correct"],
        gen_topk=decode_topk(row["gen_topk"]),
        prefill_topk=decode_topk(row["prefill_topk"]),
        prefill_token_lp=(
            [float(v) for v in row["prefill_token_lp"]]
            if row["prefill_token_lp"] is not None
            else None
        ),
    )


def encode_metrics(metrics: ItemMetrics) -> dict[str, Any]:
    """Encode per-item metrics.

    Args:
        metrics: The metrics.

    Returns:
        The JSON-compatible object.
    """
    return {
        "prefill_kl_mean": metrics.prefill_kl_mean,
        "prefill_kl_p99_position": metrics.prefill_kl_p99_position,
        "prefill_top1_agreement": metrics.prefill_top1_agreement,
        "prefill_nll_delta_mean": metrics.prefill_nll_delta_mean,
        "exact_match": metrics.exact_match,
        "first_divergence": metrics.first_divergence,
        "first_divergence_frac": metrics.first_divergence_frac,
        "gen_kl_prefix_mean": metrics.gen_kl_prefix_mean,
        "flip": metrics.flip,
        "flip_direction": metrics.flip_direction,
    }


def decode_metrics(row: Mapping[str, Any]) -> ItemMetrics:
    """Decode per-item metrics.

    Args:
        row: The stored JSON object.

    Returns:
        The metrics.
    """
    return ItemMetrics(
        prefill_kl_mean=_opt_float(row["prefill_kl_mean"]),
        prefill_kl_p99_position=_opt_float(row["prefill_kl_p99_position"]),
        prefill_top1_agreement=_opt_float(row["prefill_top1_agreement"]),
        prefill_nll_delta_mean=_opt_float(row["prefill_nll_delta_mean"]),
        exact_match=bool(row["exact_match"]),
        first_divergence=row["first_divergence"],
        first_divergence_frac=_opt_float(row["first_divergence_frac"]),
        gen_kl_prefix_mean=_opt_float(row["gen_kl_prefix_mean"]),
        flip=row["flip"],
        flip_direction=row["flip_direction"],
    )


def _opt_float(value: Any) -> float | None:
    """Convert an optional JSON number to a float.

    Args:
        value: A number or ``None``.

    Returns:
        The float, or ``None``.
    """
    return float(value) if value is not None else None


def encode_candidate(record: CandidateRecord) -> dict[str, Any]:
    """Encode a candidate record as one JSON object.

    Args:
        record: The candidate record.

    Returns:
        The JSON-compatible object.
    """
    return {
        "item_id": record.item_id,
        "suite": record.suite,
        "slice_index": record.slice_index,
        "output_ids": record.output_ids,
        "output_text": record.output_text,
        "correct": record.correct,
        "gen_topk": encode_topk(record.gen_topk),
        "prefill_topk": encode_topk(record.prefill_topk),
        "prefill_token_lp": (
            [round(value, LOGPROB_DECIMALS) for value in record.prefill_token_lp]
            if record.prefill_token_lp is not None
            else None
        ),
        "metrics": encode_metrics(record.metrics),
    }


def decode_candidate(row: Mapping[str, Any]) -> CandidateRecord:
    """Decode a candidate record.

    Args:
        row: The stored JSON object.

    Returns:
        The candidate record.
    """
    return CandidateRecord(
        item_id=str(row["item_id"]),
        suite=str(row["suite"]),
        slice_index=int(row["slice_index"]),
        output_ids=[int(t) for t in row["output_ids"]] if row["output_ids"] is not None else None,
        output_text=str(row["output_text"]),
        correct=row["correct"],
        gen_topk=decode_topk(row["gen_topk"]),
        prefill_topk=decode_topk(row["prefill_topk"]),
        prefill_token_lp=(
            [float(v) for v in row["prefill_token_lp"]]
            if row["prefill_token_lp"] is not None
            else None
        ),
        metrics=decode_metrics(row["metrics"]),
    )


@dataclass(frozen=True)
class LoadedReference:
    """The reference artifacts of a run, as read from ``reference/``.

    Attributes:
        path: Directory the artifacts were read from.
        records: Reference quality records, in item order.
        perf: Contents of ``perf.json``, or ``None`` when the perf phase was skipped.
        noise_floor: Contents of ``noise_floor.json``, or ``None``.
        fingerprint: Contents of ``fingerprint.json``.
        status: Contents of ``status.json``.
    """

    path: Path
    records: list[ReferenceRecord]
    perf: dict[str, Any] | None
    noise_floor: dict[str, Any] | None
    fingerprint: dict[str, Any]
    status: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedCandidate:
    """The artifacts of one candidate, as read from ``candidates/<name>/``.

    Attributes:
        name: Candidate name.
        path: Directory the artifacts were read from.
        records: Candidate quality records, in item order.
        perf: Contents of ``perf.json``, or ``None``.
        determinism: Contents of ``determinism.json``, or ``None``.
        status: Contents of ``status.json`` (``status``, ``error``, timing, flags).
    """

    name: str
    path: Path
    records: list[CandidateRecord]
    perf: dict[str, Any] | None
    determinism: dict[str, Any] | None
    status: dict[str, Any] = field(default_factory=dict)

    @property
    def error(self) -> str | None:
        """The error message when the candidate failed, otherwise ``None``."""
        if self.status.get("status") == "ERROR":
            return str(self.status.get("error") or "unknown error")
        return None


def _optional_json(path: Path) -> Any:
    """Read a JSON file if it exists.

    Args:
        path: File to read.

    Returns:
        The decoded value, or ``None`` when the file is absent.
    """
    return read_json(path) if path.is_file() else None


def load_reference(path: Path) -> LoadedReference:
    """Load the reference artifacts from a reference directory.

    Args:
        path: The ``reference/`` directory.

    Returns:
        The loaded artifacts.
    """
    quality = path / "quality.jsonl.gz"
    records = [decode_reference(row) for row in read_jsonl_gz(quality)] if quality.is_file() else []
    return LoadedReference(
        path=path,
        records=records,
        perf=_optional_json(path / "perf.json"),
        noise_floor=_optional_json(path / "noise_floor.json"),
        fingerprint=dict(_optional_json(path / "fingerprint.json") or {}),
        status=dict(_optional_json(path / "status.json") or {}),
    )


def load_candidate(name: str, path: Path) -> LoadedCandidate:
    """Load the artifacts of one candidate.

    Args:
        name: Candidate name.
        path: The ``candidates/<name>/`` directory.

    Returns:
        The loaded artifacts. Quality records are empty when the file is absent.
    """
    quality = path / "quality.jsonl.gz"
    records = [decode_candidate(row) for row in read_jsonl_gz(quality)] if quality.is_file() else []
    return LoadedCandidate(
        name=name,
        path=path,
        records=records,
        perf=_optional_json(path / "perf.json"),
        determinism=_optional_json(path / "determinism.json"),
        status=dict(_optional_json(path / "status.json") or {}),
    )
