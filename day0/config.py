"""Pydantic configuration models for day0 runs.

Configurations are loaded from YAML, validated strictly (unknown keys are errors), and hashed
canonically so that every report can state which configuration produced it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from day0.errors import ConfigError

NAME_PATTERN = r"^[a-z0-9-]+$"


class _StrictModel(BaseModel):
    """Base class that rejects unknown keys, so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class RunConfig(_StrictModel):
    """Run identity and reproducibility settings.

    Attributes:
        name: Short slug used in the run directory name.
        seed: Base seed for every derived random stream.
        output_dir: Directory for run outputs. Relative paths resolve against the YAML file.
    """

    name: str = Field(pattern=NAME_PATTERN)
    seed: int = 1234
    output_dir: Path = Path("results")


class ModelConfig(_StrictModel):
    """The model under test.

    Attributes:
        id: Hugging Face id or local path passed to ``vllm serve``.
        max_model_len: Maximum context length in tokens.
    """

    id: str = Field(min_length=1)
    max_model_len: int = Field(gt=0)


class HardwareConfig(_StrictModel):
    """Hardware description used for cost metrics.

    Attributes:
        num_gpus: Number of GPUs used by each server.
        gpu_hour_usd: Price per GPU-hour in US dollars.
        label: Informational label; the real hardware is captured at runtime.
    """

    num_gpus: int = Field(1, ge=1)
    gpu_hour_usd: float = Field(ge=0)
    label: str | None = None


class BackendConfig(_StrictModel):
    """How servers are launched.

    Attributes:
        kind: ``vllm`` launches ``vllm serve``, ``fake`` launches the built-in fake server, and
            ``external`` talks to a server that is already running.
        executable: Path or name of the vLLM executable.
        host: Interface the launched server binds to.
        port: Fixed port used by every launched server.
        base_url: Base URL for the external backend.
        startup_timeout_s: Seconds to wait for ``/health``.
        cooldown_s: Seconds to wait after stopping a server, so GPU memory is released.
        common_args: Extra arguments passed to every launched server.
        env: Extra environment variables for every launched server.
    """

    kind: Literal["vllm", "fake", "external"]
    executable: str = "vllm"
    host: str = "127.0.0.1"
    port: int = Field(8100, ge=1, le=65535)
    base_url: str | None = None
    startup_timeout_s: int = Field(1200, gt=0)
    cooldown_s: int = Field(10, ge=0)
    common_args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class ServerConfig(_StrictModel):
    """One serving configuration: the reference or a candidate.

    Attributes:
        name: Unique slug for the configuration.
        args: Raw command-line arguments, passed through without interpretation.
        env: Extra environment variables for this server.
        base_url: Server URL for the external backend only.
        enabled: Candidates with ``False`` are skipped. The reference cannot be disabled.
        description: Free text shown in the card.
    """

    name: str = Field(pattern=NAME_PATTERN)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    base_url: str | None = None
    enabled: bool = True
    description: str | None = None


class SuiteConfig(_StrictModel):
    """One evaluation suite used by the quality measurement.

    Attributes:
        path: JSONL file. Relative paths resolve against the YAML file's directory.
        max_tokens: Generation budget for every item in the suite.
        weight_in_flips: Whether items with gold labels contribute to flip rates.
    """

    path: Path
    max_tokens: int = Field(gt=0)
    weight_in_flips: bool = True


class NoiseFloorConfig(_StrictModel):
    """Reference self-consistency check across two batch concurrencies.

    Attributes:
        enabled: Whether to run the check.
        num_prompts: Number of items to use (first items in ``item_id`` order).
        concurrency_a: First concurrency level.
        concurrency_b: Second concurrency level.
    """

    enabled: bool = True
    num_prompts: int = Field(50, ge=1)
    concurrency_a: int = Field(1, ge=1)
    concurrency_b: int = Field(32, ge=1)


class DeterminismConfig(_StrictModel):
    """Candidate self-consistency check across two batch concurrencies.

    Attributes:
        enabled: Whether to run the check.
        num_prompts: Number of items to use.
        concurrency_low: Low concurrency level.
        concurrency_high: High concurrency level.
    """

    enabled: bool = False
    num_prompts: int = Field(50, ge=1)
    concurrency_low: int = Field(1, ge=1)
    concurrency_high: int = Field(64, ge=1)


class QualityConfig(_StrictModel):
    """Quality measurement settings.

    Attributes:
        top_k_logprobs: Number of top-k logprobs requested per position (max 20).
        concurrency: Concurrent generation requests.
        prefill_concurrency: Concurrent teacher-forced requests.
        prefill_max_prompt_tokens: Items with longer prompts skip the prefill path.
        suites: Evaluation suites.
        slices: Ascending context-length boundaries, starting at 0 and ending at or above
            ``model.max_model_len``. Slice ``i`` is ``[slices[i], slices[i + 1])``.
        noise_floor: Reference self-consistency settings.
        determinism: Candidate self-consistency settings.
    """

    top_k_logprobs: int = Field(20, ge=1, le=20)
    concurrency: int = Field(16, ge=1)
    prefill_concurrency: int = Field(4, ge=1)
    prefill_max_prompt_tokens: int = Field(8192, gt=0)
    suites: list[SuiteConfig] = Field(min_length=1)
    slices: list[int] = Field(min_length=2)
    noise_floor: NoiseFloorConfig = Field(default_factory=lambda: NoiseFloorConfig())
    determinism: DeterminismConfig = Field(default_factory=lambda: DeterminismConfig())


class BudgetConfig(_StrictModel):
    """Pre-registered quality thresholds.

    Attributes:
        flip_rate_max: Maximum allowed upper bound of the flip rate.
        kl_mean_max: Maximum allowed upper bound of the mean prefill KL, in nats.
        top1_agreement_min: Minimum allowed lower bound of prefill top-1 agreement.
        min_prompts_per_slice: Minimum items a slice needs for a gated metric.
        confidence: Two-sided confidence level of the intervals.
        bootstrap_resamples: Number of bootstrap resamples.
    """

    flip_rate_max: float = Field(0.05, ge=0, le=1)
    kl_mean_max: float = Field(0.01, ge=0)
    top1_agreement_min: float = Field(0.95, ge=0, le=1)
    min_prompts_per_slice: int = Field(30, ge=1)
    confidence: float = Field(0.95, gt=0, lt=1)
    bootstrap_resamples: int = Field(2000, ge=10)


class LengthDist(_StrictModel):
    """A length distribution for perf prompts.

    Attributes:
        dist: ``fixed`` uses ``value`` tokens, ``uniform`` samples from ``[min, max]``.
        value: Length for ``fixed``.
        min: Lower bound for ``uniform``.
        max: Upper bound for ``uniform``.
    """

    dist: Literal["fixed", "uniform"]
    value: int | None = None
    min: int | None = None
    max: int | None = None

    @model_validator(mode="after")
    def _check_fields(self) -> LengthDist:
        """Check that the fields required by ``dist`` are present and consistent.

        Returns:
            The validated model.

        Raises:
            ValueError: If required fields are missing or inconsistent.
        """
        if self.dist == "fixed":
            if self.value is None or self.value <= 0:
                raise ValueError("fixed length requires a positive 'value'")
        elif self.min is None or self.max is None or not 0 < self.min <= self.max:
            raise ValueError("uniform length requires 0 < min <= max")
        return self

    def max_value(self) -> int:
        """Return the largest length this distribution can produce.

        Returns:
            ``value`` for ``fixed``, ``max`` for ``uniform``.
        """
        if self.dist == "fixed":
            return int(self.value or 0)
        return int(self.max or 0)


class PerfConfig(_StrictModel):
    """Performance measurement settings.

    Attributes:
        warmup_requests: Requests per level whose results are discarded.
        concurrency_levels: Closed-loop concurrency levels to sweep.
        target_concurrency: Level used for the headline numbers. Must be in the sweep.
        requests_per_level: Measured requests per level.
        input_len: Prompt-length distribution for the unique part of each prompt.
        output_len: Fixed number of output tokens per request.
        shared_prefix_len: Tokens shared by every prompt.
        max_error_rate: Levels whose error rate exceeds this are marked invalid.
    """

    warmup_requests: int = Field(20, ge=0)
    concurrency_levels: list[int] = Field(min_length=1)
    target_concurrency: int = Field(gt=0)
    requests_per_level: int = Field(256, ge=1)
    input_len: LengthDist
    output_len: int = Field(gt=0)
    shared_prefix_len: int = Field(0, ge=0)
    max_error_rate: float = Field(0.01, ge=0, le=1)


class SLOConfig(_StrictModel):
    """Service-level objective, applied at the target concurrency.

    Attributes:
        ttft_p99_ms: Maximum allowed p99 time to first token, in milliseconds.
        tpot_p99_ms: Maximum allowed p99 time per output token, in milliseconds.
    """

    ttft_p99_ms: float = Field(gt=0)
    tpot_p99_ms: float = Field(gt=0)


class Day0Config(_StrictModel):
    """Root configuration of a day0 run.

    Attributes:
        schema_version: Must be 1.
        run: Run identity and seed.
        model: Model under test.
        hardware: Hardware and price.
        backend: How servers are launched.
        reference: The reference configuration.
        candidates: Candidate configurations.
        quality: Quality measurement settings.
        budget: Pre-registered quality thresholds.
        perf: Performance measurement settings.
        slo: Service-level objective.
        base_dir: Directory of the YAML file. Set by :func:`load_config`, excluded from hashes.
    """

    schema_version: Literal[1]
    run: RunConfig
    model: ModelConfig
    hardware: HardwareConfig
    backend: BackendConfig
    reference: ServerConfig
    candidates: list[ServerConfig] = Field(default_factory=list)
    quality: QualityConfig
    budget: BudgetConfig = Field(default_factory=lambda: BudgetConfig())
    perf: PerfConfig
    slo: SLOConfig
    base_dir: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _check_cross_field_rules(self) -> Day0Config:
        """Check rules that involve more than one field.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If a cross-field rule is violated.
        """
        names = [self.reference.name, *(server.name for server in self.candidates)]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"server names must be unique across reference and candidates: {duplicates}"
            )
        if not self.reference.enabled:
            raise ValueError("the reference server cannot be disabled")

        if self.perf.target_concurrency not in self.perf.concurrency_levels:
            raise ValueError(
                f"perf.target_concurrency ({self.perf.target_concurrency}) must be one of "
                f"perf.concurrency_levels {self.perf.concurrency_levels}"
            )

        slices = self.quality.slices
        if slices[0] != 0:
            raise ValueError("quality.slices must start at 0")
        if any(later <= earlier for earlier, later in zip(slices, slices[1:], strict=False)):
            raise ValueError("quality.slices must be strictly ascending")
        if slices[-1] < self.model.max_model_len:
            raise ValueError(
                f"quality.slices must end at or above model.max_model_len "
                f"({self.model.max_model_len}); last boundary is {slices[-1]}"
            )

        if self.backend.kind == "external" and self.backend.base_url is None:
            for server in (self.reference, *self.candidates):
                if server.base_url is None:
                    raise ValueError(
                        f"backend.kind is 'external' but server '{server.name}' has no base_url"
                    )

        longest = (
            self.perf.input_len.max_value() + self.perf.shared_prefix_len + self.perf.output_len
        )
        if longest > self.model.max_model_len:
            raise ValueError(
                f"perf prompts can reach {longest} tokens (input max + shared prefix + output), "
                f"which exceeds model.max_model_len ({self.model.max_model_len})"
            )
        return self


def canonical_json(payload: Any) -> str:
    """Serialize a JSON-compatible object in a canonical form.

    Args:
        payload: Any JSON-compatible value.

    Returns:
        A string with sorted keys and no optional whitespace.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(cfg: Day0Config) -> str:
    """Return the SHA-256 of the canonical JSON form of a configuration.

    Args:
        cfg: Validated configuration.

    Returns:
        A 64-character lowercase hexadecimal digest.
    """
    payload = cfg.model_dump(mode="json")
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def reference_fingerprint(cfg: Day0Config, suites_hash: str) -> str:
    """Fingerprint everything that determines the reference artifacts.

    A stored reference can be reused only when this fingerprint matches (see §6.5 of the
    specification). Fields that only affect candidates or reporting are excluded.

    Args:
        cfg: Validated configuration.
        suites_hash: SHA-256 over the contents of every suite file.

    Returns:
        A 64-character lowercase hexadecimal digest.
    """
    payload: dict[str, Any] = {
        "model": {"id": cfg.model.id, "max_model_len": cfg.model.max_model_len},
        "seed": cfg.run.seed,
        "backend": {"kind": cfg.backend.kind, "common_args": cfg.backend.common_args},
        "reference_args": cfg.reference.args,
        "suites_hash": suites_hash,
        "quality": {
            "top_k_logprobs": cfg.quality.top_k_logprobs,
            "slices": cfg.quality.slices,
            "prefill_max_prompt_tokens": cfg.quality.prefill_max_prompt_tokens,
            "suites": [suite.model_dump(mode="json") for suite in cfg.quality.suites],
        },
        "perf": cfg.perf.model_dump(mode="json"),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def resolve_path(cfg: Day0Config, path: Path) -> Path:
    """Resolve a path from the configuration against the YAML file's directory.

    Args:
        cfg: Validated configuration whose ``base_dir`` is set by :func:`load_config`.
        path: Absolute or relative path taken from the configuration.

    Returns:
        An absolute path. Relative paths are resolved against ``base_dir``, or against the current
        working directory when the configuration was built in memory.
    """
    if path.is_absolute():
        return path
    base = cfg.base_dir if cfg.base_dir is not None else Path.cwd()
    return (base / path).resolve()


def _format_validation_error(exc: ValidationError) -> str:
    """Turn a pydantic error into one readable line per problem.

    Args:
        exc: The pydantic validation error.

    Returns:
        Newline-separated bullet points, each naming the offending field.
    """
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        message = str(error["msg"]).removeprefix("Value error, ")
        lines.append(f"  - {location}: {message}")
    return "\n".join(lines)


def load_config(path: Path, *, check_suites: bool = True) -> Day0Config:
    """Load, validate, and anchor a YAML configuration file.

    Args:
        path: Path to the YAML file.
        check_suites: If false, suite files are not checked for existence. Reports use this when
            they read a stored run whose suite files may have moved.

    Returns:
        The validated configuration, with ``base_dir`` set to the YAML file's directory.

    Raises:
        ConfigError: If the file cannot be read, is not valid YAML or configuration, or references
            a suite file that does not exist.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read configuration file {path}: {exc}") from exc
    try:
        raw: object = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: the top level of the file must be a mapping")

    try:
        cfg = Day0Config.model_validate(dict(raw))
    except ValidationError as exc:
        raise ConfigError(
            f"invalid configuration {path}:\n{_format_validation_error(exc)}"
        ) from exc

    base_dir = path.resolve().parent
    cfg.base_dir = base_dir
    if not check_suites:
        return cfg
    for suite in cfg.quality.suites:
        suite_file = resolve_path(cfg, suite.path)
        if not suite_file.is_file():
            raise ConfigError(
                f"suite file not found: {suite_file} (referenced by {path}). Suites under "
                f"suites/ are built by scripts/build_suites.py; see docs/runbook_gpu.md."
            )
    return cfg
