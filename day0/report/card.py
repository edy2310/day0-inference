"""Render the serving profile card as Markdown (§17.1).

The renderer reads only the report data built from artifacts, so ``card.md`` is a pure function of
the run directory. Number formats follow §17.1: throughput with two decimals, latency in
milliseconds with one decimal, rates as percentages with two decimals, KL in scientific notation
with two significant digits, and intervals as ``[lo, hi]``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

LIMITATIONS_LINKS = (
    "Top-k KL is an approximation; see `docs/kl_validation.md`.",
    "The prefill path does not exercise speculative decoding, and vLLM ignores the prefix cache "
    "for prompt-logprob requests.",
    "Prompts above `prefill_max_prompt_tokens` are measured only on the generation path.",
    "Single GPU, single runtime, single model per run; results do not transfer across hardware.",
    "Many comparisons (configurations × slices × metrics) with no multiplicity correction.",
    "Client-side load generation can become CPU-bound at high concurrency.",
    "Dataset licences and the synthetic nature of the long-context suite; see "
    "`docs/limitations.md`.",
)


def _missing(value: object) -> bool:
    """Whether a value should be shown as ``n/a``.

    Args:
        value: Any value from the report data.

    Returns:
        True for ``None`` and non-finite floats.
    """
    return value is None or (isinstance(value, float) and not math.isfinite(value))


def fmt_ratio(value: Any) -> str:
    """Format a throughput ratio with two decimals.

    Args:
        value: Ratio, or ``None``.

    Returns:
        For example ``1.23×``, or ``n/a``.
    """
    return "n/a" if _missing(value) else f"{float(value):.2f}×"


def fmt_ms(value: Any) -> str:
    """Format milliseconds with one decimal.

    Args:
        value: Milliseconds, or ``None``.

    Returns:
        For example ``245.0``, or ``n/a``.
    """
    return "n/a" if _missing(value) else f"{float(value):.1f}"


def fmt_pct(value: Any) -> str:
    """Format a fraction as a percentage with two decimals.

    Args:
        value: Fraction, or ``None``.

    Returns:
        For example ``1.20%``, or ``n/a``.
    """
    return "n/a" if _missing(value) else f"{100 * float(value):.2f}%"


def fmt_kl(value: Any) -> str:
    """Format a KL value in scientific notation with two significant digits.

    Args:
        value: KL in nats, or ``None``.

    Returns:
        For example ``1.2e-03``, or ``n/a``.
    """
    return "n/a" if _missing(value) else f"{float(value):.1e}"


def fmt_money(value: Any) -> str:
    """Format a dollar amount.

    Args:
        value: Amount, or ``None``.

    Returns:
        For example ``$0.3333``, or ``n/a``.
    """
    return "n/a" if _missing(value) else f"${float(value):.4f}"


def _interval(entry: Mapping[str, Any] | None, formatter: Any) -> str:
    """Format a point estimate with its interval.

    Args:
        entry: Interval mapping with ``value``, ``lower``, and ``upper``.
        formatter: Value formatter.

    Returns:
        ``value [lo, hi]``, or ``n/a``.
    """
    if not entry or _missing(entry.get("value")):
        return "n/a"
    value = formatter(entry.get("value"))
    lower = formatter(entry.get("lower"))
    upper = formatter(entry.get("upper"))
    return f"{value} [{lower}, {upper}]"


def _cell(text: str) -> str:
    """Escape a cell for a Markdown table.

    Args:
        text: Cell text.

    Returns:
        The text with pipe characters escaped.
    """
    return text.replace("|", "\\|")


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render a Markdown table.

    Args:
        headers: Column headers.
        rows: Rows of cell text.

    Returns:
        Lines of the table.
    """
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(cell) for cell in row) + " |")
    return lines


def _target_cells(target: Mapping[str, Any] | None) -> tuple[str, str, str, str, str]:
    """Format the headline perf values of a target-concurrency summary.

    Args:
        target: Target-concurrency summary, or ``None``.

    Returns:
        Output tokens/s, p99 TTFT, p99 TPOT, max concurrency placeholder, and cost.
    """
    if not target:
        return ("n/a", "n/a", "n/a", "n/a", "n/a")
    ttft = target.get("ttft_ms") or {}
    tpot = target.get("tpot_ms") or {}
    tok = target.get("output_tok_per_s")
    return (
        "n/a" if tok is None or _missing(tok) else f"{float(tok):.2f}",
        fmt_ms(ttft.get("p99")),
        fmt_ms(tpot.get("p99")),
        "",
        fmt_money(target.get("cost_per_1m_output_tokens_usd")),
    )


def render_card(data: Mapping[str, Any]) -> str:
    """Render the whole card.

    Args:
        data: Report data, as built by ``day0.report.build_report``.

    Returns:
        The Markdown document.
    """
    run = data["run"]
    env = data.get("environment") or {}
    gpus = env.get("gpus") or []
    gpu_name = gpus[0]["name"] if gpus else (run.get("hardware_label") or "unknown GPU")
    title = f"Serving Profile Card — {run['model_id']} on {gpu_name} ({run['run_name']})"
    lines: list[str] = [f"# {title}", ""]

    lines += _recommendation_box(data)
    lines += ["", "## Main table", ""]
    lines += _main_table(data)
    lines += ["", "## Per-slice results", ""]
    lines += _slice_tables(data)
    lines += ["", "## Gate failures", ""]
    lines += _gate_failures(data)
    lines += ["", "## Speculative decoding", ""]
    lines += _spec_decode(data)
    lines += ["", "## Noise floor and determinism", ""]
    lines += _noise_section(data)
    lines += ["", "## Budget and SLO", ""]
    lines += _budget_section(data)
    lines += ["", "## Environment", ""]
    lines += _environment_section(data)
    lines += ["", "## Reproduce", "", "```bash", " ".join(run["command"]), "```", ""]
    lines += ["## Limitations", ""]
    lines += [f"- {item}" for item in LIMITATIONS_LINKS]
    lines += ["", "See `docs/limitations.md` for the full list.", ""]
    return "\n".join(lines)


def _recommendation_box(data: Mapping[str, Any]) -> list[str]:
    """Render the recommendation box at the top of the card.

    Args:
        data: Report data.

    Returns:
        Quoted Markdown lines.
    """
    rec = data["recommendation"]
    if rec.get("name") is None:
        return [f"> **No recommendation.** {rec.get('reason')}"]
    flip = rec.get("flip_rate") or {}
    kl = rec.get("prefill_kl_mean") or {}
    return [
        f"> **Recommendation:** `{rec['name']}`",
        f"> Throughput {fmt_ratio(rec.get('throughput_vs_reference'))} of reference · "
        f"p99 TTFT {fmt_ms(rec.get('p99_ttft_ms'))} ms · "
        f"p99 TPOT {fmt_ms(rec.get('p99_tpot_ms'))} ms · "
        f"{fmt_money(rec.get('cost_per_1m_output_tokens_usd'))} per 1M output tokens",
        f"> Flip rate {_interval(flip, fmt_pct)} · prefill KL {_interval(kl, fmt_kl)} nats",
    ]


def _main_table(data: Mapping[str, Any]) -> list[str]:
    """Render the one-row-per-configuration main table.

    Args:
        data: Report data.

    Returns:
        Table lines.
    """
    headers = [
        "Config",
        "Status",
        "×throughput",
        "Output tok/s",
        "p99 TTFT (ms)",
        "p99 TPOT (ms)",
        "Max conc. meeting SLO",
        "$/1M tokens",
        "Flip rate [CI]",
        "Exact match",
        "Prefill KL [CI]",
        "Top-1 [CI]",
        "Gen KL (prefix)",
    ]
    rows: list[list[str]] = []
    for config in data["configs"]:
        tok, ttft, tpot, _, cost = _target_cells(config.get("target"))
        overall = config.get("overall") or {}
        max_conc = config.get("max_concurrency_meeting_slo")
        rows.append(
            [
                f"`{config['name']}`" + (" (reference)" if config["role"] == "reference" else ""),
                config["status"],
                fmt_ratio(config.get("throughput_vs_reference")),
                tok,
                ttft,
                tpot,
                "n/a" if max_conc is None else str(max_conc),
                cost,
                _interval(overall.get("flip_rate"), fmt_pct) if overall else "baseline",
                fmt_pct((overall.get("exact_match_rate") or {}).get("value"))
                if overall
                else "100%",
                _interval(overall.get("prefill_kl_mean"), fmt_kl) if overall else "baseline",
                _interval(overall.get("prefill_top1_agreement"), fmt_pct)
                if overall
                else "baseline",
                fmt_kl((overall.get("gen_kl_prefix_mean") or {}).get("value"))
                if overall
                else "baseline",
            ]
        )
    return _table(headers, rows)


def _slice_tables(data: Mapping[str, Any]) -> list[str]:
    """Render one per-slice table for each candidate.

    Args:
        data: Report data.

    Returns:
        Lines of all tables, or a note when there are none.
    """
    lines: list[str] = []
    for config in data["configs"]:
        if config["role"] != "candidate" or not config.get("slices"):
            continue
        lines += [f"### `{config['name']}`", ""]
        rows: list[list[str]] = []
        for entry in config["slices"]:
            agg = entry["aggregate"]
            lo, hi = entry["bounds"]
            checks = entry["checks"]
            rows.append(
                [
                    f"[{lo}, {hi})",
                    str(entry["n_items"]),
                    _interval(agg.get("flip_rate"), fmt_pct),
                    _interval(agg.get("prefill_kl_mean"), fmt_kl),
                    _interval(agg.get("prefill_top1_agreement"), fmt_pct),
                    f"flip: {checks.get('flip_rate')}; KL: {checks.get('prefill_kl_mean')}; "
                    f"top-1: {checks.get('prefill_top1_agreement')}",
                ]
            )
        lines += _table(
            ["Slice", "n", "Flip rate [CI]", "Prefill KL [CI]", "Top-1 [CI]", "Checks"], rows
        )
        lines.append("")
    return lines or ["No candidate has per-slice results."]


def _gate_failures(data: Mapping[str, Any]) -> list[str]:
    """List every gate failure and every insufficient or uncovered slice.

    Args:
        data: Report data.

    Returns:
        Markdown bullet lines.
    """
    lines: list[str] = []
    for config in data["configs"]:
        for failure in config.get("failures") or []:
            lo, hi = failure["bounds"]
            lines.append(
                f"- `{config['name']}`: slice [{lo}, {hi}), {failure['metric']} "
                f"{failure['side']} bound {failure['bound']:.4g} vs threshold "
                f"{failure['threshold']:.4g}"
            )
        for item in config.get("insufficient") or []:
            lo, hi = item["bounds"]
            lines.append(
                f"- `{config['name']}`: slice [{lo}, {hi}) has {item['n']} items for "
                f"{item['metric']} (need {item['required']})"
            )
        for item in config.get("not_covered") or []:
            if "metric" in item:
                lo, hi = item["bounds"]
                lines.append(
                    f"- `{config['name']}`: slice [{lo}, {hi}) is not covered for {item['metric']}"
                )
        if config.get("error"):
            lines.append(f"- `{config['name']}`: ERROR: {config['error']}")
    return lines or ["No gate failures."]


def _spec_decode(data: Mapping[str, Any]) -> list[str]:
    """Render speculative-decoding acceptance per configuration.

    Args:
        data: Report data.

    Returns:
        Markdown lines.
    """
    lines: list[str] = []
    for config in data["configs"]:
        spec = config.get("spec_decode")
        if spec:
            lines.append(
                f"- `{config['name']}`: acceptance rate {fmt_pct(spec['acceptance_rate'])}, "
                f"mean acceptance length {float(spec['mean_acceptance_length']):.2f}"
            )
    return lines or ["No speculative-decoding counters were observed in any configuration."]


def _noise_section(data: Mapping[str, Any]) -> list[str]:
    """Render the noise floor, the candidate determinism checks, and the reference self-check.

    Args:
        data: Report data.

    Returns:
        Markdown lines.
    """
    lines: list[str] = []
    floor = data.get("noise_floor")
    if floor:
        lines.append(
            f"- Reference against itself (concurrency {floor['concurrency_a']} vs "
            f"{floor['concurrency_b']}, {floor['num_prompts']} prompts): exact match "
            f"{fmt_pct(floor.get('exact_match_rate'))}, prefix KL "
            f"{fmt_kl(floor.get('gen_kl_prefix_mean'))}, flip rate "
            f"{fmt_pct(floor.get('flip_rate'))} "
            f"[{fmt_pct(floor.get('flip_rate_lower'))}, {fmt_pct(floor.get('flip_rate_upper'))}]."
        )
    else:
        lines.append("- Noise floor was not measured.")
    for config in data["configs"]:
        det = config.get("determinism")
        if det:
            lines.append(
                f"- `{config['name']}` determinism (concurrency {det['concurrency_a']} vs "
                f"{det['concurrency_b']}): identical-output rate "
                f"{fmt_pct(det.get('exact_match_rate'))}."
            )
    lines.append(
        "- Reference prefill-vs-decode self-consistency KL (mean per prompt): "
        f"{fmt_kl(data.get('reference_self_consistency_kl'))} nats."
    )
    return lines


def _budget_section(data: Mapping[str, Any]) -> list[str]:
    """Render the thresholds and the pre-registration sentence.

    Args:
        data: Report data.

    Returns:
        Markdown lines.
    """
    budget = data["budget"]
    slo = data["slo"]
    run = data["run"]
    return [
        f"- Flip rate upper bound must stay at or below {fmt_pct(budget['flip_rate_max'])}.",
        f"- Prefill KL mean upper bound must stay at or below "
        f"{fmt_kl(budget['kl_mean_max'])} nats.",
        f"- Prefill top-1 agreement lower bound must stay at or above "
        f"{fmt_pct(budget['top1_agreement_min'])}.",
        f"- At least {budget['min_prompts_per_slice']} items per slice for each gated metric; "
        f"{budget['confidence']:.0%} two-sided intervals, {budget['bootstrap_resamples']} "
        "bootstrap resamples.",
        f"- SLO at target concurrency {run['target_concurrency']}: p99 TTFT ≤ "
        f"{fmt_ms(slo['ttft_p99_ms'])} ms and p99 TPOT ≤ {fmt_ms(slo['tpot_p99_ms'])} ms.",
        f"- Thresholds were fixed in the configuration before the run (config hash "
        f"`{run['config_hash']}`).",
    ]


def _timing_text(clock: object) -> str:
    """Describe the clock that timed the requests of a run.

    Args:
        clock: ``"virtual"`` or ``"wall"``, or ``None`` when the run recorded none.

    Returns:
        A short description for the card.
    """
    if clock == "virtual":
        return "simulated on the fake server's virtual clock, not measured on hardware"
    if clock == "wall":
        return "wall clock (`time.perf_counter_ns`)"
    return "n/a"


def _environment_section(data: Mapping[str, Any]) -> list[str]:
    """Render the environment, timing, and GPU time per configuration.

    Args:
        data: Report data.

    Returns:
        Markdown lines.
    """
    run = data["run"]
    env = data.get("environment") or {}
    gpus = env.get("gpus") or []
    lines = [
        f"- GPU: {', '.join(g['name'] for g in gpus) if gpus else run.get('hardware_label')}",
        f"- Driver: {gpus[0]['driver_version'] if gpus else 'n/a'}; "
        f"CUDA: {env.get('cuda_version') or 'n/a'}",
        f"- Server: {env.get('server_version') or 'n/a'} (backend {env.get('backend_kind')})",
        f"- Timing: {_timing_text(run.get('timing_clock'))}",
        f"- day0 {run['day0_version']}; git `{run.get('git_sha') or 'n/a'}`",
        f"- Started {run['started_utc']}; ended {run['ended_utc']}",
    ]
    for config in data["configs"]:
        seconds = config.get("duration_s")
        if seconds is not None:
            lines.append(
                f"- `{config['name']}`: {float(seconds):.1f} s wall time, "
                f"{float(seconds) * run['num_gpus'] / 3600:.3f} GPU-hours"
            )
    return lines
