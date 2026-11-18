"""Tests for the card renderer and the Pareto chart.

The snapshot test renders ``tests/fixtures/card_data.json`` (the report data of a fake demo run)
and compares it with ``tests/fixtures/expected_card.md``. Set ``UPDATE_SNAPSHOTS=1`` to regenerate
the expected file after an intended change.
"""

from __future__ import annotations

import copy
import json
import os
import struct
from pathlib import Path
from typing import Any

import pytest

from day0.report.card import fmt_kl, fmt_ms, fmt_pct, fmt_ratio, render_card
from day0.report.pareto import render_pareto

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
CARD_DATA = FIXTURES / "card_data.json"
EXPECTED_CARD = FIXTURES / "expected_card.md"
REQUIRED_HEADINGS = (
    "## Main table",
    "## Per-slice results",
    "## Gate failures",
    "## Speculative decoding",
    "## Noise floor and determinism",
    "## Budget and SLO",
    "## Environment",
    "## Reproduce",
    "## Limitations",
)


def _data() -> dict[str, Any]:
    """Load the card fixture.

    Returns:
        The report data.
    """
    return json.loads(CARD_DATA.read_text(encoding="utf-8"))


def test_card_matches_snapshot() -> None:
    """The rendered card is byte-identical to the committed snapshot."""
    rendered = render_card(_data())
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        EXPECTED_CARD.write_text(rendered, encoding="utf-8")
    assert rendered == EXPECTED_CARD.read_text(encoding="utf-8")


def test_card_has_every_required_section() -> None:
    """Every section named in §17.1 appears in the card."""
    rendered = render_card(_data())
    for heading in REQUIRED_HEADINGS:
        assert heading in rendered, heading
    assert rendered.startswith("# Serving Profile Card — ")


def test_card_states_the_pre_registration_sentence() -> None:
    """The budget section says the thresholds were fixed before the run, with the config hash."""
    data = _data()
    rendered = render_card(data)
    assert "Thresholds were fixed in the configuration before the run (config hash" in rendered
    assert data["run"]["config_hash"] in rendered


def test_card_without_recommendation_gives_the_reason() -> None:
    """With no recommendation, the box shows the reason and no configuration name."""
    data = copy.deepcopy(_data())
    data["recommendation"] = {
        "name": None,
        "reason": "no configuration meets the SLO at target concurrency",
    }
    rendered = render_card(data)
    assert "**No recommendation.** no configuration meets the SLO at target concurrency" in rendered


@pytest.mark.parametrize(
    ("formatter", "value", "expected"),
    [
        (fmt_ratio, 1.2345, "1.23×"),
        (fmt_ms, 245.04, "245.0"),
        (fmt_pct, 0.0120, "1.20%"),
        (fmt_kl, 0.00123, "1.2e-03"),
        (fmt_pct, None, "n/a"),
        (fmt_kl, float("nan"), "n/a"),
    ],
)
def test_number_formats_follow_the_specification(formatter: Any, value: Any, expected: str) -> None:
    """Throughput ×2 decimals, ms ×1 decimal, percentages ×2, KL in scientific form ×2 digits.

    Args:
        formatter: Formatter under test.
        value: Input value.
        expected: Expected text.
    """
    assert formatter(value) == expected


def test_pareto_png_has_the_specified_size(tmp_path: Path) -> None:
    """The chart is 1600 × 1000 pixels and is a PNG.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    path = tmp_path / "pareto.png"
    render_pareto(
        path,
        points=[
            {
                "name": "ref",
                "x": 1.0,
                "flip_rate": 0.0,
                "flip_lower": 0.0,
                "flip_upper": 0.0,
                "status": "PASS",
            },
            {
                "name": "bad",
                "x": 1.4,
                "flip_rate": 0.2,
                "flip_lower": 0.14,
                "flip_upper": 0.27,
                "status": "FAIL_QUALITY",
            },
        ],
        front=["ref", "bad"],
        recommended="ref",
        flip_rate_max=0.05,
        title="unit test",
    )
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", header[16:24])
    assert (width, height) == (1600, 1000)
