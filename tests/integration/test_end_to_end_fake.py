"""End-to-end: ``day0 run examples/fake.yaml``, the report, and the CI gate on the fake server.

The expected outcomes are those in §18.1 of the specification. They are locked here so that any
change to the fake model, the metrics, or the gate that moves an outcome fails the build.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from day0.cli import app

EXPECTED_STATUS = {
    "identical": "PASS",
    "fast-ok": "PASS",
    "fast-noisy": "FAIL_QUALITY",
    "longctx-bad": "FAIL_QUALITY",
    "slow": "FAIL_SLO",
    "spec-fake": "PASS",
}
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _run_dir_from(output: str) -> Path:
    """Read the run directory from the CLI output.

    Args:
        output: Standard output of ``day0 run``.

    Returns:
        The run directory named on the first line that starts with ``run directory:``.
    """
    for line in output.splitlines():
        if line.startswith("run directory: "):
            return Path(line.removeprefix("run directory: ").strip())
    raise AssertionError(f"no run directory in output:\n{output}")


@pytest.fixture(scope="module")
def demo_run(examples_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the fake demo once for every test in this module.

    Args:
        examples_dir: Directory with the example configurations.
        tmp_path_factory: Factory for module-scoped temporary directories.

    Returns:
        The run directory.
    """
    out = tmp_path_factory.mktemp("demo")
    result = CliRunner().invoke(app, ["run", str(examples_dir / "fake.yaml"), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "recommendation: fast-ok" in result.output
    return _run_dir_from(result.output)


def _card(run_dir: Path) -> dict[str, Any]:
    """Load ``card.json``.

    Args:
        run_dir: Run directory.

    Returns:
        The decoded card data.
    """
    return json.loads((run_dir / "card.json").read_text(encoding="utf-8"))


def test_every_candidate_has_its_specified_status(demo_run: Path) -> None:
    """Each candidate receives the status that §18.1 specifies.

    Args:
        demo_run: The demo run directory.
    """
    statuses = {
        c["name"]: c["status"] for c in _card(demo_run)["configs"] if c["role"] == "candidate"
    }
    assert statuses == EXPECTED_STATUS


def test_fast_ok_is_the_recommendation(demo_run: Path) -> None:
    """The fastest candidate that passes the gates and meets the SLO is recommended.

    Args:
        demo_run: The demo run directory.
    """
    assert _card(demo_run)["recommendation"]["name"] == "fast-ok"


def test_long_context_failure_is_confined_to_the_long_slice(demo_run: Path) -> None:
    """``longctx-bad`` fails only in the slice of 2048 tokens and above.

    Args:
        demo_run: The demo run directory.
    """
    config = next(c for c in _card(demo_run)["configs"] if c["name"] == "longctx-bad")
    failing_slices = {failure["slice"] for failure in config["failures"]}
    assert failing_slices == {2}


def test_spec_decode_summary_is_reported(demo_run: Path) -> None:
    """``spec-fake`` reports an acceptance rate close to its configured 0.6.

    Args:
        demo_run: The demo run directory.
    """
    config = next(c for c in _card(demo_run)["configs"] if c["name"] == "spec-fake")
    assert config["spec_decode"] is not None
    assert config["spec_decode"]["acceptance_rate"] == pytest.approx(0.6, abs=0.1)


def test_artifacts_and_card_files_exist(demo_run: Path) -> None:
    """Every artifact named in §15.2 is present, and pareto.png is a real PNG.

    Args:
        demo_run: The demo run directory.
    """
    for name in ("run.json", "config.resolved.yaml", "env.json", "card.md", "card.json"):
        assert (demo_run / name).is_file(), name
    for name in (
        "quality.jsonl.gz",
        "perf.json",
        "noise_floor.json",
        "perf_workload.jsonl.gz",
        "fingerprint.json",
        "status.json",
    ):
        assert (demo_run / "reference" / name).is_file(), name
    for candidate in EXPECTED_STATUS:
        cand_dir = demo_run / "candidates" / candidate
        for name in ("quality.jsonl.gz", "perf.json", "status.json", "summary.json", "server.log"):
            assert (cand_dir / name).is_file(), f"{candidate}/{name}"
    assert (demo_run / "pareto.png").read_bytes().startswith(PNG_MAGIC)


def test_report_reproduces_the_card_byte_for_byte(demo_run: Path) -> None:
    """``day0 report`` regenerates ``card.md`` identically from the artifacts alone.

    Args:
        demo_run: The demo run directory.
    """
    before = (demo_run / "card.md").read_bytes()
    result = CliRunner().invoke(app, ["report", str(demo_run)])
    assert result.exit_code == 0, result.output
    assert (demo_run / "card.md").read_bytes() == before


def test_check_gates_fast_ok_and_fast_noisy(
    examples_dir: Path, demo_run: Path, tmp_path: Path
) -> None:
    """``day0 check`` returns 0 for a passing candidate and 1 for a failing one.

    Args:
        examples_dir: Directory with the example configurations.
        demo_run: The demo run directory.
        tmp_path: Pytest's per-test temporary directory.
    """
    reference = demo_run / "reference"
    config = str(examples_dir / "fake.yaml")
    passing = CliRunner().invoke(
        app,
        [
            "check",
            config,
            "--candidate",
            "fast-ok",
            "--reference",
            str(reference),
            "--out",
            str(tmp_path),
        ],
    )
    assert passing.exit_code == 0, passing.output
    failing = CliRunner().invoke(
        app,
        [
            "check",
            config,
            "--candidate",
            "fast-noisy",
            "--reference",
            str(reference),
            "--out",
            str(tmp_path),
        ],
    )
    assert failing.exit_code == 1, failing.output
