"""Smoke tests for the package skeleton."""

from __future__ import annotations

import re

from typer.testing import CliRunner

import day0
from day0.cli import app


def test_version_is_semver() -> None:
    """The package exposes a MAJOR.MINOR.PATCH version string."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", day0.__version__)


def test_cli_version_flag_prints_version() -> None:
    """``day0 --version`` prints the package version and exits 0."""
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"day0 {day0.__version__}"
