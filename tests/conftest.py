"""Shared pytest fixtures for the day0 test suite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root directory.

    Returns:
        Absolute path of the directory that contains ``pyproject.toml``.
    """
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def examples_dir(repo_root: Path) -> Path:
    """Return the directory that holds the example YAML configurations.

    Args:
        repo_root: Repository root fixture.

    Returns:
        Absolute path of ``examples/``.
    """
    return repo_root / "examples"
