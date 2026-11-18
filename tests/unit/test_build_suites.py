"""Tests for the suite builder's GSM8K sampling and its deterministic outputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_suites.py"


def _load_builder() -> ModuleType:
    """Import ``scripts/build_suites.py`` as a module.

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location("build_suites", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(count: int) -> list[dict[str, str]]:
    """Synthetic rows in the GSM8K format, with thousands separators in the answers.

    Args:
        count: Number of rows.

    Returns:
        Rows with ``question`` and ``answer`` fields.
    """
    return [
        {"question": f"Question {index}?", "answer": f"Working.\n#### {index * 1000:,}"}
        for index in range(count)
    ]


def test_gsm8k_sample_has_200_items_with_clean_numeric_gold() -> None:
    """The sample has the specified size, numeric answer type, and no separators in gold."""
    builder = _load_builder()
    items = builder._gsm8k_items(_rows(300), np.random.default_rng(1234))
    assert len(items) == 200
    assert items[0]["id"] == "gsm8k-0001"
    assert all(item["answer_type"] == "number" for item in items)
    assert all("," not in str(item["gold"]) for item in items)
    assert all(str(item["gold"]).isdigit() for item in items)


def test_gsm8k_sample_is_deterministic() -> None:
    """The same rows and seed always give the same items."""
    builder = _load_builder()
    first = builder._gsm8k_items(_rows(300), np.random.default_rng(1234))
    second = builder._gsm8k_items(_rows(300), np.random.default_rng(1234))
    assert first == second


def test_gsm8k_sample_rejects_small_splits() -> None:
    """A split smaller than the sample size is an error, not a silently short suite."""
    builder = _load_builder()
    with pytest.raises(ValueError, match="need at least 200"):
        builder._gsm8k_items(_rows(10), np.random.default_rng(1234))


def test_gsm8k_jsonl_reader_skips_blank_lines(tmp_path: Path) -> None:
    """A local JSONL copy is read line by line, ignoring blank lines.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    builder = _load_builder()
    path = tmp_path / "test.jsonl"
    path.write_text('{"question": "q", "answer": "a\\n#### 7"}\n\n', encoding="utf-8")
    assert builder._read_gsm8k_jsonl(path) == [{"question": "q", "answer": "a\n#### 7"}]


def test_longctx_suite_has_the_specified_shape() -> None:
    """The synthetic long-context suite has 100 items, half single-fact and half sum questions."""
    builder = _load_builder()
    items = builder._longctx_items(np.random.default_rng(1234))
    assert len(items) == 100
    kinds = [item["tags"][0] for item in items]  # type: ignore[index]
    assert kinds.count("single") == 50 and kinds.count("sum") == 50
