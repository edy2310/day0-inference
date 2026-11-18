"""Tests for suite loading, answer extraction, slices, and the tokenizer fingerprint."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from day0.errors import ConfigError
from day0.suites import (
    TOKENIZER_PROBES,
    SuiteItem,
    assign_slice,
    extract_number,
    is_correct,
    load_suite,
    load_suites,
    suites_hash,
    tokenizer_fingerprint,
)


def _item(gold: str | None, answer_type: str | None) -> SuiteItem:
    """Build a suite item for correctness tests.

    Args:
        gold: Gold answer.
        answer_type: Answer type.

    Returns:
        The item.
    """
    return SuiteItem(
        id="t-1",
        suite="t",
        messages=[{"role": "user", "content": "q"}],
        gold=gold,
        answer_type=answer_type,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is 72.", Decimal("72")),
        ("so the answer is: $1,234.50", Decimal("1234.50")),
        ("The answer is -3.", Decimal("-3")),
        ("I tried 5, then 7 and finally 9", Decimal("9")),
        ("Answer is 0.5.", Decimal("0.5")),
        ("12 apples and 3.5 kg of rice", Decimal("3.5")),
        ("Total: 1,000", Decimal("1000")),
        ("total is $8", Decimal("8")),
        ("answer IS 4", Decimal("4")),
        ("no digits here", None),
        ("", None),
    ],
)
def test_extract_number_cases(text: str, expected: Decimal | None) -> None:
    """Number extraction handles commas, signs, decimals, dollar signs, and missing numbers.

    Args:
        text: Model output.
        expected: Expected number, or ``None``.
    """
    assert extract_number(text) == expected


def test_number_correctness_is_numeric_not_textual() -> None:
    """``72.0`` and ``72`` are the same answer for number items."""
    item = _item("72", "number")
    assert is_correct("The answer is 72.0.", item) is True
    assert is_correct("The answer is 73.", item) is False
    assert is_correct("I have no idea.", item) is False


def test_exact_correctness_ignores_case_and_trailing_punctuation() -> None:
    """Exact matching strips whitespace and trailing punctuation and ignores case."""
    item = _item("Paris", "exact")
    assert is_correct("  paris. ", item) is True
    assert is_correct("London", item) is False


def test_items_without_gold_have_no_correctness() -> None:
    """Items without a gold label return ``None``, so they never contribute to flips."""
    assert is_correct("anything", _item(None, None)) is None


def test_assign_slice_boundaries_are_half_open() -> None:
    """A length equal to a boundary belongs to the slice that starts there."""
    boundaries = [0, 2048, 8192, 40960]
    assert assign_slice(0, boundaries) == 0
    assert assign_slice(2047, boundaries) == 0
    assert assign_slice(2048, boundaries) == 1
    assert assign_slice(8191, boundaries) == 1
    assert assign_slice(8192, boundaries) == 2
    assert assign_slice(40959, boundaries) == 2
    with pytest.raises(ValueError):
        assign_slice(40960, boundaries)


def test_load_suite_parses_items_and_skips_blank_lines(tmp_path: Path) -> None:
    """A valid file loads in order, and blank lines are ignored.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    path = tmp_path / "demo.jsonl"
    rows = [
        {
            "id": "a-1",
            "messages": [{"role": "user", "content": "x"}],
            "gold": "1",
            "answer_type": "number",
            "tags": ["math"],
        },
        {
            "id": "a-2",
            "messages": [{"role": "user", "content": "y"}],
            "gold": None,
            "answer_type": None,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")
    items = load_suite(path)
    assert [item.id for item in items] == ["a-1", "a-2"]
    assert items[0].suite == "demo"
    assert items[0].tags == ("math",)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"id": "", "messages": [{"role": "user", "content": "x"}]}, "'id'"),
        ({"id": "a", "messages": []}, "'messages'"),
        ({"id": "a", "messages": [{"role": "user"}]}, "'role' and 'content'"),
        (
            {"id": "a", "messages": [{"role": "user", "content": "x"}], "gold": "1"},
            "needs an 'answer_type'",
        ),
        (
            {"id": "a", "messages": [{"role": "user", "content": "x"}], "answer_type": "yes"},
            "answer_type",
        ),
    ],
)
def test_load_suite_rejects_bad_items(tmp_path: Path, row: dict[str, object], message: str) -> None:
    """Each malformed item fails with a message that names the problem.

    Args:
        tmp_path: Pytest's per-test temporary directory.
        row: Malformed item.
        message: Expected substring of the error.
    """
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_suite(path)


def test_load_suite_reports_line_number_for_invalid_json(tmp_path: Path) -> None:
    """Invalid JSON names the file and the line.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    path = tmp_path / "broken.jsonl"
    path.write_text(
        '{"id": "a", "messages": [{"role": "user", "content": "x"}]}\n{not json\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match=r"broken\.jsonl:2"):
        load_suite(path)


def test_ids_must_be_unique_across_suites(tmp_path: Path) -> None:
    """The same item id in two suites is rejected, and so are duplicate suite names.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    row = json.dumps({"id": "dup", "messages": [{"role": "user", "content": "x"}]})
    first = tmp_path / "one" / "shared.jsonl"
    second = tmp_path / "two" / "other.jsonl"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text(row + "\n", encoding="utf-8")
    second.write_text(row + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="more than one suite"):
        load_suites([first, second])
    twin = tmp_path / "three" / "shared.jsonl"
    twin.parent.mkdir()
    twin.write_text(
        json.dumps({"id": "b", "messages": [{"role": "user", "content": "x"}]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="must be unique"):
        load_suites([first, twin])


def test_suites_hash_changes_with_content(tmp_path: Path) -> None:
    """The suite hash depends on file contents and on file names.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    path = tmp_path / "s.jsonl"
    path.write_text("one\n", encoding="utf-8")
    before = suites_hash([path])
    path.write_text("two\n", encoding="utf-8")
    assert suites_hash([path]) != before


def test_tokenizer_probes_are_ten_fixed_messages() -> None:
    """There are exactly ten probes, and each is a non-empty chat."""
    assert len(TOKENIZER_PROBES) == 10
    assert all(probe and probe[0]["role"] for probe in TOKENIZER_PROBES)


def test_tokenizer_fingerprint_is_stable_and_order_sensitive() -> None:
    """The fingerprint is a pure function of the token lists in order.

    The second assertion checks that it is order sensitive.
    """
    lists = [[1, 2], [3]]
    assert tokenizer_fingerprint(lists) == tokenizer_fingerprint([[1, 2], [3]])
    assert tokenizer_fingerprint(lists) != tokenizer_fingerprint([[3], [1, 2]])
