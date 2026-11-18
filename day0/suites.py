"""Evaluation suites: loading JSONL items, answer extraction, slices, and tokenizer probes."""

from __future__ import annotations

import hashlib
import json
import re
import string
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from day0.errors import ConfigError

AnswerType = Literal["number", "exact"]
_ANSWER_IS = re.compile(r"answer is\s*:?\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?[\d,]*\.?\d+")

# Ten fixed chat prompts: ASCII, Unicode, emoji, code, numbers, whitespace, and a long string.
# Their token ids fingerprint the tokenizer (§9.3). Do not change them: that breaks every stored
# fingerprint.
TOKENIZER_PROBES: tuple[list[dict[str, str]], ...] = (
    [{"role": "user", "content": "Hello, world!"}],
    [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
    [{"role": "user", "content": "Café déjà vu, naïve résumé: 北京 東京 서울."}],
    [{"role": "user", "content": "Emoji: 🙂🚀🔥 and symbols ∑ ∞ ≠ ≤."}],
    [{"role": "user", "content": "def f(x):\n    return x ** 2  # square\n"}],
    [{"role": "user", "content": "Numbers: 0 1 -2 3.14 1,000,000 6.02e23 0xFF"}],
    [
        {"role": "assistant", "content": "Assistant turn first."},
        {"role": "user", "content": "Follow-up?"},
    ],
    [{"role": "user", "content": "Tabs\tand\r\nnewlines\n  and   spaces"}],
    [{"role": "user", "content": "Long: " + "lorem ipsum dolor sit amet " * 20}],
    [{"role": "user", "content": "Quotes \"double\" and 'single' and `code` and \\ slash"}],
)


@dataclass(frozen=True)
class SuiteItem:
    """One evaluation item.

    Attributes:
        id: Unique id across all suites.
        suite: Name of the suite file (its stem).
        messages: OpenAI-style chat messages.
        gold: Expected answer, or ``None`` for open-ended items.
        answer_type: ``number``, ``exact``, or ``None``. Required when ``gold`` is set.
        tags: Free-form labels.
    """

    id: str
    suite: str
    messages: list[dict[str, str]]
    gold: str | None
    answer_type: AnswerType | None
    tags: tuple[str, ...] = field(default_factory=tuple)


def _parse_item(raw: Mapping[str, Any], suite: str, where: str) -> SuiteItem:
    """Validate one decoded JSON object and build a :class:`SuiteItem`.

    Args:
        raw: Decoded JSON object.
        suite: Suite name.
        where: Location used in error messages.

    Returns:
        The validated item.

    Raises:
        ConfigError: If a field is missing or has the wrong type.
    """
    item_id = raw.get("id")
    if not isinstance(item_id, str) or not item_id:
        raise ConfigError(f"{where}: 'id' must be a non-empty string")
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ConfigError(f"{where}: 'messages' must be a non-empty list")
    parsed_messages: list[dict[str, str]] = []
    for message in messages:
        if (
            not isinstance(message, Mapping)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
        ):
            raise ConfigError(f"{where}: every message needs string 'role' and 'content'")
        parsed_messages.append({"role": str(message["role"]), "content": str(message["content"])})
    gold = raw.get("gold")
    if gold is not None and not isinstance(gold, str):
        raise ConfigError(f"{where}: 'gold' must be a string or null")
    answer_type = raw.get("answer_type")
    if answer_type not in (None, "number", "exact"):
        raise ConfigError(f"{where}: 'answer_type' must be 'number', 'exact', or null")
    if gold is not None and answer_type is None:
        raise ConfigError(f"{where}: an item with a 'gold' value needs an 'answer_type'")
    tags = raw.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ConfigError(f"{where}: 'tags' must be a list of strings")
    return SuiteItem(
        id=item_id,
        suite=suite,
        messages=parsed_messages,
        gold=gold,
        answer_type=answer_type,
        tags=tuple(tags),
    )


def load_suite(path: Path) -> list[SuiteItem]:
    """Load one suite file.

    Args:
        path: JSONL file with one item per line. Blank lines are ignored.

    Returns:
        The items in file order.

    Raises:
        ConfigError: If the file cannot be read, a line is not valid JSON, an item is invalid, or
            an id repeats within the file.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"cannot read suite {path}: {exc}") from exc
    suite = path.stem
    items: list[SuiteItem] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            raw: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{where}: invalid JSON ({exc.msg})") from exc
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{where}: each line must be a JSON object")
        item = _parse_item(raw, suite, where)
        if item.id in seen:
            raise ConfigError(f"{where}: duplicate item id {item.id!r}")
        seen.add(item.id)
        items.append(item)
    return items


def load_suites(paths: Sequence[Path]) -> list[SuiteItem]:
    """Load several suites and check that suite names and item ids are unique across them.

    Args:
        paths: Suite files in configuration order.

    Returns:
        All items, suite by suite.

    Raises:
        ConfigError: If two suites share a file stem or an item id.
    """
    stems = [path.stem for path in paths]
    duplicate_stems = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicate_stems:
        raise ConfigError(f"suite file names must be unique; duplicated: {duplicate_stems}")
    items: list[SuiteItem] = []
    ids: set[str] = set()
    for path in paths:
        for item in load_suite(path):
            if item.id in ids:
                raise ConfigError(f"item id {item.id!r} appears in more than one suite")
            ids.add(item.id)
            items.append(item)
    return items


def suites_hash(paths: Sequence[Path]) -> str:
    """SHA-256 over the names and contents of the suite files, in order.

    Args:
        paths: Suite files.

    Returns:
        A 64-character lowercase hexadecimal digest.
    """
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\n")
        hasher.update(path.read_bytes())
        hasher.update(b"\n")
    return hasher.hexdigest()


def _normalize_number(raw: str) -> Decimal | None:
    """Convert a matched number string to a :class:`Decimal`.

    Args:
        raw: Matched text, possibly with thousands separators or a trailing period.

    Returns:
        The number, or ``None`` if it cannot be parsed.
    """
    cleaned = raw.replace(",", "").strip().rstrip(".")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def extract_number(text: str) -> Decimal | None:
    """Extract the final numeric answer from model output.

    The first match of "answer is N" wins. Otherwise the last number in the text is used.

    Args:
        text: Generated text.

    Returns:
        The number, or ``None`` when no number can be read.
    """
    match = _ANSWER_IS.search(text)
    if match is not None:
        return _normalize_number(match.group(1))
    numbers = _NUMBER.findall(text)
    if not numbers:
        return None
    return _normalize_number(numbers[-1])


def _normalize_exact(text: str) -> str:
    """Normalize text for exact-match comparison.

    Args:
        text: Input text.

    Returns:
        Text with surrounding whitespace and trailing punctuation removed, case-folded.
    """
    return text.strip().rstrip(string.punctuation).strip().casefold()


def check_answer(output_text: str, gold: str | None, answer_type: str | None) -> bool | None:
    """Check a generated answer against a gold label.

    Args:
        output_text: Generated text.
        gold: Gold answer, or ``None`` for open-ended items.
        answer_type: ``number`` or ``exact``.

    Returns:
        ``None`` when there is no gold label, otherwise whether the answer matches.
    """
    if gold is None:
        return None
    if answer_type == "number":
        expected = _normalize_number(gold)
        found = extract_number(output_text)
        return expected is not None and found is not None and found == expected
    return _normalize_exact(output_text) == _normalize_exact(gold)


def is_correct(output_text: str, item: SuiteItem) -> bool | None:
    """Check a generated answer against the gold label of an item.

    Args:
        output_text: Generated text.
        item: The evaluation item.

    Returns:
        ``None`` when the item has no gold label, otherwise whether the answer matches.
    """
    return check_answer(output_text, item.gold, item.answer_type)


def assign_slice(n_tokens: int, boundaries: Sequence[int]) -> int:
    """Return the slice index of a prompt length.

    Slice ``i`` is the half-open interval ``[boundaries[i], boundaries[i + 1])``.

    Args:
        n_tokens: Prompt length in tokens.
        boundaries: Ascending boundaries that start at 0.

    Returns:
        The slice index.

    Raises:
        ValueError: If the length is outside the boundaries.
    """
    if n_tokens < boundaries[0]:
        raise ValueError(f"{n_tokens} tokens is below the first boundary {boundaries[0]}")
    index = bisect_right(boundaries, n_tokens) - 1
    if index >= len(boundaries) - 1:
        raise ValueError(f"{n_tokens} tokens is at or above the last boundary {boundaries[-1]}")
    return index


def tokenizer_fingerprint(token_lists: Sequence[Sequence[int]]) -> str:
    """SHA-256 over the token ids of the tokenizer probes.

    Args:
        token_lists: One token-id list per probe, in probe order.

    Returns:
        A 64-character lowercase hexadecimal digest.
    """
    payload = json.dumps([list(ids) for ids in token_lists], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
