"""Deterministic filler text built from a small built-in English word list.

Used by the perf workload, the synthetic suites, and the fake suites. Every function takes an
explicit :class:`numpy.random.Generator`, so the same seed always yields the same text.
"""

from __future__ import annotations

import numpy as np

WORDLIST: tuple[str, ...] = (
    "time",
    "year",
    "people",
    "way",
    "day",
    "man",
    "thing",
    "woman",
    "life",
    "child",
    "world",
    "school",
    "state",
    "family",
    "student",
    "group",
    "country",
    "problem",
    "hand",
    "part",
    "place",
    "case",
    "week",
    "company",
    "system",
    "program",
    "question",
    "work",
    "government",
    "number",
    "night",
    "point",
    "home",
    "water",
    "room",
    "mother",
    "area",
    "money",
    "story",
    "fact",
    "month",
    "lot",
    "right",
    "study",
    "book",
    "eye",
    "job",
    "word",
    "business",
    "issue",
    "side",
    "kind",
    "head",
    "house",
    "service",
    "friend",
    "father",
    "power",
    "hour",
    "game",
    "line",
    "end",
    "member",
    "law",
    "car",
    "city",
    "community",
    "name",
    "team",
    "minute",
    "idea",
    "body",
    "information",
    "back",
    "parent",
    "face",
    "level",
    "office",
    "door",
    "health",
    "art",
    "history",
    "party",
    "result",
    "change",
    "morning",
    "reason",
    "research",
    "girl",
    "moment",
    "air",
    "teacher",
    "force",
    "education",
    "foot",
    "age",
    "policy",
    "process",
    "music",
    "market",
    "sense",
    "nation",
    "plan",
    "college",
    "interest",
    "experience",
    "effect",
    "use",
    "class",
    "control",
    "care",
    "field",
    "development",
    "role",
    "effort",
    "rate",
    "heart",
    "show",
    "leader",
    "light",
    "voice",
    "mind",
    "price",
    "report",
    "decision",
    "view",
    "relationship",
    "town",
    "road",
    "value",
    "building",
    "action",
    "model",
    "season",
    "society",
    "director",
    "position",
    "record",
    "paper",
    "space",
    "ground",
    "form",
    "event",
    "matter",
    "center",
    "project",
    "activity",
    "need",
    "court",
    "situation",
    "cost",
    "industry",
    "figure",
    "street",
    "image",
    "phone",
    "data",
    "practice",
    "piece",
    "land",
    "product",
    "doctor",
    "wall",
    "river",
    "garden",
    "bridge",
    "window",
    "forest",
    "mountain",
    "harbor",
    "lantern",
    "compass",
    "orchard",
)


def filler_sentence(rng: np.random.Generator, min_words: int = 8, max_words: int = 18) -> str:
    """Build one capitalized sentence of random words from :data:`WORDLIST`.

    Args:
        rng: Seeded random generator.
        min_words: Smallest number of words in the sentence.
        max_words: Largest number of words in the sentence.

    Returns:
        A sentence that ends with a period.
    """
    count = int(rng.integers(min_words, max_words + 1))
    indices = rng.integers(0, len(WORDLIST), size=count)
    words = [WORDLIST[int(index)] for index in indices]
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def filler_text(rng: np.random.Generator, min_chars: int) -> str:
    """Concatenate filler sentences until the text has at least ``min_chars`` characters.

    Args:
        rng: Seeded random generator.
        min_chars: Minimum length of the returned text in characters.

    Returns:
        Space-separated sentences. The text may be longer than ``min_chars``.
    """
    sentences: list[str] = []
    total = 0
    while total < min_chars:
        sentence = filler_sentence(rng)
        sentences.append(sentence)
        total += len(sentence) + 1
    return " ".join(sentences)
