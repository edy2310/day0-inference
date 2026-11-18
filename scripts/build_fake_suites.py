"""Build ``examples/fake_suites/echo_mixed.jsonl`` for the fake-server demo.

The output is deterministic: the same seed always produces byte-identical files. The suite has 150
items. Each item carries an ``<<ECHO:n>>`` marker, so the fake server answers "The answer is n."
and the gold label is ``n``. Prompt lengths are spread over three buckets (short, medium, long),
and each bucket maps to a different slice of ``examples/fake.yaml``.

Usage:
    python scripts/build_fake_suites.py [--output PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from day0.corpus import filler_text

SEED = 4242
ITEMS_PER_BUCKET = 50
# (bucket name, smallest and largest total content length in characters).
# Each bucket keeps its tokens inside one slice of [0, 512, 2048, 65536) after the ~25-token chat
# template overhead is added.
BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("short", 100, 420),
    ("medium", 600, 1900),
    ("long", 2100, 11000),
)
QUESTION = " Question: what is the access code? Reply with the number only. "
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "examples" / "fake_suites" / "echo_mixed.jsonl"


def build_items(seed: int = SEED) -> list[dict[str, object]]:
    """Generate the fake suite items.

    Args:
        seed: Seed for the numpy generator.

    Returns:
        A list of JSON-compatible item dictionaries in ``id`` order.
    """
    rng = np.random.default_rng(seed)
    items: list[dict[str, object]] = []
    counter = 0
    for bucket, low, high in BUCKETS:
        for _ in range(ITEMS_PER_BUCKET):
            counter += 1
            answer = int(rng.integers(0, 1000))
            marker = f"<<ECHO:{answer}>>"
            suffix_len = len(QUESTION) + len(marker)
            target = int(rng.integers(low, high + 1))
            body = filler_text(rng, max(target - suffix_len, 1))[: max(target - suffix_len, 1)]
            content = f"{body}{QUESTION}{marker}"
            items.append(
                {
                    "id": f"echo-{counter:04d}",
                    "messages": [{"role": "user", "content": content}],
                    "gold": str(answer),
                    "answer_type": "number",
                    "tags": [bucket],
                }
            )
    return items


def write_items(items: list[dict[str, object]], output: Path) -> None:
    """Write items as JSONL with sorted keys, one object per line.

    Args:
        items: Items to write.
        output: Destination file. Parent directories are created.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in items]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Parse arguments, build the suite, and write it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    items = build_items()
    write_items(items, args.output)
    print(f"wrote {len(items)} items to {args.output}")


if __name__ == "__main__":
    main()
