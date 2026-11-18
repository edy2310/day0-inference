"""Deterministic seed derivation.

Every random choice in day0 (request seeds, bootstrap resamples, workload construction, fake model
noise) is derived from ``run.seed`` through :func:`derive_seed`, so a run can be reproduced exactly.
"""

from __future__ import annotations

import hashlib

_MAX_SEED = 2**31 - 1
_UNIT_SEPARATOR = b"\x1f"


def derive_seed(base: int, *labels: str | int) -> int:
    """Derive a stable seed from a base seed and a sequence of labels.

    The base and every label are hashed with SHA-256, so the result does not depend on Python's
    hash randomization and is identical across processes and platforms. Labels are converted with
    ``str()``, so the integer ``1`` and the string ``"1"`` produce the same seed.

    Args:
        base: Base seed, normally ``run.seed``.
        *labels: Strings or integers that name the sub-stream, for example ``"bootstrap"``.

    Returns:
        An integer in ``[0, 2**31 - 1]``.
    """
    hasher = hashlib.sha256()
    for part in (str(int(base)), *(str(label) for label in labels)):
        hasher.update(part.encode("utf-8"))
        # The separator keeps ("a", "b") distinct from ("ab",).
        hasher.update(_UNIT_SEPARATOR)
    return int.from_bytes(hasher.digest()[:4], "big") & _MAX_SEED
