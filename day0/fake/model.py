"""Deterministic toy language model behind the fake server.

The model is a pure function of the token context, so the same prompt gives the same logits
whatever the batch composition, and quality metrics have known outcomes in tests. The design
follows §13.1 of the specification.

Two departures from the specification's constants are recorded in ``docs/decisions.md``:

* The forced token's logit is the current maximum plus ``FORCE_MARGIN`` instead of ``+50``
  on top of the base logit. With ``+50`` the forced distributions are numerically degenerate,
  so no candidate can ever have a non-zero KL. A fixed bias can also lose the argmax to an
  unlucky base draw, so the margin is relative to the maximum.
* ``BOS`` is masked during generation, like the unused ids, so the fake model never emits it.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from day0.seeds import derive_seed

VOCAB_SIZE = 512
BOS_ID = 256
EOS_ID = 257
FIRST_UNUSED_ID = 258
PRINTABLE_IDS = slice(32, 127)  # ASCII 32..126 inclusive
PRINTABLE_BOOST = 3.0
BASE_LOGIT_SCALE = 2.0
UNUSED_LOGIT = -30.0
# The forced token's logit is set to the current maximum plus this margin. A fixed bias could lose
# the argmax against an unlucky base draw; a margin relative to the maximum cannot.
FORCE_MARGIN = 2.0
EOS_RAMP_START = 40
EOS_RAMP_SLOPE = 0.02
ASSISTANT_MARKER = b"<|assistant|>\n"
ECHO_RE = re.compile(rb"<<ECHO:(-?\d+)>>")
_TWO_POW_31 = float(2**31)
_CONTEXT_WINDOW = 3


@dataclass(frozen=True)
class ModelParams:
    """Parameters that define one fake model and, optionally, a perturbed candidate.

    Attributes:
        model_seed: Seed for the base logits.
        noise: Scale of the candidate perturbation. ``0`` disables it.
        noise_seed: Seed for the perturbation.
        flip_prob: Probability, decided once per prompt, that the echoed answer is off by one.
        flip_seed: Seed for the per-prompt flip decision.
        long_ctx_start: Absolute position after which the perturbation grows.
        long_ctx_factor: Growth of the perturbation per 1000 positions past ``long_ctx_start``.
    """

    model_seed: int = 7
    noise: float = 0.0
    noise_seed: int = 11
    flip_prob: float = 0.0
    flip_seed: int = 13
    long_ctx_start: int = 1_000_000
    long_ctx_factor: float = 0.0


@dataclass(frozen=True)
class TurnPlan:
    """What the model knows about the assistant turn of a sequence.

    Attributes:
        assistant_end: Index just after the first ``<|assistant|>`` marker, or ``None``.
        forced: Tokens the echo rule forces, ending with EOS. Empty when there is no echo marker.
    """

    assistant_end: int | None
    forced: tuple[int, ...]


def encode_text(text: str, *, add_special_tokens: bool) -> list[int]:
    """Tokenize text into bytes, optionally preceded by BOS.

    Args:
        text: Text to encode.
        add_special_tokens: Whether to prepend BOS.

    Returns:
        Token ids in ``0..255``, with BOS in front when requested.
    """
    ids = list(text.encode("utf-8"))
    return [BOS_ID, *ids] if add_special_tokens else ids


def render_chat(messages: Sequence[tuple[str, str]], *, add_generation_prompt: bool) -> str:
    """Render chat messages with the fake chat template.

    Args:
        messages: ``(role, content)`` pairs in order.
        add_generation_prompt: Whether to append the assistant header.

    Returns:
        The rendered prompt text.
    """
    text = "".join(f"<|{role}|>\n{content}\n" for role, content in messages)
    return text + ("<|assistant|>\n" if add_generation_prompt else "")


def decode(token_ids: Sequence[int]) -> str:
    """Decode byte tokens to text. Ids outside ``0..255`` are skipped.

    Args:
        token_ids: Token ids to decode.

    Returns:
        UTF-8 text with replacement characters for partial sequences.
    """
    return bytes(token for token in token_ids if 0 <= token < 256).decode("utf-8", errors="replace")


def plan_turn(seq: Sequence[int], params: ModelParams) -> TurnPlan:
    """Find the assistant turn of a sequence and the answer it forces.

    The first marker in the sequence is used. Prompts and prompt-plus-output sequences therefore
    yield the same plan, which keeps generation and teacher-forced scoring consistent.

    Args:
        seq: Full token sequence, prompt first.
        params: Model parameters; ``flip_prob`` and ``flip_seed`` decide the flip.

    Returns:
        The turn plan. Without a marker, ``assistant_end`` is ``None``.
    """
    marker = list(ASSISTANT_MARKER)
    width = len(marker)
    end: int | None = None
    for start in range(len(seq) - width + 1):
        if seq[start] == marker[0] and list(seq[start : start + width]) == marker:
            end = start + width
            break
    if end is None:
        return TurnPlan(assistant_end=None, forced=())

    prompt_bytes = bytes(token for token in seq[:end] if 0 <= token < 256)
    match = ECHO_RE.search(prompt_bytes)
    if match is None:
        return TurnPlan(assistant_end=end, forced=())
    answer = int(match.group(1))
    digest = hashlib.sha256(prompt_bytes).hexdigest()
    if derive_seed(params.flip_seed, digest) / _TWO_POW_31 < params.flip_prob:
        answer += 1
    forced = [*f"The answer is {answer}.".encode(), EOS_ID]
    return TurnPlan(assistant_end=end, forced=tuple(forced))


def next_token_logits(
    seq: Sequence[int], position: int, plan: TurnPlan, params: ModelParams
) -> np.ndarray:
    """Return the logits for the token that follows ``seq[:position]``.

    Args:
        seq: Token sequence. Only ``seq[:position]`` is used.
        position: Context length.
        plan: Turn plan of the sequence.
        params: Model parameters.

    Returns:
        A float64 array of length ``VOCAB_SIZE``.
    """
    tail = tuple(int(token) for token in seq[max(0, position - _CONTEXT_WINDOW) : position])
    rng = np.random.default_rng(derive_seed(params.model_seed, *tail))
    logits = rng.normal(0.0, BASE_LOGIT_SCALE, VOCAB_SIZE)
    logits[PRINTABLE_IDS] += PRINTABLE_BOOST

    if plan.assistant_end is not None and position >= plan.assistant_end:
        offset = position - plan.assistant_end
        logits[EOS_ID] += EOS_RAMP_SLOPE * max(0, offset - EOS_RAMP_START)
        if offset < len(plan.forced):
            logits[plan.forced[offset]] = float(np.max(logits)) + FORCE_MARGIN

    logits[BOS_ID] = UNUSED_LOGIT
    logits[FIRST_UNUSED_ID:] = UNUSED_LOGIT

    if params.noise > 0:
        scale = 1.0 + params.long_ctx_factor * max(0, position - params.long_ctx_start) / 1000.0
        noise_rng = np.random.default_rng(derive_seed(params.noise_seed, *tail, "noise"))
        logits = logits + params.noise * scale * noise_rng.normal(0.0, 1.0, VOCAB_SIZE)
    return logits


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax.

    Args:
        logits: Unnormalized scores.

    Returns:
        Log-probabilities that sum to one in probability space.
    """
    shifted = logits - np.max(logits)
    result: np.ndarray = shifted - np.log(np.sum(np.exp(shifted)))
    return result


def greedy_token(logits: np.ndarray, *, ignore_eos: bool) -> int:
    """Pick the argmax token. Ties go to the lowest id.

    Args:
        logits: Unnormalized scores.
        ignore_eos: If true, EOS cannot be picked.

    Returns:
        The chosen token id.
    """
    scores = logits.copy()
    if ignore_eos:
        scores[EOS_ID] = -np.inf
    return int(np.argmax(scores))


def top_k_pairs(logprobs: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return the ``k`` most likely tokens, ties broken by lowest id.

    Args:
        logprobs: Log-probabilities over the vocabulary.
        k: Number of tokens to return.

    Returns:
        ``(token_id, logprob)`` pairs in descending order of probability.
    """
    if k <= 0:
        return []
    order = np.lexsort((np.arange(logprobs.shape[0]), -logprobs))[:k]
    return [(int(index), float(logprobs[index])) for index in order]


def token_text(token_id: int) -> str:
    """Return the display text of one token.

    Args:
        token_id: Token id.

    Returns:
        The decoded byte, or an empty string for ids outside ``0..255``.
    """
    return decode([token_id])
