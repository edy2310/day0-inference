"""Deterministic toy language model behind the fake server (vectorized).

The model is a pure function of the token context, so the same prompt gives the same logits
whatever the batch composition, and quality metrics have known outcomes in tests. The design
follows §13.1 of the specification.

Deviations from the specification's constants and mechanics are recorded in ``docs/decisions.md``:

* The forced token's logit is the current maximum plus ``FORCE_MARGIN`` instead of ``+50`` on top
  of the base logit (D-08). The margin is relative, so an unlucky base draw cannot lose the argmax.
* ``BOS`` is masked during generation, like the unused ids (D-09).
* Base noise comes from a counter-based SplitMix64 generator instead of one ``default_rng`` per
  position (D-22). It is a pure function of the context, so it is the same whether a position is
  computed alone or in a batch. Positions with the same 3-token context share their noise, which
  keeps prompt-logprob scoring of long prompts fast.
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
_MASK64 = (1 << 64) - 1
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)
_TAIL_STEP = np.uint64(0xD6E8FEB86659FD93)
_VOCAB_STEP = np.uint64(0xA24BAED4963EE407)
_SECOND_STREAM = np.uint64(0x5851F42D4C957F2D)
_CACHE_LIMIT = 20_000
_gauss_cache: dict[tuple[int, int, tuple[int, ...]], np.ndarray] = {}


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


TOKEN_TEXT: tuple[str, ...] = tuple(decode([token]) for token in range(VOCAB_SIZE))


def token_text(token_id: int) -> str:
    """Return the display text of one token.

    Args:
        token_id: Token id.

    Returns:
        The decoded byte, or an empty string for ids outside ``0..255``.
    """
    return TOKEN_TEXT[token_id]


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


def _mix(values: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer applied elementwise to unsigned 64-bit values.

    Args:
        values: ``uint64`` array.

    Returns:
        Mixed ``uint64`` array of the same shape.
    """
    x = values + _GOLDEN
    x = (x ^ (x >> np.uint64(30))) * _MIX_A
    x = (x ^ (x >> np.uint64(27))) * _MIX_B
    return x ^ (x >> np.uint64(31))


def _uniform(bits: np.ndarray) -> np.ndarray:
    """Map 64-bit values to open-interval uniforms in (0, 1).

    Args:
        bits: ``uint64`` array.

    Returns:
        Float64 array with the same shape.
    """
    return ((bits >> np.uint64(11)).astype(np.float64) + 0.5) / float(2**53)


def _context_keys(tails: np.ndarray, seed: int, salt: int) -> np.ndarray:
    """One 64-bit key per 3-token context, from the seed and a salt.

    Args:
        tails: Array of shape ``(n, 3)`` with token ids (``-1`` for missing history).
        seed: Seed of the stream.
        salt: Distinguishes streams that share a seed.

    Returns:
        Array of shape ``(n,)`` with ``uint64`` keys.
    """
    start = np.uint64((seed ^ salt) & _MASK64)
    keys = _mix(np.full(tails.shape[0], start, dtype=np.uint64))
    for column in range(tails.shape[1]):
        step = tails[:, column].astype(np.uint64) + np.uint64(1)
        keys = _mix(keys + step * _TAIL_STEP)
    return keys


def _gaussian_rows(keys: np.ndarray) -> np.ndarray:
    """Standard-normal vectors over the vocabulary, one per key (Box-Muller on hashed uniforms).

    Args:
        keys: Array of shape ``(n,)`` with ``uint64`` keys.

    Returns:
        Array of shape ``(n, VOCAB_SIZE)``.
    """
    vocab = np.arange(VOCAB_SIZE, dtype=np.uint64)
    base = keys[:, None] ^ (vocab[None, :] * _VOCAB_STEP)
    u1 = _uniform(_mix(base))
    u2 = _uniform(_mix(base ^ _SECOND_STREAM))
    gaussian: np.ndarray = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return gaussian


def _cached_gaussians(tails: np.ndarray, seed: int, salt: int) -> np.ndarray:
    """Standard-normal rows for each unique context, memoized per process.

    Args:
        tails: Unique contexts, shape ``(n, 3)``.
        seed: Stream seed.
        salt: Stream salt.

    Returns:
        Array of shape ``(n, VOCAB_SIZE)`` aligned with ``tails``.
    """
    missing = [
        i
        for i, row in enumerate(map(tuple, tails.tolist()))
        if (seed, salt, row) not in _gauss_cache
    ]
    if missing:
        if len(_gauss_cache) + len(missing) > _CACHE_LIMIT:
            _gauss_cache.clear()
        fresh = _gaussian_rows(_context_keys(tails[missing], seed, salt))
        for offset, index in enumerate(missing):
            _gauss_cache[(seed, salt, tuple(int(t) for t in tails[index]))] = fresh[offset]
    return np.stack([_gauss_cache[(seed, salt, tuple(int(t) for t in row))] for row in tails])


def context_tails(seq: Sequence[int] | np.ndarray, positions: np.ndarray) -> np.ndarray:
    """The three tokens before each position, padded with ``-1`` at the start of the sequence.

    Args:
        seq: Token sequence.
        positions: Context lengths (position ``p`` is predicted from ``seq[:p]``).

    Returns:
        Array of shape ``(len(positions), 3)``.
    """
    array = seq if isinstance(seq, np.ndarray) else np.asarray(seq, dtype=np.int64)
    indices = positions[:, None] - _CONTEXT_WINDOW + np.arange(_CONTEXT_WINDOW)[None, :]
    valid = indices >= 0
    gathered = array[np.clip(indices, 0, max(array.shape[0] - 1, 0))]
    return np.where(valid, gathered, -1).astype(np.int64)


def logits_rows(
    seq: Sequence[int] | np.ndarray,
    positions: np.ndarray,
    plan: TurnPlan,
    params: ModelParams,
) -> np.ndarray:
    """Logits for the token that follows ``seq[:p]``, for every ``p`` in ``positions``.

    Args:
        seq: Token sequence. Only the prefixes ``seq[:p]`` are used.
        positions: Context lengths, int64.
        plan: Turn plan of the sequence.
        params: Model parameters.

    Returns:
        Float64 array of shape ``(len(positions), VOCAB_SIZE)``.
    """
    tails = context_tails(seq, positions)
    unique, inverse = np.unique(tails, axis=0, return_inverse=True)
    rows = _cached_gaussians(unique, params.model_seed, 0)[inverse.reshape(-1)] * BASE_LOGIT_SCALE
    rows[:, PRINTABLE_IDS] += PRINTABLE_BOOST
    rows[:, BOS_ID] = UNUSED_LOGIT
    rows[:, FIRST_UNUSED_ID:] = UNUSED_LOGIT

    if plan.assistant_end is not None:
        offsets = positions - plan.assistant_end
        in_turn = offsets >= 0
        ramp = EOS_RAMP_SLOPE * np.maximum(0, offsets - EOS_RAMP_START)
        rows[in_turn, EOS_ID] += ramp[in_turn]
        if plan.forced:
            forced_rows = np.nonzero((offsets >= 0) & (offsets < len(plan.forced)))[0]
            if forced_rows.size:
                maxima = rows[forced_rows].max(axis=1)
                tokens = np.asarray(plan.forced, dtype=np.int64)[offsets[forced_rows]]
                rows[forced_rows, tokens] = maxima + FORCE_MARGIN

    if params.noise > 0:
        scale = (
            1.0 + params.long_ctx_factor * np.maximum(0, positions - params.long_ctx_start) / 1000.0
        )
        noise_unique = _cached_gaussians(unique, params.noise_seed, _NOISE_SALT)
        noise = noise_unique[inverse.reshape(-1)]
        rows = rows + params.noise * scale[:, None] * noise
    return rows


_NOISE_SALT = 0x6E6F697365  # distinguishes the perturbation stream from the base stream


def next_token_logits(
    seq: Sequence[int] | np.ndarray, position: int, plan: TurnPlan, params: ModelParams
) -> np.ndarray:
    """Logits for the token that follows ``seq[:position]``.

    Args:
        seq: Token sequence. Only ``seq[:position]`` is used.
        position: Context length.
        plan: Turn plan of the sequence.
        params: Model parameters.

    Returns:
        A float64 array of length ``VOCAB_SIZE``.
    """
    row: np.ndarray = logits_rows(seq, np.array([position], dtype=np.int64), plan, params)[0]
    return row


def log_softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis.

    Args:
        logits: Array of shape ``(..., VOCAB_SIZE)``.

    Returns:
        Log-probabilities with the same shape.
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    result: np.ndarray = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return result


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Log-softmax of one logit vector.

    Args:
        logits: Vector of length ``VOCAB_SIZE``.

    Returns:
        Log-probabilities.
    """
    return log_softmax_rows(logits)


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


def top_k_rows(logprobs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` most likely tokens of each row, ordered by probability then by id.

    Args:
        logprobs: Array of shape ``(rows, VOCAB_SIZE)``.
        k: Number of tokens per row. Zero returns empty arrays.

    Returns:
        ``(ids, values)``, each of shape ``(rows, k)``.
    """
    rows = logprobs.shape[0]
    if k <= 0:
        return np.zeros((rows, 0), dtype=np.int64), np.zeros((rows, 0))
    part = np.argpartition(-logprobs, k - 1, axis=1)[:, :k]
    values = np.take_along_axis(logprobs, part, axis=1)
    order = np.lexsort((part, -values), axis=-1)
    ids = np.take_along_axis(part, order, axis=1).astype(np.int64)
    return ids, np.take_along_axis(values, order, axis=1)


def top_k_pairs(logprobs: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Top-k of one log-probability vector as ``(token_id, logprob)`` pairs.

    Args:
        logprobs: Vector of length ``VOCAB_SIZE``.
        k: Number of tokens.

    Returns:
        Pairs in descending order of probability, ties broken by lowest id.
    """
    ids, values = top_k_rows(logprobs.reshape(1, -1), k)
    return [(int(i), float(v)) for i, v in zip(ids[0], values[0], strict=True)]
