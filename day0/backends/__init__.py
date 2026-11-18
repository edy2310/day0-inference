"""Serving backends: how day0 starts, health-checks, and stops inference servers."""

from __future__ import annotations

from typing import Literal

from day0.backends.base import Backend, BackendContext, ServerHandle
from day0.backends.external import ExternalBackend
from day0.backends.fake import FakeBackend
from day0.backends.vllm import VLLMBackend
from day0.config import BackendConfig
from day0.errors import ConfigError

__all__ = [
    "Backend",
    "BackendContext",
    "ExternalBackend",
    "FakeBackend",
    "ServerHandle",
    "VLLMBackend",
    "get_backend",
]


def get_backend(
    kind: Literal["vllm", "fake", "external"],
    cfg: BackendConfig,
    context: BackendContext,
) -> Backend:
    """Create the backend for a kind.

    Args:
        kind: Backend kind, normally ``cfg.kind``.
        cfg: Backend settings.
        context: Run-level values.

    Returns:
        A backend instance.

    Raises:
        ConfigError: If the kind is unknown or does not match ``cfg.kind``.
    """
    if kind != cfg.kind:
        raise ConfigError(f"backend kind {kind!r} does not match configuration {cfg.kind!r}")
    if kind == "vllm":
        return VLLMBackend(cfg, context)
    if kind == "fake":
        return FakeBackend(cfg, context)
    if kind == "external":
        return ExternalBackend(cfg, context)
    raise ConfigError(f"unknown backend kind {kind!r}")
