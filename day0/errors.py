"""Exception types raised by day0.

Each error that the orchestrator must treat differently has its own class, so that a failed
candidate can be recorded as ``ERROR`` while a failed reference aborts the whole run.
"""

from __future__ import annotations


class Day0Error(Exception):
    """Base class for every error that day0 raises on purpose."""


class ConfigError(Day0Error):
    """The configuration file is unreadable, invalid, or references missing files."""


class BackendStartError(Day0Error):
    """A serving backend failed to start, to become healthy, or to shut down cleanly."""


class ServerResponseError(Day0Error):
    """An inference server answered with an HTTP error or with an unexpected payload.

    Attributes:
        status: HTTP status code when the error came from an HTTP response, otherwise ``None``.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        """Create the error.

        Args:
            message: Human-readable description, including the server body when available.
            status: HTTP status code, if the error came from an HTTP response.
        """
        super().__init__(message)
        self.status = status


class TokenizerMismatchError(Day0Error):
    """The candidate server tokenizes the probe set differently from the reference server."""


class ReferenceMismatchError(Day0Error):
    """Stored reference artifacts were produced by a different configuration."""
