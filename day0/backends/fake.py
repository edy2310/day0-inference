"""Backend that launches the built-in fake server as a subprocess.

The fake backend uses the same process, health, and stop logic as the vLLM backend. The shared
``common_args`` are passed before each server's own ``args``, as in the ``fake.yaml`` example.
"""

from __future__ import annotations

import sys

from day0.backends.base import ProcessBackend
from day0.config import BackendConfig, ServerConfig


def build_fake_command(cfg: BackendConfig, server: ServerConfig) -> list[str]:
    """Build the command that starts the fake server for one configuration.

    Args:
        cfg: Backend settings.
        server: The configuration to serve.

    Returns:
        The argument list, starting with the current Python interpreter.
    """
    return [
        sys.executable,
        "-m",
        "day0.fake.server",
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        *cfg.common_args,
        *server.args,
    ]


class FakeBackend(ProcessBackend):
    """Launches the fake OpenAI-compatible server.

    The fake server starts in well under a second, so it is polled every 0.25 s instead of every
    2 s (decision D-23). The vLLM backend keeps the specified 2 s.
    """

    poll_interval_s: float = 0.25

    def build_command(self, server: ServerConfig) -> list[str]:
        """Return the fake-server command for a server.

        Args:
            server: The configuration to serve.

        Returns:
            The argument list.
        """
        return build_fake_command(self.cfg, server)
