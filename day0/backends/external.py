"""Backend for a server that is already running (SGLang, a remote vLLM, or any other server).

day0 does not start or stop anything. It only checks health and reads the served model, so the
measurement covers whatever the external server is doing.
"""

from __future__ import annotations

from pathlib import Path

from day0.backends.base import Backend, ServerHandle
from day0.backends.process import read_served_model, wait_for_health
from day0.config import ServerConfig
from day0.errors import BackendStartError


class ExternalBackend(Backend):
    """Attaches to a server that is already running."""

    async def start(self, server: ServerConfig, log_path: Path) -> ServerHandle:
        """Check health of the external server and read its model id.

        Args:
            server: The configuration. Its ``base_url`` overrides ``backend.base_url``.
            log_path: Ignored, because day0 does not own the server's output.

        Returns:
            A handle with ``pid`` and ``log_path`` set to ``None``.

        Raises:
            BackendStartError: If no base URL is configured or the server is not healthy in time.
        """
        del log_path
        base_url = server.base_url or self.cfg.base_url
        if base_url is None:
            raise BackendStartError(f"server '{server.name}' has no base_url")
        await wait_for_health(base_url, timeout_s=self.cfg.startup_timeout_s)
        served = await read_served_model(base_url, fallback=self.context.model_id)
        return ServerHandle(base_url=base_url, served_model=served, pid=None, log_path=None)

    async def stop(self, handle: ServerHandle) -> None:
        """Do nothing, because day0 does not own the external server.

        Args:
            handle: Handle returned by :meth:`start`.
        """
        del handle
