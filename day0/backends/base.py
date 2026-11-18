"""Backend interface and the shared logic for backends that launch a process."""

from __future__ import annotations

import asyncio
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from day0.backends.process import (
    port_is_free,
    read_served_model,
    start_process,
    terminate_process,
    wait_for_health,
    wait_gpu_memory_released,
)
from day0.config import BackendConfig, Day0Config, ServerConfig
from day0.errors import BackendStartError


@dataclass(frozen=True)
class ServerHandle:
    """A running (or external) server.

    Attributes:
        base_url: Server root, for example ``http://127.0.0.1:8100``.
        served_model: Model id reported by ``GET /v1/models``.
        pid: Process id of the launched server, or ``None`` for external servers.
        log_path: Log file of the launched server, or ``None``.
    """

    base_url: str
    served_model: str
    pid: int | None
    log_path: Path | None


@dataclass(frozen=True)
class BackendContext:
    """Run-level values that backends need to build their command lines.

    Attributes:
        model_id: Model id from the configuration. Used when the server does not report one.
        max_model_len: Maximum context length passed to the server.
        seed: Run seed passed to the server.
        max_logprobs: Upper bound for logprobs, set to ``quality.top_k_logprobs``.
    """

    model_id: str
    max_model_len: int
    seed: int
    max_logprobs: int


def backend_context_from(cfg: Day0Config) -> BackendContext:
    """Derive the run-level backend values from a configuration.

    Args:
        cfg: Validated configuration.

    Returns:
        The backend context for that configuration.
    """
    return BackendContext(
        model_id=cfg.model.id,
        max_model_len=cfg.model.max_model_len,
        seed=cfg.run.seed,
        max_logprobs=cfg.quality.top_k_logprobs,
    )


class Backend(ABC):
    """Starts, health-checks, and stops inference servers."""

    def __init__(self, cfg: BackendConfig, context: BackendContext) -> None:
        """Create a backend.

        Args:
            cfg: Backend settings from the configuration.
            context: Run-level values.
        """
        self.cfg = cfg
        self.context = context

    @abstractmethod
    async def start(self, server: ServerConfig, log_path: Path) -> ServerHandle:
        """Start one server and wait until it is healthy.

        Args:
            server: The configuration to serve.
            log_path: File that receives the server's output.

        Returns:
            A handle to the running server.
        """

    @abstractmethod
    async def stop(self, handle: ServerHandle) -> None:
        """Stop a server started by this backend.

        Args:
            handle: The handle returned by :meth:`start`.
        """


class ProcessBackend(Backend):
    """Common start and stop logic for backends that launch a process on a fixed port."""

    def __init__(self, cfg: BackendConfig, context: BackendContext) -> None:
        """Create the backend and an empty process table.

        Args:
            cfg: Backend settings.
            context: Run-level values.
        """
        super().__init__(cfg, context)
        self._processes: dict[int, subprocess.Popen[bytes]] = {}

    @abstractmethod
    def build_command(self, server: ServerConfig) -> list[str]:
        """Return the argument list that launches the server.

        Args:
            server: The configuration to serve.

        Returns:
            The argument list. It is never passed through a shell.
        """

    @property
    def base_url(self) -> str:
        """The URL of the fixed port this backend uses."""
        return f"http://{self.cfg.host}:{self.cfg.port}"

    async def start(self, server: ServerConfig, log_path: Path) -> ServerHandle:
        """Launch the server, wait for ``/health``, and read the served model.

        If the server does not become healthy, its process group is stopped before the error is
        raised, so no orphan keeps the GPU busy.

        Args:
            server: The configuration to serve.
            log_path: File that receives the server's output.

        Returns:
            A handle to the healthy server.

        Raises:
            BackendStartError: If the port is busy, the executable is missing, the process exits,
                or the health check times out.
        """
        if not port_is_free(self.cfg.host, self.cfg.port):
            raise BackendStartError(
                f"port {self.cfg.port} on {self.cfg.host} is already in use; stop the other "
                f"process or change backend.port"
            )
        env = {**os.environ, **self.cfg.env, **server.env}
        process = start_process(self.build_command(server), env, log_path)
        try:
            await wait_for_health(
                self.base_url,
                timeout_s=self.cfg.startup_timeout_s,
                process=process,
                log_path=log_path,
            )
            served = await read_served_model(self.base_url, fallback=self.context.model_id)
        except BackendStartError:
            await terminate_process(process)
            raise
        self._processes[process.pid] = process
        return ServerHandle(
            base_url=self.base_url, served_model=served, pid=process.pid, log_path=log_path
        )

    async def stop(self, handle: ServerHandle) -> None:
        """Stop the process group, wait for the cooldown, and wait for GPU memory to be released.

        Args:
            handle: Handle returned by :meth:`start`.
        """
        process = self._processes.pop(handle.pid, None) if handle.pid is not None else None
        if process is None:
            return
        await terminate_process(process)
        if self.cfg.cooldown_s > 0:
            await asyncio.sleep(self.cfg.cooldown_s)
        await wait_gpu_memory_released()
