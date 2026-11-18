"""Process helpers for launching, health-checking, and stopping inference servers.

Every launched server runs in its own session, so SIGTERM and SIGKILL reach the whole process
group, including worker processes spawned by vLLM.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import signal
import socket
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx

from day0.errors import BackendStartError

logger = logging.getLogger(__name__)

LOG_TAIL_LINES = 50
HEALTH_POLL_S = 2.0
STOP_GRACE_S = 60.0
GPU_FREE_MIB = 2048
GPU_WAIT_S = 60.0
GPU_POLL_S = 2.0


def port_is_free(host: str, port: int) -> bool:
    """Check whether a TCP port can be bound on a host.

    ``SO_REUSEADDR`` is set so that sockets left in ``TIME_WAIT`` do not count as busy, while a
    port with an active listener still does.

    Args:
        host: Interface to bind.
        port: Port number.

    Returns:
        True when the port can be bound.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def tail_log(path: Path | None, lines: int = LOG_TAIL_LINES) -> str:
    """Return the last lines of a log file for error messages.

    Args:
        path: Log file, or ``None``.
        lines: Number of lines to return.

    Returns:
        The tail of the log, or a short note when there is no log.
    """
    if path is None or not path.exists():
        return "(no log file)"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:]) or "(log file is empty)"


def start_process(
    command: Sequence[str], env: Mapping[str, str], log_path: Path
) -> subprocess.Popen[bytes]:
    """Start a server process with stdout and stderr going to a log file.

    Args:
        command: Argument list. It is never passed through a shell.
        env: Complete environment for the child process.
        log_path: File that receives stdout and stderr. It is truncated.

    Returns:
        The running process.

    Raises:
        BackendStartError: If the executable cannot be launched.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        try:
            return subprocess.Popen(
                list(command),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(env),
                start_new_session=True,
            )
        except OSError as exc:
            raise BackendStartError(f"cannot launch {command[0]!r}: {exc}") from exc


async def wait_for_health(
    base_url: str,
    *,
    timeout_s: float,
    process: subprocess.Popen[bytes] | None = None,
    log_path: Path | None = None,
    poll_interval_s: float = HEALTH_POLL_S,
) -> None:
    """Poll ``GET /health`` until it returns 200, the process dies, or time runs out.

    Args:
        base_url: Server root.
        timeout_s: Seconds to wait in total.
        process: Launched process to watch, or ``None`` for an external server.
        log_path: Log file whose tail is included in errors.
        poll_interval_s: Seconds between polls.

    Raises:
        BackendStartError: If the process exits, or the server is not healthy in time.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
        while True:
            if process is not None and process.poll() is not None:
                raise BackendStartError(
                    f"server process exited during startup with code {process.returncode}. "
                    f"Last {LOG_TAIL_LINES} log lines:\n{tail_log(log_path)}"
                )
            try:
                response = await client.get("/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if loop.time() >= deadline:
                raise BackendStartError(
                    f"server at {base_url} was not healthy after {timeout_s:g}s. "
                    f"Last {LOG_TAIL_LINES} log lines:\n{tail_log(log_path)}"
                )
            await asyncio.sleep(poll_interval_s)


async def read_served_model(base_url: str, *, fallback: str) -> str:
    """Read the served model id from ``GET /v1/models``.

    Args:
        base_url: Server root.
        fallback: Value returned when the endpoint has no models.

    Returns:
        ``data[0].id`` from the response, or ``fallback``.

    Raises:
        BackendStartError: If the endpoint cannot be read.
    """
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
            response = await client.get("/v1/models")
    except httpx.HTTPError as exc:
        raise BackendStartError(f"cannot read /v1/models from {base_url}: {exc}") from exc
    if response.status_code != 200:
        raise BackendStartError(f"/v1/models returned HTTP {response.status_code}")
    data: object = response.json().get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        model_id = data[0].get("id")
        if isinstance(model_id, str) and model_id:
            return model_id
    return fallback


def _signal_group(pid: int, sig: int) -> None:
    """Send a signal to a process group, ignoring groups that are already gone.

    Args:
        pid: Process id, which is also the group id for sessions started by day0.
        sig: Signal number.
    """
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, sig)


async def terminate_process(
    process: subprocess.Popen[bytes], *, grace_s: float = STOP_GRACE_S
) -> None:
    """Stop a process group with SIGTERM, then SIGKILL after the grace period.

    Args:
        process: Process started with :func:`start_process`.
        grace_s: Seconds to wait after SIGTERM before sending SIGKILL.
    """
    if process.poll() is None:
        _signal_group(process.pid, signal.SIGTERM)
        try:
            await asyncio.to_thread(process.wait, grace_s)
        except subprocess.TimeoutExpired:
            logger.warning("server %s ignored SIGTERM; sending SIGKILL", process.pid)
            _signal_group(process.pid, signal.SIGKILL)
            await asyncio.to_thread(process.wait)
    else:
        process.wait()


def _gpu_memory_used_mib(executable: str) -> list[int] | None:
    """Read the used memory of each GPU with ``nvidia-smi``.

    Args:
        executable: Path to ``nvidia-smi``.

    Returns:
        Used memory in MiB for each GPU, or ``None`` when the query fails.
    """
    try:
        result = subprocess.run(
            [executable, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    values: list[int] = []
    for line in result.stdout.splitlines():
        if line.strip():
            values.append(int(float(line.strip())))
    return values or None


async def wait_gpu_memory_released(
    *, max_wait_s: float = GPU_WAIT_S, threshold_mib: int = GPU_FREE_MIB
) -> bool:
    """Wait until every GPU uses less than ``threshold_mib``, if ``nvidia-smi`` exists.

    Args:
        max_wait_s: Seconds to wait before giving up.
        threshold_mib: Memory threshold in MiB.

    Returns:
        True when memory was released, or when there is nothing to check. False on timeout.
    """
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return True
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max_wait_s
    while True:
        used = await asyncio.to_thread(_gpu_memory_used_mib, executable)
        if used is None or max(used) < threshold_mib:
            return True
        if loop.time() >= deadline:
            logger.warning("GPU memory still in use after %gs: %s MiB", max_wait_s, used)
            return False
        await asyncio.sleep(GPU_POLL_S)
