"""Environment capture: Python, platform, server version, GPUs, and a hashed host name.

Every value is captured on a best-effort basis. A missing tool or endpoint records ``None`` and
never fails a run. The raw host name is never stored, only a short hash of it.
"""

from __future__ import annotations

import hashlib
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from day0 import __version__
from day0.client import InferenceClient

_TIMEOUT_S = 30
_CUDA_RE = re.compile(r"CUDA Version:\s*([0-9.]+)")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str]) -> str | None:
    """Run a short command and return its standard output.

    Args:
        args: Argument list.

    Returns:
        Stripped standard output, or ``None`` when the command is missing or fails.
    """
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_sha(repo: Path = REPO_ROOT) -> str | None:
    """Return the commit of the day0 checkout, if there is one.

    Args:
        repo: Directory to query.

    Returns:
        The full commit hash, or ``None`` outside a git checkout.
    """
    if shutil.which("git") is None:
        return None
    return _run(["git", "-C", str(repo), "rev-parse", "HEAD"])


def gpu_info() -> list[dict[str, str]] | None:
    """Query each GPU with ``nvidia-smi``.

    Returns:
        One mapping per GPU with ``name``, ``memory_total``, and ``driver_version``, or ``None``
        when ``nvidia-smi`` is unavailable.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    output = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
    )
    if output is None:
        return None
    gpus: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            gpus.append({"name": parts[0], "memory_total": parts[1], "driver_version": parts[2]})
    return gpus or None


def cuda_version() -> str | None:
    """Read the CUDA version from the ``nvidia-smi`` header.

    Returns:
        The version string, or ``None``.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    output = _run(["nvidia-smi"])
    if output is None:
        return None
    match = _CUDA_RE.search(output)
    return match.group(1) if match else None


def host_hash() -> str:
    """Return a short SHA-256 of the host name, so machines can be told apart without naming them.

    Returns:
        The first twelve hexadecimal characters of the digest.
    """
    return hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()[:12]


def local_server_version(executable: str) -> str | None:
    """Run ``<executable> --version`` for a server that does not expose ``/version``.

    Args:
        executable: The server executable.

    Returns:
        The first line of the output, or ``None``.
    """
    output = _run([executable, "--version"])
    return output.splitlines()[0] if output else None


async def capture_env(
    client: InferenceClient, *, backend_kind: str, executable: str
) -> dict[str, Any]:
    """Capture the environment of a run.

    Args:
        client: Client connected to the server under test.
        backend_kind: ``vllm``, ``fake``, or ``external``.
        executable: vLLM executable, used when the server has no ``/version``.

    Returns:
        A JSON-compatible dictionary.
    """
    server_version = await client.version()
    if server_version is None and backend_kind == "vllm":
        server_version = local_server_version(executable)
    return {
        "day0_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname_hash": host_hash(),
        "server_version": server_version,
        "backend_kind": backend_kind,
        "gpus": gpu_info(),
        "cuda_version": cuda_version(),
    }
