"""Backend tests: real subprocesses for the fake backend, and command construction for vLLM."""

from __future__ import annotations

import socket
from pathlib import Path

import httpx
import pytest

from day0.backends import ExternalBackend, FakeBackend, VLLMBackend, get_backend
from day0.backends.base import BackendContext
from day0.backends.process import port_is_free
from day0.backends.vllm import build_vllm_command, unknown_flags
from day0.config import BackendConfig, ServerConfig
from day0.errors import BackendStartError, ConfigError


def _free_port() -> int:
    """Return a TCP port that was free when queried.

    Returns:
        The port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _context() -> BackendContext:
    """Return the run-level values used by these tests.

    Returns:
        A backend context for the fake model.
    """
    return BackendContext(model_id="fake-model", max_model_len=4096, seed=1, max_logprobs=5)


def _backend_config(port: int, **overrides: object) -> BackendConfig:
    """Build a fake backend configuration on a given port.

    Args:
        port: Port to use.
        **overrides: Fields to replace.

    Returns:
        The backend configuration.
    """
    data: dict[str, object] = {
        "kind": "fake",
        "port": port,
        "startup_timeout_s": 30,
        "cooldown_s": 0,
    }
    data.update(overrides)
    return BackendConfig.model_validate(data)


async def test_fake_backend_starts_serves_and_stops(tmp_path: Path) -> None:
    """The fake backend launches the server, serves health, and stops its process group.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    port = _free_port()
    backend = FakeBackend(_backend_config(port), _context())
    log_path = tmp_path / "reference" / "server.log"
    handle = await backend.start(ServerConfig(name="ref"), log_path)
    try:
        assert handle.served_model == "fake-model"
        assert handle.pid is not None
        assert handle.base_url == f"http://127.0.0.1:{port}"
        assert handle.log_path == log_path
        async with httpx.AsyncClient(base_url=handle.base_url) as http:
            assert (await http.get("/health")).status_code == 200
    finally:
        await backend.stop(handle)
    assert port_is_free("127.0.0.1", port)
    assert log_path.exists()


async def test_common_args_reach_the_server(tmp_path: Path) -> None:
    """``common_args`` and per-server ``args`` both reach the server command line.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    port = _free_port()
    backend = FakeBackend(
        _backend_config(port, common_args=["--startup-delay-s", "0.5"]), _context()
    )
    handle = await backend.start(
        ServerConfig(name="cand", args=["--spec-accept-rate", "0.6"]), tmp_path / "server.log"
    )
    try:
        metrics = httpx.get(f"{handle.base_url}/metrics").text
        assert "spec_decode_num_drafts_total" in metrics
    finally:
        await backend.stop(handle)


async def test_startup_timeout_raises_and_kills_the_process(tmp_path: Path) -> None:
    """A server that is still starting after the timeout raises and is stopped.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    port = _free_port()
    backend = FakeBackend(
        _backend_config(port, startup_timeout_s=1, common_args=["--startup-delay-s", "60"]),
        _context(),
    )
    with pytest.raises(BackendStartError, match="not healthy after"):
        await backend.start(ServerConfig(name="slow"), tmp_path / "server.log")
    assert port_is_free("127.0.0.1", port)


async def test_crashing_server_reports_log_tail(tmp_path: Path) -> None:
    """A process that exits during startup raises with the end of its log.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    backend = FakeBackend(_backend_config(_free_port()), _context())
    with pytest.raises(BackendStartError) as info:
        await backend.start(
            ServerConfig(name="broken", args=["--no-such-flag"]), tmp_path / "server.log"
        )
    assert "exited during startup" in str(info.value)
    assert "unrecognized arguments" in str(info.value)


async def test_busy_port_is_reported_clearly(tmp_path: Path) -> None:
    """A port that another process listens on is refused before any launch.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    port = _free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupant:
        occupant.bind(("127.0.0.1", port))
        occupant.listen(1)
        backend = FakeBackend(_backend_config(port), _context())
        with pytest.raises(BackendStartError, match="already in use"):
            await backend.start(ServerConfig(name="ref"), tmp_path / "server.log")
    assert not (tmp_path / "server.log").exists()


async def test_missing_executable_is_reported(tmp_path: Path) -> None:
    """A vLLM executable that does not exist raises a clear error.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    cfg = BackendConfig.model_validate(
        {"kind": "vllm", "executable": "/nonexistent/day0-vllm", "port": _free_port()}
    )
    backend = VLLMBackend(cfg, _context())
    with pytest.raises(BackendStartError, match="cannot launch"):
        await backend.start(ServerConfig(name="ref"), tmp_path / "server.log")


async def test_external_backend_attaches_without_owning_the_server(tmp_path: Path) -> None:
    """The external backend checks health and reads the model, and stop leaves the server running.

    Args:
        tmp_path: Pytest's per-test temporary directory.
    """
    port = _free_port()
    owner = FakeBackend(_backend_config(port), _context())
    owned = await owner.start(ServerConfig(name="owned"), tmp_path / "owned.log")
    try:
        external_cfg = BackendConfig.model_validate(
            {"kind": "external", "base_url": owned.base_url, "startup_timeout_s": 10}
        )
        attached = ExternalBackend(external_cfg, _context())
        handle = await attached.start(ServerConfig(name="ext"), tmp_path / "unused.log")
        assert handle.served_model == "fake-model"
        assert handle.pid is None
        assert handle.log_path is None
        await attached.stop(handle)
        assert httpx.get(f"{owned.base_url}/health").status_code == 200
    finally:
        await owner.stop(owned)


def test_get_backend_rejects_kind_mismatch() -> None:
    """Asking for a backend kind that does not match the configuration is a ConfigError."""
    cfg = _backend_config(8100)
    with pytest.raises(ConfigError):
        get_backend("vllm", cfg, _context())
    assert isinstance(get_backend("fake", cfg, _context()), FakeBackend)


def test_vllm_command_has_required_flags_and_default_logprobs_mode() -> None:
    """The vLLM command includes the fixed flags and ``raw_logprobs`` unless the user sets it."""
    cfg = BackendConfig.model_validate({"kind": "vllm", "executable": "vllm", "port": 8100})
    command = build_vllm_command(
        cfg, _context(), ServerConfig(name="fp8", args=["--quantization", "fp8"])
    )
    assert command[:3] == ["vllm", "serve", "fake-model"]
    for flag, value in (
        ("--host", "127.0.0.1"),
        ("--port", "8100"),
        ("--seed", "1"),
        ("--max-model-len", "4096"),
        ("--max-logprobs", "5"),
    ):
        index = command.index(flag)
        assert command[index + 1] == value
    assert command[-2:] == ["--quantization", "fp8"]
    mode_index = command.index("--logprobs-mode")
    assert command[mode_index + 1] == "raw_logprobs"
    assert mode_index < len(command) - 2
    assert command.count("--logprobs-mode") == 1


def test_vllm_command_respects_user_logprobs_mode() -> None:
    """A user-supplied ``--logprobs-mode`` is not duplicated."""
    cfg = BackendConfig.model_validate(
        {"kind": "vllm", "common_args": ["--logprobs-mode=processed_logprobs"]}
    )
    command = build_vllm_command(cfg, _context(), ServerConfig(name="ref"))
    assert command.count("--logprobs-mode") == 0
    assert "--logprobs-mode=processed_logprobs" in command


def test_unknown_flags_are_reported_once_and_values_ignored() -> None:
    """Only ``--flag`` tokens are checked, and each unknown flag is listed once."""
    help_text = "usage: vllm serve\n  --host HOST\n  --port PORT\n  --seed SEED\n"
    args = [
        "vllm",
        "serve",
        "m",
        "--host",
        "x",
        "--bogus",
        "1",
        "--bogus",
        "--port=8000",
        "--speculative-config",
        '{"method": "ngram"}',
    ]
    assert unknown_flags(args, help_text) == ["--bogus", "--speculative-config"]
