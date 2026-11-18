"""Backend that launches ``vllm serve`` as a subprocess.

day0 never imports vLLM. The executable is launched from the configured path, and the flags are
passed through as raw arguments. Flag semantics are never hard-coded.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

from day0.backends.base import BackendContext, ProcessBackend
from day0.config import BackendConfig, ServerConfig

LOGPROBS_MODE_FLAG = "--logprobs-mode"
DEFAULT_LOGPROBS_MODE = "raw_logprobs"
HELP_ARGS: tuple[tuple[str, ...], ...] = (("serve", "--help=all"), ("serve", "--help"))


def build_vllm_command(
    cfg: BackendConfig, context: BackendContext, server: ServerConfig
) -> list[str]:
    """Build the ``vllm serve`` argument list for one server.

    ``--logprobs-mode raw_logprobs`` is added unless the user already passes that flag. Raw
    logprobs are what the quality metrics assume, so the choice is always explicit.

    Args:
        cfg: Backend settings.
        context: Run-level values (model, length limit, seed, logprob limit).
        server: The configuration to serve.

    Returns:
        The full argument list, starting with the executable.
    """
    command = [
        cfg.executable,
        "serve",
        context.model_id,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--seed",
        str(context.seed),
        "--max-model-len",
        str(context.max_model_len),
        "--max-logprobs",
        str(context.max_logprobs),
    ]
    extra = [*cfg.common_args, *server.args]
    if not any(
        arg == LOGPROBS_MODE_FLAG or arg.startswith(f"{LOGPROBS_MODE_FLAG}=") for arg in extra
    ):
        command += [LOGPROBS_MODE_FLAG, DEFAULT_LOGPROBS_MODE]
    return [*command, *extra]


def unknown_flags(args: Sequence[str], help_text: str) -> list[str]:
    """Return the ``--flag`` names in ``args`` that do not appear in the help text.

    Only tokens that start with ``--`` are checked. Values are not checked.

    Args:
        args: Argument list to check.
        help_text: Output of ``vllm serve --help``.

    Returns:
        The unknown flag names, in order of first appearance and without duplicates.
    """
    unknown: list[str] = []
    for arg in args:
        if not arg.startswith("--"):
            continue
        flag = arg.split("=", 1)[0]
        if flag not in help_text and flag not in unknown:
            unknown.append(flag)
    return unknown


def read_help_text(executable: str) -> str | None:
    """Run ``vllm serve --help=all`` (falling back to ``--help``) and return its output.

    Args:
        executable: The vLLM executable.

    Returns:
        The combined output, or ``None`` when the executable cannot be run.
    """
    for args in HELP_ARGS:
        try:
            result = subprocess.run(
                [executable, *args], capture_output=True, text=True, timeout=120, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        output = result.stdout + result.stderr
        if result.returncode == 0 and output.strip():
            return output
    return None


class VLLMBackend(ProcessBackend):
    """Launches ``vllm serve`` on the configured host and port."""

    def build_command(self, server: ServerConfig) -> list[str]:
        """Return the ``vllm serve`` command for a server.

        Args:
            server: The configuration to serve.

        Returns:
            The argument list.
        """
        return build_vllm_command(self.cfg, self.context, server)
