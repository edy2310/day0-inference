"""Command-line entry point for day0.

Subcommands are added milestone by milestone. Errors that day0 raises on purpose are reported on
stderr and turn into exit code 2, as the specification requires for configuration errors.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Annotated

import typer

from day0 import __version__
from day0.backends import get_backend
from day0.backends.base import ProcessBackend, backend_context_from
from day0.backends.vllm import read_help_text, unknown_flags
from day0.config import Day0Config, config_hash, load_config
from day0.errors import Day0Error

app = typer.Typer(
    name="day0",
    help="Find the fastest LLM serving configuration that is still quality-equivalent.",
    no_args_is_help=True,
    add_completion=False,
)

EXIT_CONFIG_ERROR = 2


def _version_callback(value: bool) -> None:
    """Print the package version and exit when ``--version`` is given.

    Args:
        value: Whether the ``--version`` flag was passed.

    Raises:
        typer.Exit: Always exits with code 0 when the flag is set.
    """
    if value:
        typer.echo(f"day0 {__version__}")
        raise typer.Exit(code=0)


@app.callback()
def _main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the day0 version and exit.",
    ),
) -> None:
    """Root command group. The flag itself is handled by ``_version_callback``.

    Args:
        version: Parsed value of ``--version``; the callback does the work.
    """
    del version


def _load_or_exit(path: Path) -> Day0Config:
    """Load a configuration, printing the error and exiting with code 2 on failure.

    Args:
        path: Path to the YAML configuration.

    Returns:
        The validated configuration.

    Raises:
        typer.Exit: With code 2 when the configuration is invalid.
    """
    try:
        return load_config(path)
    except Day0Error as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=EXIT_CONFIG_ERROR) from exc


@app.command()
def validate(
    config: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="YAML configuration.")
    ],
    check_flags: Annotated[
        bool,
        typer.Option(
            "--check-flags",
            help="Warn about flags that vllm serve --help does not list (vllm backend only).",
        ),
    ] = False,
) -> None:
    """Validate a configuration, print the resolved server commands, and optionally check flags.

    Flag warnings never fail the command, because flags differ between vLLM versions.

    Args:
        config: Path to the YAML configuration.
        check_flags: Whether to compare flags with ``vllm serve --help``.
    """
    cfg = _load_or_exit(config)
    typer.echo(f"OK: {config} (config hash {config_hash(cfg)[:12]})")
    typer.echo(f"run: {cfg.run.name}  model: {cfg.model.id}  backend: {cfg.backend.kind}")
    for suite in cfg.quality.suites:
        typer.echo(f"suite: {suite.path}")

    backend = get_backend(cfg.backend.kind, cfg.backend, backend_context_from(cfg))
    servers = [("reference", cfg.reference), *(("candidate", s) for s in cfg.candidates)]
    commands: list[tuple[str, list[str]]] = []
    for role, server in servers:
        state = "" if server.enabled else " (disabled)"
        typer.echo(f"{role}: {server.name}{state}")
        if isinstance(backend, ProcessBackend):
            command = backend.build_command(server)
            commands.append((server.name, command))
            typer.echo(f"  $ {shlex.join(command)}")

    if not check_flags:
        return
    if cfg.backend.kind != "vllm":
        typer.echo("note: --check-flags only applies to the vllm backend", err=True)
        return
    help_text = read_help_text(cfg.backend.executable)
    if help_text is None:
        typer.echo(
            f"warning: cannot run '{cfg.backend.executable} serve --help'; flags not checked",
            err=True,
        )
        return
    for name, command in commands:
        for flag in unknown_flags(command, help_text):
            typer.echo(f"warning: {name}: {flag} does not appear in vllm serve --help", err=True)


@app.command(
    "fake-server",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def fake_server(ctx: typer.Context) -> None:
    """Run the fake OpenAI-compatible server. Arguments match ``python -m day0.fake.server``.

    Args:
        ctx: Typer context. Its ``args`` are forwarded unchanged.
    """
    from day0.fake.server import main as fake_main

    fake_main(list(ctx.args))
