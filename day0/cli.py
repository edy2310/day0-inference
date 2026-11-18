"""Command-line entry point for day0.

Subcommands are added milestone by milestone. Errors that day0 raises on purpose are reported on
stderr and turn into exit code 2, as the specification requires for configuration errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from day0 import __version__
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
) -> None:
    """Validate a configuration and print a summary of its servers.

    Args:
        config: Path to the YAML configuration.
    """
    cfg = _load_or_exit(config)
    typer.echo(f"OK: {config} (config hash {config_hash(cfg)[:12]})")
    typer.echo(f"run: {cfg.run.name}  model: {cfg.model.id}  backend: {cfg.backend.kind}")
    typer.echo(f"reference: {cfg.reference.name}")
    for server in cfg.candidates:
        state = "enabled" if server.enabled else "disabled"
        typer.echo(f"candidate: {server.name} ({state})")
    for suite in cfg.quality.suites:
        typer.echo(f"suite: {suite.path}")
