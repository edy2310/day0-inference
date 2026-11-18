"""Command-line entry point for day0.

Subcommands are added milestone by milestone; M0 only exposes ``--version``.
"""

from __future__ import annotations

import typer

from day0 import __version__

app = typer.Typer(
    name="day0",
    help="Find the fastest LLM serving configuration that is still quality-equivalent.",
    no_args_is_help=True,
    add_completion=False,
)


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
