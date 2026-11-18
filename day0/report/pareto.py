"""Quality-versus-throughput Pareto chart (§17.3)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

WIDTH_PX = 1600
HEIGHT_PX = 1000
DPI = 100
MARKERS: dict[str, tuple[str, str]] = {
    "PASS": ("o", "#2a9d8f"),
    "FAIL_QUALITY": ("X", "#e63946"),
    "FAIL_SLO": ("s", "#f4a261"),
    "INSUFFICIENT_DATA": ("^", "#8d99ae"),
    "ERROR": ("P", "#000000"),
}


def render_pareto(
    path: Path,
    *,
    points: Sequence[Mapping[str, Any]],
    front: Sequence[str],
    recommended: str | None,
    flip_rate_max: float,
    title: str,
) -> None:
    """Draw the Pareto chart and write it as a PNG.

    Args:
        path: Output PNG file.
        points: One mapping per configuration with keys ``name``, ``x``, ``flip_rate``,
            ``flip_lower``, ``flip_upper``, and ``status``. Values are fractions, and ``x`` is the
            throughput ratio.
        front: Names on the Pareto front, drawn as a step line.
        recommended: Name of the recommended configuration, marked with a star.
        flip_rate_max: Threshold drawn as a dashed line.
        title: Figure title.
    """
    figure, axis = plt.subplots(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    try:
        for point in points:
            status = str(point["status"])
            marker, colour = MARKERS.get(status, ("o", "#444444"))
            y = float(point["flip_rate"]) * 100.0
            lower = float(point["flip_lower"]) * 100.0
            upper = float(point["flip_upper"]) * 100.0
            axis.errorbar(
                float(point["x"]),
                y,
                yerr=[[y - lower], [upper - y]],
                fmt=marker,
                color=colour,
                ecolor=colour,
                markersize=10,
                capsize=4,
                label=status,
            )
            axis.annotate(
                str(point["name"]),
                (float(point["x"]), y),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=9,
            )
        front_points = sorted(
            (p for p in points if str(p["name"]) in front), key=lambda p: float(p["x"])
        )
        if front_points:
            axis.step(
                [float(p["x"]) for p in front_points],
                [float(p["flip_rate"]) * 100.0 for p in front_points],
                where="post",
                color="#264653",
                linewidth=1.5,
                label="Pareto front",
            )
        if recommended is not None:
            for point in points:
                if str(point["name"]) == recommended:
                    axis.scatter(
                        [float(point["x"])],
                        [float(point["flip_rate"]) * 100.0],
                        marker="*",
                        s=400,
                        color="#ffb703",
                        edgecolors="black",
                        zorder=5,
                        label="recommended",
                    )
        axis.axhline(
            flip_rate_max * 100.0,
            linestyle="--",
            color="#6c757d",
            label=f"flip_rate_max = {flip_rate_max * 100:.2f}%",
        )
        axis.set_xlabel("Target throughput vs reference (×)")
        axis.set_ylabel("Flip rate (%), with Wilson interval")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        handles, labels = axis.get_legend_handles_labels()
        unique: dict[str, Any] = dict(zip(labels, handles, strict=False))
        axis.legend(unique.values(), unique.keys(), loc="best")
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, format="png", dpi=DPI)
    finally:
        plt.close(figure)
