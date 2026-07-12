"""Shared plotting helpers — one consistent visual style across all stages.

Import `apply_style()` once at the top of a pipeline. Use `save_fig()` so every
figure lands in outputs/figures with a uniform DPI and tight bounding box.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt

from . import config as C

# Color for each class label, used consistently everywhere.
CLASS_COLORS = {-1: "#c0392b", 0: "#7f8c8d", 1: "#27ae60"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    })


def save_fig(fig, name: str, subdir: Path | None = None) -> Path:
    """Save a figure to outputs/figures/<name>.png and close it."""
    out_dir = subdir if subdir is not None else C.OUT_FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
