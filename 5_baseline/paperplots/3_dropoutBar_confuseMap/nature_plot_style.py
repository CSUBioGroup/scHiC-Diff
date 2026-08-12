"""Shared Nature-style configuration for dropout and SZ/DO figures."""
from __future__ import annotations

import matplotlib as mpl
import numpy as np


MM = 1 / 25.4
GR_DOUBLE_COL_IN = 174 * MM

FS_TICK = 8
FS_ANNOT = 8
FS_LABEL = 9
FS_TITLE = 10
FS_PANEL = 12

C_INK = "#1A1A1A"
C_GRID = "#4D4D4D"

NATURE_COLORS = {
    "scHiCluster": "#8491B4",
    "HiCImpute": "#7E6148",
    "Higashi-nbr0": "#91D1C2",
    "Higashi-nbr5": "#00A087",
    "scVI-3D": "#4DBBD5",
    "Tensor-FLAMINGO": "#3C5488",
    "scHiC-Diff": "#E64B35",
}

HERO_METHOD = "scHiC-Diff"
HERO_EDGE = "#5A1522"


def apply_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": FS_TICK,
            "axes.titlesize": FS_TITLE,
            "axes.labelsize": FS_LABEL,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_TICK,
            "mathtext.default": "regular",
            "axes.linewidth": 0.5,
            "lines.linewidth": 0.8,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "text.color": C_INK,
            "axes.labelcolor": C_INK,
            "axes.edgecolor": C_GRID,
            "xtick.color": C_INK,
            "ytick.color": C_INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "standard",
            "savefig.pad_inches": 0.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def soft_grid(ax, yticks) -> None:
    """Draw restrained horizontal reference lines behind the data."""
    for ytick in yticks:
        ax.axhline(ytick, color="#DBDBDB", lw=0.5, zorder=0)


def grouped_bars(
    ax,
    data,
    methods,
    palette=None,
    errors=None,
    group_gap=0.35,
    hero=HERO_METHOD,
):
    """Draw grouped bars while skipping undefined values cleanly."""
    palette = palette or NATURE_COLORS
    values = np.asarray(data, dtype=float)
    error_values = None if errors is None else np.asarray(errors, dtype=float)
    n_groups, n_methods = values.shape
    bar_width = (1 - group_gap) / n_methods
    x = np.arange(n_groups)

    for method_idx, method in enumerate(methods):
        offset = (method_idx - (n_methods - 1) / 2) * bar_width
        finite = np.isfinite(values[:, method_idx])
        if not finite.any():
            continue
        is_hero = method == hero
        yerr = None if error_values is None else error_values[finite, method_idx]
        ax.bar(
            x[finite] + offset,
            values[finite, method_idx],
            bar_width,
            color=palette[method],
            edgecolor=HERO_EDGE if is_hero else "white",
            linewidth=0.5 if is_hero else 0.4,
            zorder=3,
            yerr=yerr,
            error_kw={
                "elinewidth": 0.7,
                "capsize": 0,
                "ecolor": "#3F3F3F",
                "zorder": 5,
            },
        )
    ax.spines["bottom"].set_zorder(6)
    ax.spines["left"].set_zorder(6)
    ax.set_xticks(x)
    return x, bar_width
