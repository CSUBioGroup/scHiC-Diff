"""Shared Genome Research style for imputation-runtime figures.
Encodes the acceptance-stage Digital Art specs from
https://genome.cshlp.org/site/misc/ifora_digartsubm.xhtml

HARD RULES baked in here
------------------------
* In-figure fonts: Helvetica / Arial, 8-10 pt.  Panel letters 12 pt bold UPPERCASE.
  Font-size variation *within one figure* must stay <= 2 pt.
* All strokes >= 0.25 pt.
* Colour mode RGB; avoid pale tints; hues far apart AND colour-blind safe.
* Descriptive panel titles go in the CAPTION, not on the figure; the key lives
  inside the figure body.
* One figure per file; all panels of a figure in the SAME file.
* Submit charts as VECTOR (PDF/EPS); if rasterised, combination art 600-900 dpi.

Arial substitute: Liberation Sans (metric-compatible with Arial, embeds cleanly).

The plotting entry point is ``plot_imputation_runtime.py``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import LogLocator, NullFormatter

# ----------------------------------------------------------------------------- units / widths
MM = 1 / 25.4
GR_WIDTH_1COL = 85 * MM
GR_WIDTH_1P5COL = 114 * MM
GR_WIDTH_2COL = 174 * MM

# ----------------------------------------------------------------------------- font sizes (8-10 pt band)
FS_TICK = 8
FS_AXIS = 9
FS_LEGEND = 8
FS_ANNOT = 8
FS_PANEL = 12

# ----------------------------------------------------------------------------- colours
# Hero (the author's method) pops in warm crimson; every baseline recedes into a
# single muted slate so the eye is not asked to decode a rainbow — position and
# length already carry the information, colour only flags the hero.
HERO_FILL = "#D1495B"
HERO_EDGE = "#8A2E3B"
BASE_FILL = "#93A6B8"
BASE_EDGE = "#4E6072"
DOT_COLOR = "#22303C"   # per-run dots (dark, semi-transparent)
GRID_COLOR = "#C9CCD1"   # >= 0.25 pt gridlines
LAND_COLOR = "#9AA0A6"   # time-landmark rules
HW_GPU = "#2C5F8A"   # hardware glyph: GPU
HW_CPU = "#B07A2E"   # hardware glyph: CPU

# Optional per-method palette (used only by the scaling line plot, where colour
# must separate methods). Okabe-Ito-based, colour-blind safe, hero overridden.
GR_COLORS = {
    "scHiC-Diff":       HERO_FILL,
    "scVI-3D":          "#0072B2",
    "HiCImpute":        "#009E73",
    "scHiCluster":      "#E69F00",
    "Higashi_nbr0":     "#56B4E9",
    "Higashi_nbr5":     "#6A4C93",
    "Tensor-FLAMINGO":  "#7F7F7F",
}

HERO_METHOD = "scHiC-Diff"
METHOD_DISPLAY_NAMES = {
    "Higashi_nbr0": "Higashi-nbr0",
    "Higashi_nbr5": "Higashi-nbr5",
}


def display_method_name(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method, method)


# ----------------------------------------------------------------------------- global style
def apply_gr_style() -> None:
    mpl.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size":        FS_TICK,
        "axes.titlesize":   FS_AXIS,
        "axes.labelsize":   FS_AXIS,
        "xtick.labelsize":  FS_TICK,
        "ytick.labelsize":  FS_TICK,
        "legend.fontsize":  FS_LEGEND,
        "axes.linewidth":   0.6,       # >= 0.25 pt
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size":  2.6,
        "ytick.major.size":  2.6,
        "xtick.direction":  "out",
        "ytick.direction":  "out",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":   "#2b2b2b",
        "pdf.fonttype":     42,        # embed TrueType so text stays editable/searchable
        "ps.fonttype":      42,
        "svg.fonttype":     "none",
        "figure.dpi":       150,
        "savefig.dpi":      600,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.02,
    })


# ----------------------------------------------------------------------------- helpers: time
def fmt_duration(sec: float) -> str:
    """Compact human-readable duration, e.g. '43 s', '2.4 min', '4.4 h', '1.9 d'."""
    if sec < 90:
        return f"{sec:.0f} s"
    if sec < 90 * 60:
        return f"{sec/60:.1f} min"
    if sec < 36 * 3600:
        return f"{sec/3600:.1f} h"
    return f"{sec/86400:.1f} d"


# landmark seconds and their short labels
_LANDMARKS = [
    (1,        "1 s"),
    (10,       "10 s"),
    (60,       "1 min"),
    (600,      "10 min"),
    (3600,     "1 h"),
    (21600,    "6 h"),
    (86400,    "1 d"),
    (172800,   "2 d"),
]


def time_landmarks(ax, lo: float, hi: float) -> None:
    """Faint vertical rules + top-axis labels at 1 s / 1 min / 1 h / 1 day ..."""
    top = ax.secondary_xaxis("top")
    ticks, labels = [], []
    for sec, lab in _LANDMARKS:
        if lo <= sec <= hi:
            ax.axvline(sec, color=LAND_COLOR, lw=0.4, ls=(0, (3, 3)), zorder=0)
            ticks.append(sec)
            labels.append(lab)
    top.set_xscale("log")
    top.set_xlim(ax.get_xlim())
    top.set_xticks(ticks)
    top.set_xticklabels(labels, fontsize=FS_TICK - 0.5)
    top.tick_params(length=0, pad=1.5)
    for s in ("left", "right", "top"):
        top.spines[s].set_visible(False)


# ----------------------------------------------------------------------------- helpers: panels
def add_panel_label(ax, letter: str, dx: float = -0.14, dy: float = 1.04) -> None:
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="right")


# ----------------------------------------------------------------------------- core: shared axis
def _axis_bounds(ax, runs, medians, floor):
    n = len(medians)
    y = np.arange(n)[::-1]
    allv = np.concatenate([np.asarray(r, float) for r in runs] + [np.asarray(medians, float)])
    vmin, vmax = allv.min(), allv.max()
    if floor is None:
        floor = 10 ** np.floor(np.log10(vmin))
    right = 10 ** (np.log10(vmax) + 0.55)
    ax.set_xscale("log")
    ax.set_xlim(floor, right)
    ax.set_ylim(-0.7, n - 0.3)
    return y, floor, right


def _decorate(
    ax,
    names,
    y,
    hardware,
    hms,
    hero,
    floor,
    right,
    readout_ax=None,
):
    # y labels (hero bold)
    ax.set_yticks(y)
    ax.set_yticklabels([display_method_name(name) for name in names], fontsize=FS_TICK)
    for tick, name in zip(ax.get_yticklabels(), names):
        if name == hero:
            tick.set_fontweight("bold")
            tick.set_color(HERO_EDGE)

    target_ax = readout_ax or ax
    if readout_ax is not None:
        readout_ax.set_xlim(0, 1)
        readout_ax.set_ylim(ax.get_ylim())
        readout_ax.axis("off")
        readout_ax.axvline(0.0, color=GRID_COLOR, lw=0.5, clip_on=False)
        marker_x, text_x = 0.07, 0.15
        transform = blended_transform_factory(readout_ax.transAxes, readout_ax.transData)
    else:
        marker_x, text_x = 1.005, 1.015
        transform = blended_transform_factory(ax.transAxes, ax.transData)

    for i, yi in enumerate(y):
        glyph = HW_GPU if _is_gpu(hardware[i]) else HW_CPU
        target_ax.text(
            text_x,
            yi,
            f"{hms[i]}  \u00b7  {hardware[i]}",
            transform=transform,
            fontsize=FS_ANNOT,
            va="center",
            ha="left",
            fontweight="normal",
            color="#222222",
        )
        target_ax.scatter(
            [marker_x],
            [yi],
            transform=transform,
            s=16,
            marker="s",
            color=glyph,
            clip_on=False,
            zorder=6,
        )
    ax.set_xlabel("End-to-end runtime  (seconds, log scale)", fontsize=FS_AXIS)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10, subs=tuple(np.arange(2, 10) * 0.1), numticks=15))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=1.4, color="#9aa0a6")
    time_landmarks(ax, floor, right)


# ----------------------------------------------------------------------------- core: runtime bars
def log_runtime_bars(
    ax,
    names,
    medians,
    runs,
    hardware,
    hms,
    hero=HERO_METHOD,
    floor=None,
    readout_ax=None,
):
    y, floor, right = _axis_bounds(ax, runs, medians, floor)
    for yi, name, med in zip(y, names, medians):
        is_hero = (name == hero)
        ax.barh(yi, med - floor, left=floor, height=0.62,
                color=HERO_FILL if is_hero else BASE_FILL,
                edgecolor=HERO_EDGE if is_hero else BASE_EDGE,
                linewidth=0.7 if is_hero else 0.5, zorder=2)
    rng = np.random.default_rng(0)
    for yi, rr in zip(y, runs):
        rr = np.asarray(rr, float)
        jit = (rng.random(len(rr)) - 0.5) * 0.34
        ax.scatter(rr, np.full(len(rr), yi) + jit, s=7, color=DOT_COLOR,
                   alpha=0.55, linewidths=0, zorder=4)
    _decorate(
        ax, names, y, hardware, hms, hero, floor, right, readout_ax
    )
    return y


# ----------------------------------------------------------------------------- core: lollipop
def log_runtime_lollipop(
    ax,
    names,
    medians,
    runs,
    hardware,
    hms,
    hero=HERO_METHOD,
    floor=None,
    show_dots=True,
    show_range=True,
    readout_ax=None,
):
    """Log-honest alternative to bars: light stem + median head + min-max whisker."""
    y, floor, right = _axis_bounds(ax, runs, medians, floor)
    rng = np.random.default_rng(0)
    for i, (yi, name, med, rr) in enumerate(zip(y, names, medians, runs)):
        is_hero = (name == hero)
        head = HERO_FILL if is_hero else BASE_FILL
        edge = HERO_EDGE if is_hero else BASE_EDGE
        rr = np.asarray(rr, float)
        # light stem from floor to median (guides the eye; no misleading area)
        ax.plot([floor, med], [yi, yi], color=edge, lw=0.7 if is_hero else 0.5,
                alpha=0.55, zorder=2, solid_capstyle="round")
        # min-max whisker
        if show_range and rr.size > 1 and rr.min() < rr.max():
            ax.plot([rr.min(), rr.max()], [yi, yi], color=edge, lw=1.4,
                    alpha=0.9, zorder=3, solid_capstyle="round")
            for xv in (rr.min(), rr.max()):
                ax.plot([xv, xv], [yi - 0.14, yi + 0.14], color=edge, lw=1.0, zorder=3)
        # optional faint run dots
        if show_dots and rr.size > 1:
            jit = (rng.random(rr.size) - 0.5) * 0.26
            ax.scatter(rr, np.full(rr.size, yi) + jit, s=5, color=DOT_COLOR,
                       alpha=0.4, linewidths=0, zorder=4)
        # median head
        ax.scatter([med], [yi], s=54 if is_hero else 40, facecolor=head,
                   edgecolor=edge, linewidths=0.9 if is_hero else 0.6, zorder=6)
    _decorate(
        ax, names, y, hardware, hms, hero, floor, right, readout_ax
    )
    return y


def log_runtime_dots(
    ax,
    names,
    values,
    hardware,
    hms,
    hero=HERO_METHOD,
    floor=None,
    xlabel=None,
    readout_ax=None,
):
    """Cleveland dot plot: one value per method. No filled stem (nothing to imply
    area on a log axis) — just a dotted leader line and a dot at the true value.
    """
    runs = [[v] for v in values]
    y, floor, right = _axis_bounds(ax, runs, values, floor)

    for yi, name, v in zip(y, names, values):
        is_hero = (name == hero)
        edge = HERO_EDGE if is_hero else BASE_EDGE
        ax.plot([floor, v], [yi, yi], color=GRID_COLOR, lw=0.5,
                ls=(0, (1.6, 1.8)), zorder=1)          # leader line, not a bar
        ax.scatter([v], [yi], s=56 if is_hero else 40,
                   facecolor=HERO_FILL if is_hero else BASE_FILL,
                   edgecolor=edge, linewidths=0.9 if is_hero else 0.6, zorder=6)

    _decorate(
        ax, names, y, hardware, hms, hero, floor, right, readout_ax
    )
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_AXIS)
    return y


def _is_gpu(tag: str) -> bool:
    t = tag.lower()
    return ("v100" in t) or ("gpu" in t) or ("a100" in t)


# ----------------------------------------------------------------------------- save
def save_figure(fig, name, outdir, formats=("png", "pdf"), dpi=600):
    """Save a figure in the requested publication and preview formats."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = outdir / f"{name}.{extension}"
        save_kwargs = {"dpi": dpi} if extension == "png" else {}
        fig.savefig(path, **save_kwargs)
        paths.append(path)
    return paths
