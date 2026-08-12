"""Shared publication style and primitives for the formal Ramani figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import Bbox


def mm(value):
    return value / 25.4


COL1 = mm(85)
COL15 = mm(114)
COL2 = mm(170)

FS_TICK = 8
FS_LABEL = 9
FS_TITLE = 9

INK = "#2B2F33"
HERO_COLOR = "#C0392B"
HERO_EDGE = "#7F1D14"
BASELINE_FACE = "#AEB4BC"
BASELINE_EDGE = "#5B6169"
RAW_FACE = "#D9DCE0"

CELL_TYPE_ORDER = ("HeLa", "HAP1", "K562", "GM12878")
CELL_TYPE_COLORS = {
    "HeLa": "#3F6FAE",
    "HAP1": "#D6A33B",
    "K562": "#8A6BB1",
    "GM12878": "#3C927D",
}
CELL_TYPE_MARKERS = {
    "HeLa": "o",
    "HAP1": "s",
    "K562": "^",
    "GM12878": "D",
}


def set_gr_style():
    chosen = "DejaVu Sans"
    for candidate in ("Liberation Sans", "Arial", "Helvetica"):
        try:
            fm.findfont(
                fm.FontProperties(family=candidate), fallback_to_default=False
            )
            chosen = candidate
            break
        except Exception:
            continue

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [chosen, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": FS_TICK,
            "axes.titlesize": FS_TITLE,
            "axes.labelsize": FS_LABEL,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_TICK,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "patch.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlepad": 4,
            "axes.labelpad": 2.5,
        }
    )
    return chosen


def panel_letter(axis, letter, x=-0.08, y=1.04):
    axis.text(
        x,
        y,
        letter,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="bottom",
        color=INK,
    )


def save_figure(
    figure,
    stem,
    output_dir,
    formats=("pdf", "png"),
    dpi=600,
    tight=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffixes = {
        "svg": ".svg",
        "pdf": ".pdf",
        "png": ".png",
        "tiff": ".tiff",
    }
    outputs = []
    for extension in formats:
        normalized = "tiff" if extension.lower() in {"tif", "tiff"} else extension.lower()
        if normalized not in suffixes:
            raise ValueError(f"unsupported figure format: {extension}")
        path = output_dir / f"{stem}{suffixes[normalized]}"
        kwargs = {"dpi": dpi}
        if tight:
            kwargs.update({"bbox_inches": "tight", "pad_inches": 0.02})
        else:
            width, height = figure.get_size_inches()
            kwargs.update(
                {
                    "bbox_inches": Bbox.from_bounds(0, 0, width, height),
                    "pad_inches": 0,
                }
            )
        if normalized == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        figure.savefig(path, **kwargs)
        outputs.append(path)
    return outputs


def plot_ari_bars(
    methods,
    values,
    errors=None,
    hero="scHiC-Diff",
    raw_names=("Raw", "Unprocessed"),
    axis=None,
    value_labels=True,
    value_label_size=8,
):
    values = np.asarray(values, dtype=float)
    errors = None if errors is None else np.asarray(errors, dtype=float)
    if axis is None:
        _, axis = plt.subplots(figsize=(COL2, mm(66)))

    x = np.arange(len(methods))
    faces = []
    edges = []
    widths = []
    for method in methods:
        if method == hero:
            faces.append(HERO_COLOR)
            edges.append(HERO_EDGE)
            widths.append(0.9)
        elif method in raw_names:
            faces.append(RAW_FACE)
            edges.append(BASELINE_EDGE)
            widths.append(0.6)
        else:
            faces.append(BASELINE_FACE)
            edges.append(BASELINE_EDGE)
            widths.append(0.6)

    bars = axis.bar(
        x,
        values,
        width=0.66,
        color=faces,
        edgecolor=edges,
        linewidth=widths,
        zorder=3,
    )
    if errors is not None:
        axis.errorbar(
            x,
            values,
            yerr=errors,
            fmt="none",
            ecolor=INK,
            elinewidth=0.8,
            capsize=2.2,
            capthick=0.8,
            zorder=4,
        )

    if value_labels:
        offset = max(float(np.nanmax(values)) * 0.02, 0.01)
        plotted_errors = errors if errors is not None else np.zeros(len(methods))
        for bar, value, error in zip(bars, values, plotted_errors):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + error + offset,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=value_label_size,
                fontweight="normal",
                color=INK,
                zorder=5,
            )

    axis.axhline(0, color=INK, linewidth=0.6, zorder=2)
    axis.set_xticks(x)
    axis.set_xticklabels(methods, rotation=30, ha="right")
    for label in axis.get_xticklabels():
        label.set_fontweight("normal")
        label.set_color("#4A4F55")
    axis.set_ylabel("ARI")
    high = float(np.nanmax(values + (errors if errors is not None else 0)))
    axis.set_ylim(min(0, float(np.nanmin(values))), high * 1.16)
    axis.yaxis.set_major_locator(MultipleLocator(0.2))
    axis.tick_params(axis="x", length=0)
    axis.margins(x=0.02)
    return axis


def plot_ari_point_ranges(
    methods,
    values,
    errors=None,
    hero="scHiC-Diff",
    raw_names=("Raw", "Unprocessed"),
    display_labels=None,
    axis=None,
    value_labels=True,
    value_label_size=7,
):
    values = np.asarray(values, dtype=float)
    has_errors = errors is not None
    plotted_errors = (
        np.zeros(len(values), dtype=float)
        if errors is None
        else np.asarray(errors, dtype=float)
    )
    if len(methods) != len(values) or len(plotted_errors) != len(values):
        raise ValueError("methods, values, and errors must have equal lengths")
    if axis is None:
        _, axis = plt.subplots(figsize=(COL15, mm(82)))

    display_labels = {} if display_labels is None else dict(display_labels)
    order = np.argsort(-values, kind="stable")
    ordered_methods = [methods[index] for index in order]
    y = np.arange(len(order))

    label_offset = max(float(np.nanmax(values)) * 0.045, 0.035)
    for position, index, method in zip(y, order, ordered_methods):
        is_hero = method == hero
        is_raw = method in raw_names
        face = HERO_COLOR if is_hero else ("white" if is_raw else BASELINE_FACE)
        edge = HERO_EDGE if is_hero else BASELINE_EDGE
        axis.errorbar(
            values[index],
            position,
            xerr=plotted_errors[index] if has_errors else None,
            fmt="o",
            markersize=4.8 if is_hero else 4.2,
            markerfacecolor=face,
            markeredgecolor=edge,
            markeredgewidth=0.7,
            ecolor=edge,
            elinewidth=0.8,
            capsize=2.0 if has_errors else 0,
            capthick=0.8,
            zorder=3,
        )
        if value_labels:
            marker_radius = 2.4 if is_hero else 2.1
            axis.annotate(
                f"{values[index]:.3f}",
                xy=(
                    values[index] + plotted_errors[index] + label_offset,
                    position,
                ),
                xytext=(0, -marker_radius),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=value_label_size,
                color=INK,
                zorder=4,
            )

    axis.set_yticks(
        y,
        [display_labels.get(method, method) for method in ordered_methods],
    )
    axis.invert_yaxis()
    right_extent = float(np.nanmax(values + plotted_errors)) + label_offset + 0.07
    axis.set_xlim(-0.02, max(0.85, right_extent))
    axis.set_xlabel("ARI")
    axis.xaxis.set_major_locator(MultipleLocator(0.2))
    axis.grid(axis="x", color="#E2E5E9", linewidth=0.55, zorder=0)
    axis.set_axisbelow(True)
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", length=0, pad=3)
    return axis


def plot_ari_sweep(
    axis,
    sweep,
    dims,
    hero="scHiC-Diff",
    mark_dim=10,
    selected_dims=None,
):
    methods = list(sweep)
    baselines = [method for method in methods if method != hero]
    grays = plt.cm.Greys(np.linspace(0.35, 0.72, max(len(baselines), 1)))
    for method, gray in zip(baselines, grays):
        is_raw = method in {"Raw", "Unprocessed"}
        axis.plot(
            dims,
            np.asarray(sweep[method], dtype=float),
            color=BASELINE_EDGE if is_raw else gray,
            linewidth=1.2 if is_raw else 1.0,
            linestyle=(0, (4, 2)) if is_raw else "-",
            marker="o",
            markersize=2.8,
            markerfacecolor=BASELINE_EDGE if is_raw else gray,
            markeredgecolor=BASELINE_EDGE if is_raw else gray,
            label=method,
            zorder=3,
        )
    if hero in sweep:
        axis.plot(
            dims,
            np.asarray(sweep[hero], dtype=float),
            color=HERO_COLOR,
            linewidth=2.2,
            marker="o",
            markersize=4.4,
            markerfacecolor=HERO_COLOR,
            markeredgecolor=HERO_EDGE,
            markeredgewidth=0.6,
            label=hero,
            zorder=5,
        )

    axis.set_xlim(float(np.min(dims)) - 0.25, float(np.max(dims)) + 0.25)
    axis.set_ylim(-0.05, 0.85)
    axis.grid(axis="y", color="#E2E5E9", linewidth=0.55, zorder=0)
    axis.set_axisbelow(True)
    if mark_dim is not None:
        axis.axvline(mark_dim, color=INK, linewidth=0.6, linestyle=(0, (3, 2)))
        axis.text(
            mark_dim + 0.45,
            0.54,
            f"ARI panel: first {mark_dim} components",
            transform=axis.get_xaxis_transform(),
            fontsize=7,
            ha="left",
            va="top",
            color=INK,
        )
    if selected_dims is not None:
        dims_array = np.asarray(dims, dtype=int)
        for method, selected_dim in selected_dims.items():
            if method not in sweep:
                continue
            matches = np.flatnonzero(dims_array == int(selected_dim))
            if len(matches) != 1:
                raise ValueError(f"selected dimension is absent for {method}")
            selected_value = np.asarray(sweep[method], dtype=float)[matches[0]]
            axis.scatter(
                [selected_dim],
                [selected_value],
                s=38,
                facecolors="none",
                edgecolors=INK,
                linewidths=0.9,
                zorder=7,
            )
        axis.text(
            0.985,
            0.52,
            "Black rings: ARI panel values",
            transform=axis.transAxes,
            fontsize=7,
            ha="right",
            va="center",
            color=INK,
        )
    axis.set_xlabel("Number of retained SVD components")
    axis.set_ylabel("ARI")
    axis.set_xticks(dims)
    axis.yaxis.set_major_locator(MultipleLocator(0.2))
    legend = axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        fontsize=7.0,
        ncol=4,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.4,
    )
    for text in legend.get_texts():
        text.set_fontweight("normal")
    return axis


def normalize_embedding(embedding):
    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2 or embedding.shape[1] != 2:
        raise ValueError("embedding must have shape (n, 2)")
    minimum = np.nanmin(embedding, axis=0)
    maximum = np.nanmax(embedding, axis=0)
    center = (minimum + maximum) / 2
    scale = float(np.max(maximum - minimum))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("embedding must contain a nonzero finite range")
    return (embedding - center) / scale


def plot_cluster_scatter(
    axis,
    embedding,
    cell_types,
    method_name,
    dot_size=4.8,
    alpha=0.84,
    axis_indicator=False,
    axis_indicator_fraction=0.13,
    axis_indicator_fontsize=6.5,
    title_size=FS_TITLE,
    title_pad=2.5,
    display_limits=(-0.78, 0.60),
    show_frame=False,
    frame_color="#8A8F95",
    frame_linewidth=0.55,
):
    display = normalize_embedding(embedding)
    cell_types = np.asarray(cell_types, dtype=object)
    counts = {
        cell_type: int(np.count_nonzero(cell_types == cell_type))
        for cell_type in CELL_TYPE_ORDER
    }
    for cell_type in sorted(CELL_TYPE_ORDER, key=lambda name: -counts[name]):
        mask = cell_types == cell_type
        axis.scatter(
            display[mask, 0],
            display[mask, 1],
            s=dot_size,
            marker=CELL_TYPE_MARKERS[cell_type],
            facecolor=CELL_TYPE_COLORS[cell_type],
            edgecolor="none",
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )

    axis.set_xlim(*display_limits)
    axis.set_ylim(*display_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_box_aspect(1)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(show_frame)
        if show_frame:
            spine.set_color(frame_color)
            spine.set_linewidth(frame_linewidth)
    axis.set_title(
        method_name,
        fontsize=title_size,
        fontweight="normal",
        pad=title_pad,
    )
    if axis_indicator:
        draw_umap_axis_indicator(
            axis,
            fraction=axis_indicator_fraction,
            fontsize=axis_indicator_fontsize,
        )
    return axis


def draw_umap_axis_indicator(
    axis,
    fraction=0.13,
    fontsize=6.5,
    origin=(0.055, 0.065),
    vertical_y_label=False,
    linewidth=0.8,
    mutation_scale=10,
):
    origin_x, origin_y = origin
    arrow = {
        "arrowstyle": "-|>",
        "color": INK,
        "lw": linewidth,
        "mutation_scale": mutation_scale,
        "shrinkA": 0,
        "shrinkB": 0,
        "zorder": 8,
    }
    axis.annotate(
        "",
        xy=(origin_x + fraction, origin_y),
        xytext=(origin_x, origin_y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=arrow,
        annotation_clip=False,
        zorder=8,
    )
    axis.annotate(
        "",
        xy=(origin_x, origin_y + fraction),
        xytext=(origin_x, origin_y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=arrow,
        annotation_clip=False,
        zorder=8,
    )
    axis.text(
        origin_x + fraction * 1.05,
        origin_y,
        "UMAP1",
        transform=axis.transAxes,
        fontsize=fontsize,
        ha="left",
        va="center",
        clip_on=False,
        zorder=8,
    )
    if vertical_y_label:
        axis.text(
            origin_x - 0.035,
            origin_y + fraction * 0.5,
            "UMAP2",
            transform=axis.transAxes,
            fontsize=fontsize,
            rotation=90,
            ha="right",
            va="center",
            clip_on=False,
            zorder=8,
        )
    else:
        axis.text(
            origin_x,
            origin_y + fraction * 1.05,
            "UMAP2",
            transform=axis.transAxes,
            fontsize=fontsize,
            ha="center",
            va="bottom",
            clip_on=False,
            zorder=8,
        )


def cell_type_legend_handles(counts, show_counts=True):
    handles = []
    for cell_type in CELL_TYPE_ORDER:
        label = (
            f"{cell_type} (n={int(counts[cell_type])})"
            if show_counts
            else cell_type
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker=CELL_TYPE_MARKERS[cell_type],
                linestyle="none",
                markerfacecolor=CELL_TYPE_COLORS[cell_type],
                markeredgecolor="white",
                markeredgewidth=0.3,
                markersize=5.5,
                label=label,
            )
        )
    return handles
