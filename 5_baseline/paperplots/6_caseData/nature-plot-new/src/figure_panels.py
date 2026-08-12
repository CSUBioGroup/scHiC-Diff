"""Publication style and draw-only primitives for formal paper figures."""

import warnings
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from figure_registry import (
    ALL_METHODS,
    APA_TOP_N_VALUES,
    DISPLAY_LABELS,
    GROUP_STYLES,
    IMPUTED_METHODS,
    METHOD_STYLES,
    TOP_N_VALUES,
)


MM_PER_INCH = 25.4
PT_SMALL = 8.0
PT_BASE = 9.0
PT_TAG = 12.0
LW_HAIR = 0.4
TEXT_MAIN = "#000000"
TEXT_MUTED = "#5F5E5A"


def mm(value):
    return float(value) / MM_PER_INCH


def _pick_font():
    available = {item.name for item in fm.fontManager.ttflist}
    for name in ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "Arimo"):
        if name in available:
            return name
    warnings.warn("No Arial-compatible font found; using DejaVu Sans", stacklevel=2)
    return "DejaVu Sans"


def set_publication_style():
    font = _pick_font()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font, "DejaVu Sans"],
            "font.size": PT_BASE,
            "axes.labelsize": PT_BASE,
            "axes.titlesize": PT_BASE,
            "xtick.labelsize": PT_SMALL,
            "ytick.labelsize": PT_SMALL,
            "legend.fontsize": PT_SMALL,
            "axes.linewidth": LW_HAIR,
            "xtick.major.width": LW_HAIR,
            "ytick.major.width": LW_HAIR,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "lines.linewidth": 0.8,
            "patch.linewidth": LW_HAIR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def hic_cmap():
    return LinearSegmentedColormap.from_list(
        "hic_fall",
        ["#FFFFFF", "#FDD9C7", "#F7A98A", "#E8623C", "#B32316", "#5C0A06"],
    )


def add_axes_mm(fig, fig_w_mm, fig_h_mm, x_mm, top_mm, width_mm, height_mm):
    return fig.add_axes(
        [
            x_mm / fig_w_mm,
            1.0 - (top_mm + height_mm) / fig_h_mm,
            width_mm / fig_w_mm,
            height_mm / fig_h_mm,
        ]
    )


def panel_letter(fig, letter, x_mm, top_mm, fig_w_mm, fig_h_mm):
    fig.text(
        x_mm / fig_w_mm,
        1.0 - top_mm / fig_h_mm,
        str(letter).upper(),
        fontsize=PT_TAG,
        fontweight="bold",
        ha="left",
        va="top",
        color=TEXT_MAIN,
    )


def coverage_normalize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    total = float(np.triu(matrix, k=1).sum())
    return matrix / total if total > 0 else matrix.copy()


def _offdiag_vmax(matrix, percentile=99.0, diagonal_gap=2):
    matrix = np.asarray(matrix, dtype=float)
    i, j = np.indices(matrix.shape)
    values = matrix[(j - i) > diagonal_gap]
    if not values.size:
        return float(np.nanmax(matrix)) or 1.0
    value = float(np.nanpercentile(values, percentile))
    return value if value > 0 else float(np.nanmax(values)) or 1.0


def global_contact_scale(matrices):
    normalized = [
        coverage_normalize(matrices[key]) for key in sorted(matrices, key=str)
    ]
    vmax = float(np.median([_offdiag_vmax(matrix) for matrix in normalized]))
    return 0.0, vmax if vmax > 0 else 1.0


def global_apa_scale(matrices):
    values = np.concatenate(
        [np.asarray(matrices[key], dtype=float).ravel() for key in sorted(matrices, key=str)]
    )
    vmax = float(np.nanpercentile(values, 99.5))
    return 0.0, vmax if vmax > 0 else 1.0


def _silhouette_value(summary, method, stage):
    selected = summary.loc[
        summary["method_id"].eq(method) & summary["stage"].eq(stage)
    ]
    if len(selected) != 1:
        raise ValueError("expected one UMAP summary row for {}/{}".format(method, stage))
    return float(selected.iloc[0]["mean_silhouette"]), int(selected.iloc[0]["n_used"])


def _draw_umap_axis_indicator(fig, fig_w_mm, fig_h_mm, x_mm, top_mm, size_mm=5.5):
    """Draw a compact, label-free orientation glyph inside the UMAP gutter."""
    axis = add_axes_mm(fig, fig_w_mm, fig_h_mm, x_mm, top_mm, size_mm, size_mm)
    axis.set_axis_off()
    axis.annotate(
        "",
        xy=(0.92, 0.16),
        xytext=(0.18, 0.16),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "lw": 0.5, "color": TEXT_MUTED},
    )
    axis.annotate(
        "",
        xy=(0.18, 0.92),
        xytext=(0.18, 0.16),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "lw": 0.5, "color": TEXT_MUTED},
    )


def draw_umap_grid(
    fig,
    points,
    summary,
    methods,
    stages,
    fig_w_mm,
    fig_h_mm,
    x_mm,
    top_mm,
    width_mm,
    left_gutter_mm=15.0,
    right_mm=1.0,
    gap_mm=1.1,
    header_mm=6.5,
    legend_mm=6.0,
    point_size=1.4,
    alpha=0.75,
    show_legend=True,
    show_axis_indicator=True,
):
    n_rows = len(methods)
    n_columns = len(stages)
    cell = (
        width_mm - left_gutter_mm - right_mm - gap_mm * (n_columns - 1)
    ) / n_columns
    if cell <= 0:
        raise ValueError("UMAP grid width is too small")
    grid_top = top_mm + header_mm

    for row_index, method in enumerate(methods):
        for column_index, stage in enumerate(stages):
            x0 = x_mm + left_gutter_mm + column_index * (cell + gap_mm)
            y0 = grid_top + row_index * (cell + gap_mm)
            axis = add_axes_mm(fig, fig_w_mm, fig_h_mm, x0, y0, cell, cell)
            selected = points.loc[
                points["method_id"].eq(method) & points["stage"].eq(stage)
            ]
            if selected.empty:
                raise ValueError("missing UMAP points for {}/{}".format(method, stage))
            for lineage in ("Red", "Blue"):
                subset = selected.loc[selected["lineage"].eq(lineage)]
                style = GROUP_STYLES[lineage]
                axis.scatter(
                    subset["schUMAP_1"],
                    subset["schUMAP_2"],
                    s=point_size,
                    c=style["color"],
                    marker=style["marker"],
                    linewidths=0,
                    alpha=alpha,
                    rasterized=True,
                )
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

            silhouette, n_used = _silhouette_value(summary, method, stage)
            label = axis.text(
                0.97,
                0.03,
                "{:.3f}".format(silhouette).replace("-0.000", "0.000"),
                transform=axis.transAxes,
                fontsize=PT_SMALL,
                ha="right",
                va="bottom",
                color=TEXT_MAIN,
                zorder=10,
            )
            label.set_path_effects(
                [pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()]
            )
            if row_index == 0:
                axis.text(
                    0.5,
                    1.20,
                    stage,
                    transform=axis.transAxes,
                    fontsize=PT_BASE,
                    ha="center",
                    va="bottom",
                )
                axis.text(
                    0.5,
                    1.04,
                    "n = {}".format(n_used),
                    transform=axis.transAxes,
                    fontsize=PT_SMALL,
                    color=TEXT_MUTED,
                    ha="center",
                    va="bottom",
                )
            if column_index == 0:
                axis.text(
                    -0.10,
                    0.5,
                    DISPLAY_LABELS[method],
                    transform=axis.transAxes,
                    fontsize=PT_BASE,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontweight="bold" if method == "schicdiff" else "normal",
                )
        if method == "schicdiff":
            y0 = grid_top + row_index * (cell + gap_mm)
            rule = add_axes_mm(
                fig,
                fig_w_mm,
                fig_h_mm,
                x_mm + left_gutter_mm - 1.8,
                y0,
                0.55,
                cell,
            )
            rule.set_axis_off()
            rule.add_patch(
                plt.Rectangle((0, 0), 1, 1, transform=rule.transAxes, color=TEXT_MAIN, lw=0)
            )

    grid_bottom = grid_top + n_rows * cell + (n_rows - 1) * gap_mm
    if show_legend:
        legend_axis = add_axes_mm(
            fig,
            fig_w_mm,
            fig_h_mm,
            x_mm + left_gutter_mm,
            grid_bottom,
            width_mm - left_gutter_mm - right_mm,
            legend_mm,
        )
        legend_axis.set_axis_off()
        handles = [
            Line2D(
                [],
                [],
                ls="none",
                marker=GROUP_STYLES[lineage]["marker"],
                ms=3.2,
                mfc=GROUP_STYLES[lineage]["color"],
                mec=GROUP_STYLES[lineage]["color"],
                label=lineage,
            )
            for lineage in ("Red", "Blue")
        ]
        legend_axis.legend(handles=handles, loc="center", ncol=2, frameon=False)
    if show_axis_indicator:
        _draw_umap_axis_indicator(
            fig,
            fig_w_mm,
            fig_h_mm,
            x_mm + 2.0,
            grid_bottom - 6.0,
        )
    return grid_bottom + (legend_mm if show_legend else 2.0)


def draw_contact_grid(
    fig,
    matrices,
    summits,
    methods,
    cell_counts,
    all_matrices,
    fig_w_mm,
    fig_h_mm,
    x_mm,
    top_mm,
    width_mm,
    left_gutter_mm=14.0,
    right_mm=1.0,
    gap_mm=1.0,
    header_mm=5.0,
    colorbar_mm=6.0,
    show_colorbar_label=True,
    colorbar_offset_mm=2.0,
    colorbar_end_labels=False,
    colorbar_width_mm=None,
    colorbar_right_mm=None,
    colorbar_end_label_right_mm=None,
    genomic_range_label=None,
    genomic_range_x_mm=None,
    genomic_range_ha="left",
):
    n_rows = len(methods)
    n_columns = len(cell_counts)
    cell = (
        width_mm - left_gutter_mm - right_mm - gap_mm * (n_columns - 1)
    ) / n_columns
    if cell <= 0:
        raise ValueError("contact grid width is too small")
    grid_top = top_mm + header_mm
    vmin, vmax = global_contact_scale(all_matrices)
    cmap = hic_cmap()
    image = None

    for row_index, method in enumerate(methods):
        for column_index, count in enumerate(cell_counts):
            x0 = x_mm + left_gutter_mm + column_index * (cell + gap_mm)
            y0 = grid_top + row_index * (cell + gap_mm)
            axis = add_axes_mm(fig, fig_w_mm, fig_h_mm, x0, y0, cell, cell)
            matrix = coverage_normalize(matrices[(method, int(count))])
            image = axis.imshow(
                matrix,
                cmap=cmap,
                origin="upper",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            loop_bins = np.asarray(summits[(method, int(count))])
            if loop_bins.size:
                axis.scatter(
                    loop_bins[:, 1],
                    loop_bins[:, 0],
                    s=6,
                    facecolors="none",
                    edgecolors=TEXT_MAIN,
                    linewidths=0.45,
                    path_effects=[pe.withStroke(linewidth=1.3, foreground="white")],
                )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(LW_HAIR)
                spine.set_edgecolor("#444441")
            if row_index == 0:
                axis.text(
                    0.5,
                    1.05,
                    "{} cells".format(count),
                    transform=axis.transAxes,
                    fontsize=PT_BASE,
                    ha="center",
                    va="bottom",
                )
            if column_index == 0:
                axis.text(
                    -0.10,
                    0.5,
                    DISPLAY_LABELS[method],
                    transform=axis.transAxes,
                    fontsize=PT_BASE,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontweight="bold" if method == "schicdiff" else "normal",
                )
        if method == "schicdiff":
            y0 = grid_top + row_index * (cell + gap_mm)
            rule = add_axes_mm(
                fig,
                fig_w_mm,
                fig_h_mm,
                x_mm + left_gutter_mm - 1.8,
                y0,
                0.55,
                cell,
            )
            rule.set_axis_off()
            rule.add_patch(
                plt.Rectangle((0, 0), 1, 1, transform=rule.transAxes, color=TEXT_MAIN, lw=0)
            )

    grid_bottom = grid_top + n_rows * cell + (n_rows - 1) * gap_mm
    if image is not None:
        grid_left = x_mm + left_gutter_mm
        grid_width = width_mm - left_gutter_mm - right_mm
        default_colorbar_width = grid_width * 0.60
        colorbar_width = (
            default_colorbar_width
            if colorbar_width_mm is None
            else colorbar_width_mm
        )
        if colorbar_width <= 0:
            raise ValueError("contact colorbar width must be positive")
        colorbar_right = (
            grid_left + grid_width * 0.80
            if colorbar_right_mm is None
            else colorbar_right_mm
        )
        color_axis = add_axes_mm(
            fig,
            fig_w_mm,
            fig_h_mm,
            colorbar_right - colorbar_width,
            grid_bottom + colorbar_offset_mm,
            colorbar_width,
            1.5,
        )
        colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
        colorbar.outline.set_linewidth(LW_HAIR)
        if colorbar_end_labels:
            colorbar.set_ticks([])
            label_y = 1.0 - (grid_bottom + colorbar_offset_mm + 0.75) / fig_h_mm
            fig.text(
                (color_axis.get_position().x0 * fig_w_mm - 1.0) / fig_w_mm,
                label_y,
                "0",
                fontsize=PT_SMALL,
                ha="right",
                va="center",
            )
            fig.text(
                (
                    color_axis.get_position().x1 * fig_w_mm + 1.0
                    if colorbar_end_label_right_mm is None
                    else colorbar_end_label_right_mm
                )
                / fig_w_mm,
                label_y,
                "max",
                fontsize=PT_SMALL,
                ha="left" if colorbar_end_label_right_mm is None else "right",
                va="center",
            )
            if genomic_range_label is not None:
                fig.text(
                    (
                        grid_left
                        if genomic_range_x_mm is None
                        else genomic_range_x_mm
                    )
                    / fig_w_mm,
                    label_y,
                    genomic_range_label,
                    fontsize=PT_SMALL,
                    ha=genomic_range_ha,
                    va="center",
                )
        else:
            colorbar.set_ticks([vmin, vmax])
            colorbar.set_ticklabels(["0", "max"])
            color_axis.tick_params(
                labelsize=PT_SMALL, length=1.5, width=LW_HAIR, pad=1
            )
        if show_colorbar_label:
            color_axis.set_xlabel(
                "Normalized contacts", fontsize=PT_SMALL, labelpad=1
            )
    colorbar_extent = colorbar_offset_mm + (1.5 if colorbar_end_labels else 4.0)
    return grid_bottom + max(colorbar_mm, colorbar_extent)


def _metric_row(metrics, method, top_n):
    selected = metrics.loc[
        metrics["method_id"].eq(method) & metrics["top_n"].astype(int).eq(int(top_n))
    ]
    if len(selected) != 1:
        raise ValueError("expected one APA metric row for {}/{}".format(method, top_n))
    return selected.iloc[0]


def _draw_apa_cell(
    fig,
    matrix,
    metric,
    fig_w_mm,
    fig_h_mm,
    x_mm,
    top_mm,
    cell_mm,
    vmin,
    vmax,
    offset_kb,
    show_x,
    show_y,
    show_sd=True,
):
    axis = add_axes_mm(fig, fig_w_mm, fig_h_mm, x_mm, top_mm, cell_mm, cell_mm)
    image = axis.imshow(
        matrix,
        cmap=hic_cmap(),
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=(-offset_kb, offset_kb, -offset_kb, offset_kb),
        interpolation="nearest",
        aspect="equal",
    )
    ticks = [-offset_kb, 0, offset_kb]
    if show_x:
        axis.set_xticks(ticks)
        axis.set_xticklabels(
            ["−{}".format(offset_kb), "0", "+{}".format(offset_kb)]
        )
    else:
        axis.set_xticks([])
    if show_y:
        axis.set_yticks(ticks)
        axis.set_yticklabels(
            ["−{}".format(offset_kb), "0", "+{}".format(offset_kb)]
        )
    else:
        axis.set_yticks([])
    axis.tick_params(labelsize=PT_SMALL, length=1.4, width=LW_HAIR, pad=1)
    for spine in axis.spines.values():
        spine.set_linewidth(LW_HAIR)
    annotation = "{:.2f}".format(float(metric["p2ll_mean"]))
    if show_sd:
        annotation += "\n±{:.2f}".format(float(metric["p2ll_sd"]))
    axis.text(
        0.05,
        0.05,
        annotation,
        transform=axis.transAxes,
        fontsize=PT_SMALL,
        ha="left",
        va="bottom",
        linespacing=0.85,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
    )
    return axis, image


def draw_apa_grid(
    fig,
    matrices,
    metrics,
    methods,
    top_n_values,
    all_matrices,
    resolution_bp,
    window_bins,
    fig_w_mm,
    fig_h_mm,
    x_mm,
    top_mm,
    width_mm,
    left_gutter_mm=22.0,
    right_mm=1.0,
    gap_mm=1.2,
    header_mm=5.5,
    colorbar_mm=24.0,
    show_sd=True,
    show_colorbar_label=True,
    show_axis_titles=True,
    row_labels_with_n=True,
    row_label_rotation=0,
    row_label_x_mm=None,
    colorbar_offset_mm=13.0,
    colorbar_orientation="horizontal",
    vertical_colorbar_x_mm=None,
    vertical_colorbar_width_mm=1.5,
    method_header_fontsize=PT_BASE,
    vertical_colorbar_height_fraction=1.0,
    vertical_colorbar_end_labels=False,
):
    n_rows = len(top_n_values)
    n_columns = len(methods)
    cell = (
        width_mm - left_gutter_mm - right_mm - gap_mm * (n_columns - 1)
    ) / n_columns
    grid_top = top_mm + header_mm
    vmin, vmax = global_apa_scale(all_matrices)
    offset_kb = int(window_bins * resolution_bp / 1000)
    image = None
    for row_index, top_n in enumerate(top_n_values):
        for column_index, method in enumerate(methods):
            x0 = x_mm + left_gutter_mm + column_index * (cell + gap_mm)
            y0 = grid_top + row_index * (cell + gap_mm)
            metric = _metric_row(metrics, method, top_n)
            axis, image = _draw_apa_cell(
                fig,
                matrices[(method, int(top_n))],
                metric,
                fig_w_mm,
                fig_h_mm,
                x0,
                y0,
                cell,
                vmin,
                vmax,
                offset_kb,
                show_x=row_index == n_rows - 1 and column_index == 0,
                show_y=column_index == 0 and row_index == n_rows // 2,
                show_sd=show_sd,
            )
            if row_index == 0:
                axis.text(
                    0.5,
                    1.05,
                    DISPLAY_LABELS[method],
                    transform=axis.transAxes,
                    fontsize=method_header_fontsize,
                    ha="center",
                    va="bottom",
                    fontweight="bold" if method == "schicdiff" else "normal",
                )
            if column_index == 0:
                row_label = "Top{}".format(top_n)
                if row_labels_with_n:
                    row_label += "\n(n={})".format(top_n)
                row_label_x = (
                    -0.27
                    if row_label_x_mm is None
                    else (row_label_x_mm - x0) / cell
                )
                axis.text(
                    row_label_x,
                    0.5,
                    row_label,
                    transform=axis.transAxes,
                    fontsize=PT_SMALL,
                    ha="right" if row_label_rotation == 0 else "center",
                    va="center",
                    rotation=row_label_rotation,
                )
    grid_bottom = grid_top + n_rows * cell + (n_rows - 1) * gap_mm
    if image is not None and colorbar_orientation == "horizontal":
        color_axis = add_axes_mm(
            fig,
            fig_w_mm,
            fig_h_mm,
            x_mm + left_gutter_mm + (width_mm - left_gutter_mm) * 0.25,
            grid_bottom + colorbar_offset_mm,
            (width_mm - left_gutter_mm) * 0.50,
            1.5,
        )
        colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
        colorbar.outline.set_linewidth(LW_HAIR)
        colorbar.set_ticks([vmin, vmax])
        colorbar.set_ticklabels(["0", "{:.1f}".format(vmax)])
        color_axis.tick_params(labelsize=PT_SMALL, length=1.4, width=LW_HAIR, pad=1)
        if show_colorbar_label:
            color_axis.set_xlabel("Normalized APA", fontsize=PT_SMALL, labelpad=1)
    elif image is not None and colorbar_orientation == "vertical":
        if not 0 < vertical_colorbar_height_fraction <= 1:
            raise ValueError("vertical colorbar height fraction must be in (0, 1]")
        color_x = (
            x_mm + width_mm + 2.0
            if vertical_colorbar_x_mm is None
            else vertical_colorbar_x_mm
        )
        grid_height = grid_bottom - grid_top
        color_height = grid_height * vertical_colorbar_height_fraction
        color_top = grid_top + (grid_height - color_height) / 2.0
        color_axis = add_axes_mm(
            fig,
            fig_w_mm,
            fig_h_mm,
            color_x,
            color_top,
            vertical_colorbar_width_mm,
            color_height,
        )
        colorbar = fig.colorbar(image, cax=color_axis, orientation="vertical")
        colorbar.outline.set_linewidth(LW_HAIR)
        if vertical_colorbar_end_labels:
            colorbar.set_ticks([])
            label_x = (color_x + vertical_colorbar_width_mm / 2.0) / fig_w_mm
            fig.text(
                label_x,
                1.0 - (color_top - 0.8) / fig_h_mm,
                "{:.1f}".format(vmax),
                fontsize=PT_SMALL,
                ha="center",
                va="bottom",
            )
            fig.text(
                label_x,
                1.0 - (color_top + color_height + 0.8) / fig_h_mm,
                "0",
                fontsize=PT_SMALL,
                ha="center",
                va="top",
            )
        else:
            colorbar.set_ticks([vmin, vmax])
            colorbar.set_ticklabels(["0", "{:.1f}".format(vmax)])
            color_axis.yaxis.set_ticks_position("left")
            color_axis.yaxis.set_label_position("left")
            color_axis.tick_params(
                labelsize=PT_SMALL, length=1.4, width=LW_HAIR, pad=1
            )
        if show_colorbar_label:
            color_axis.set_ylabel("Normalized APA", fontsize=PT_SMALL, labelpad=1)
    elif image is not None:
        raise ValueError("unknown APA colorbar orientation: {}".format(colorbar_orientation))
    if show_axis_titles:
        fig.text(
            (x_mm + width_mm * 0.56) / fig_w_mm,
            1.0 - (grid_bottom + 5.0) / fig_h_mm,
            "Anchor 2 offset (kb)",
            fontsize=PT_SMALL,
            ha="center",
            va="top",
        )
        fig.text(
            (x_mm + 2.0) / fig_w_mm,
            1.0 - (grid_top + (grid_bottom - grid_top) / 2.0) / fig_h_mm,
            "Anchor 1 offset (kb)",
            fontsize=PT_SMALL,
            ha="center",
            va="center",
            rotation=90,
        )
    return grid_bottom + colorbar_mm


def draw_apa_two_block_grid(
    fig,
    matrices,
    metrics,
    blocks,
    top_n_values,
    all_matrices,
    resolution_bp,
    window_bins,
    fig_w_mm,
    fig_h_mm,
    x_mm,
    top_mm,
    width_mm,
    cell_mm=17.2,
    left_gutter_mm=13.0,
    gap_mm=1.0,
    header_mm=5.0,
    interblock_mm=5.0,
    colorbar_mm=16.0,
):
    vmin, vmax = global_apa_scale(all_matrices)
    offset_kb = int(window_bins * resolution_bp / 1000)
    n_rows = len(top_n_values)
    block_height = header_mm + n_rows * cell_mm + (n_rows - 1) * gap_mm
    image = None
    for block_index, methods in enumerate(blocks):
        block_top = top_mm + block_index * (block_height + interblock_mm)
        grid_top = block_top + header_mm
        used_width = len(methods) * cell_mm + (len(methods) - 1) * gap_mm
        available = width_mm - left_gutter_mm
        start_x = x_mm + left_gutter_mm + max(0.0, (available - used_width) / 2.0)
        for row_index, top_n in enumerate(top_n_values):
            for column_index, method in enumerate(methods):
                x0 = start_x + column_index * (cell_mm + gap_mm)
                y0 = grid_top + row_index * (cell_mm + gap_mm)
                axis, image = _draw_apa_cell(
                    fig,
                    matrices[(method, int(top_n))],
                    _metric_row(metrics, method, top_n),
                    fig_w_mm,
                    fig_h_mm,
                    x0,
                    y0,
                    cell_mm,
                    vmin,
                    vmax,
                    offset_kb,
                    show_x=row_index == n_rows - 1 and column_index == 0,
                    show_y=column_index == 0 and row_index == n_rows // 2,
                )
                if row_index == 0:
                    axis.text(
                        0.5,
                        1.05,
                        DISPLAY_LABELS[method],
                        transform=axis.transAxes,
                        fontsize=PT_SMALL,
                        ha="center",
                        va="bottom",
                        fontweight="bold" if method == "schicdiff" else "normal",
                    )
                if column_index == 0:
                    axis.text(
                        -0.27,
                        0.5,
                        "Top{}".format(top_n),
                        transform=axis.transAxes,
                        fontsize=PT_SMALL,
                        ha="right",
                        va="center",
                    )
    grid_bottom = top_mm + len(blocks) * block_height + (len(blocks) - 1) * interblock_mm
    if image is not None:
        color_axis = add_axes_mm(
            fig,
            fig_w_mm,
            fig_h_mm,
            x_mm + left_gutter_mm + (width_mm - left_gutter_mm) * 0.22,
            grid_bottom + 7.0,
            (width_mm - left_gutter_mm) * 0.56,
            1.4,
        )
        colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
        colorbar.outline.set_linewidth(LW_HAIR)
        colorbar.set_ticks([vmin, vmax])
        colorbar.set_ticklabels(["0", "{:.1f}".format(vmax)])
        color_axis.tick_params(labelsize=PT_SMALL, length=1.3, width=LW_HAIR, pad=1)
        color_axis.set_xlabel("Normalized APA", fontsize=PT_SMALL, labelpad=1)
    return grid_bottom + colorbar_mm


def actual_n_for(method, requested_n, support_frame):
    selected = support_frame.loc[
        support_frame["method_id"].eq(method)
        & support_frame["requested_n"].astype(int).eq(int(requested_n))
    ]
    if len(selected) != 1:
        raise ValueError("expected one actual-N row for {}/{}".format(method, requested_n))
    return float(selected.iloc[0]["actual_n_mean"])


def draw_support_fraction(axis, support_frame, show_legend=True, compact=False):
    x = np.arange(len(TOP_N_VALUES), dtype=float)
    for method in IMPUTED_METHODS:
        selected = support_frame.loc[support_frame["method_id"].eq(method)].copy()
        selected["_rank"] = selected["requested_n"].map(
            {value: index for index, value in enumerate(TOP_N_VALUES)}
        )
        selected = selected.sort_values("_rank")
        mean = selected["supported_fraction_mean"].to_numpy(float)
        sd = selected["supported_fraction_std"].to_numpy(float)
        style = METHOD_STYLES[method]
        visible_count = 4 if method == "schicluster" else len(x)
        axis.fill_between(
            x[:visible_count],
            np.clip(mean[:visible_count] - sd[:visible_count], 0, 1),
            np.clip(mean[:visible_count] + sd[:visible_count], 0, 1),
            color=style.color,
            alpha=0.13,
            linewidth=0,
            zorder=1,
        )
        plot_kwargs = {
            "color": style.color,
            "lw": 1.3 if method == "schicdiff" else 0.9,
            "marker": style.marker,
            "ms": 3.4 if method == "schicdiff" else 3.0,
            "mfc": style.color if style.filled else "white",
            "mec": style.color,
            "mew": 0.6,
            "zorder": 4 if method == "schicdiff" else 3,
        }
        if method == "schicluster":
            axis.plot(
                x[:3],
                mean[:3],
                ls="-",
                label=DISPLAY_LABELS[method],
                **plot_kwargs
            )
            axis.plot(x[2:4], mean[2:4], ls=":", label="_nolegend_", **plot_kwargs)
        else:
            axis.plot(
                x,
                mean,
                ls=style.linestyle,
                label=DISPLAY_LABELS[method],
                **plot_kwargs
            )

    axis.set_xticks(x)
    axis.set_xticklabels(["Top{}".format(value) for value in TOP_N_VALUES])
    axis.set_ylim(0, 1.0)
    axis.set_ylabel(
        "Raw-supported fraction" if compact else "Held-out raw-supported fraction"
    )
    axis.grid(axis="y", color="#D7D9DC", lw=0.4)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=LW_HAIR, length=2)
    if show_legend:
        # Keep the two-column legend in the manuscript reading order.  Matplotlib
        # fills this legend column-wise, so the first three entries form the left
        # column and the final three entries form the right column.
        legend_order = (
            "schicluster",
            "higashi_nbr0",
            "higashi_nbr5",
            "scvi3d",
            "flamingo",
            "schicdiff",
        )
        handles, labels = axis.get_legend_handles_labels()
        handles_by_label = dict(zip(labels, handles))
        legend_handles = [handles_by_label[DISPLAY_LABELS[method]] for method in legend_order]
        legend_labels = [
            "T-FLAMINGO" if method == "flamingo" else DISPLAY_LABELS[method]
            for method in legend_order
        ]
        axis.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            ncol=2,
            frameon=False,
            handlelength=2.1,
            columnspacing=0.9,
            labelspacing=0.25,
        )
    return axis


def _series_style(method):
    if method == "raw":
        return {
            "color": "#000000",
            "marker": "o",
            "linestyle": "--",
            "filled": False,
        }
    style = METHOD_STYLES[method]
    return {
        "color": style.color,
        "marker": style.marker,
        "linestyle": style.linestyle,
        "filled": style.filled,
    }


def draw_repeat_counts(axis, frame, metric, show_legend=True):
    if metric not in ("summit_count", "loop_count"):
        raise ValueError("unknown repeat-count metric: {}".format(metric))
    x = np.arange(4, dtype=float)
    for method in ALL_METHODS:
        selected = frame.loc[frame["method_id"].eq(method)].copy()
        selected["_rank"] = selected["cell_count"].map(
            {10: 0, 100: 1, 200: 2, 476: 3}
        )
        selected = selected.sort_values("_rank")
        mean = selected[metric + "_mean"].to_numpy(float)
        sd = selected[metric + "_sd"].to_numpy(float)
        style = _series_style(method)
        axis.plot(
            x,
            mean,
            color=style["color"],
            ls=style["linestyle"],
            lw=1.25 if method == "schicdiff" else 0.8,
            marker=style["marker"],
            ms=3.1,
            mfc=style["color"] if style["filled"] else "white",
            mec=style["color"],
            mew=0.6,
            label=DISPLAY_LABELS[method],
        )
        finite = np.isfinite(sd)
        axis.errorbar(
            x[finite],
            mean[finite],
            yerr=sd[finite],
            fmt="none",
            ecolor=style["color"],
            elinewidth=0.6,
            capsize=2,
            capthick=0.6,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(["10", "100", "200", "476"])
    axis.set_xlabel("Aggregated cells")
    axis.set_ylabel(
        "Distinct loop summits" if metric == "summit_count" else "Significant loop pixels"
    )
    axis.grid(axis="y", color="#D7D9DC", lw=0.4)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if show_legend:
        axis.legend(loc="upper left", ncol=3, frameon=False, columnspacing=0.8)
    return axis


def draw_supported_counts(axis, frame):
    ordered = []
    for method in IMPUTED_METHODS:
        selected = frame.loc[frame["method_id"].eq(method)]
        if len(selected) != 1:
            raise ValueError("expected one supported-count row for {}".format(method))
        ordered.append(selected.iloc[0])
    x = np.arange(len(ordered), dtype=float)
    values = np.asarray([float(row["supported_mean"]) for row in ordered])
    errors = np.asarray([float(row["supported_sd"]) for row in ordered])
    colors = [
        METHOD_STYLES[row["method_id"]].color
        if row["method_id"] == "schicdiff"
        else "#B4BAC2"
        for row in ordered
    ]
    axis.bar(
        x,
        values,
        yerr=errors,
        color=colors,
        edgecolor="none",
        error_kw={"ecolor": "#30343B", "elinewidth": 0.8, "capsize": 2},
    )
    for index, row in enumerate(ordered):
        axis.text(
            index,
            values[index] + errors[index] + 1.3,
            "{:.1f}\nEligible {:.1f}".format(
                float(row["supported_mean"]), float(row["eligible_mean"])
            ),
            ha="center",
            va="bottom",
            fontsize=PT_SMALL,
            color=TEXT_MUTED,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(
        [DISPLAY_LABELS[row["method_id"]] for row in ordered], rotation=25, ha="right"
    )
    axis.set_ylabel("Held-out raw-supported loops")
    axis.grid(axis="y", color="#D7D9DC", lw=0.4)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    return axis
