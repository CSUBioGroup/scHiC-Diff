#!/usr/bin/env python3
"""Create the three-panel HiCImputeData PCC/MAE, ROC, and PR main figure."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from nature_plot_style import (
    C_INK,
    FS_LABEL,
    FS_PANEL,
    FS_TICK,
    FS_TITLE,
    GR_DOUBLE_COL_IN,
    HERO_METHOD,
    NATURE_COLORS,
    apply_nature_style,
)
from plot_dropout_pcc_mae_bar import (
    DATASET_CONFIGS,
    METHODS,
    draw_hicimpute_metric_grid,
    load_metrics as load_dropout_metrics,
)
from plot_sz_do_roc_pr_curves import CELL_TYPES, DEPTHS, load_inputs


PLOT_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = PLOT_ROOT / "1_HiCImputedData"
DEFAULT_DROPOUT_INPUT = DATASET_CONFIGS["hicimpute"]["metrics_path"]
DEFAULT_SUMMARY = INPUT_ROOT / "HiCImputeData_SZ_DO_5fold_OOF_ROC_PR_AUC.tsv"
DEFAULT_POINTS = INPUT_ROOT / "HiCImputeData_SZ_DO_5fold_OOF_curve_plot_points.tsv"
DEFAULT_OUTPUT_DIR = INPUT_ROOT / "figures/main_figure"
BAR_METRICS = ("pcc", "mae")
OUTPUT_STEM = "HiCImputeData_main_PCC_MAE_ROC_PR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dropout-input", type=Path, default=DEFAULT_DROPOUT_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=("pdf", "png"))
    parser.add_argument("--raster-dpi", type=int, default=600)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def curve_legend_handles() -> list[Line2D]:
    """Use line handles because the shared legend describes ROC/PR curves."""
    return [
        Line2D(
            [0],
            [0],
            color=NATURE_COLORS[method],
            lw=0.8,
            label=method,
        )
        for method in METHODS
    ]


def style_compact_curve_axis(
    axis,
    column_index: int,
    row_index: int,
) -> None:
    """Format a compact single-condition ROC or PR panel."""
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_box_aspect(1)
    ticks = np.array((0.0, 0.5, 1.0))
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_facecolor("#F8F8F8")
    axis.grid(axis="both", color="#E0E0E0", lw=0.35, zorder=0)
    axis.tick_params(axis="both", pad=0.7, labelsize=FS_TICK, length=2.4)

    if column_index:
        axis.set_yticklabels([])
    else:
        axis.set_yticklabels(("0", "", "1"))

    if row_index < len(CELL_TYPES) - 1:
        axis.set_xticklabels([])
    else:
        # The mini-panels share boundaries. Suppressing adjacent endpoint
        # labels keeps the four-column group readable at journal width.
        labels = ["0", "0.5", "1"]
        if column_index:
            labels[0] = ""
        if column_index < len(DEPTHS) - 1:
            labels[-1] = ""
        axis.set_xticklabels(labels)


def draw_compact_curve_grid(
    figure,
    subplot_spec,
    summary: pd.DataFrame,
    points: pd.DataFrame,
    kind: str,
) -> list:
    """Draw a 3 x 4 ROC or PR grid into one half of the bottom row."""
    if kind not in {"roc", "pr"}:
        raise ValueError(kind)

    grid = subplot_spec.subgridspec(
        len(CELL_TYPES) + 1,
        len(DEPTHS),
        height_ratios=(1.0, 1.0, 1.0, 0.30),
        hspace=0.18,
        wspace=0.12,
    )
    axes = []
    for row_index, ctype in enumerate(CELL_TYPES):
        for column_index, depth in enumerate(DEPTHS):
            axis = figure.add_subplot(grid[row_index, column_index])
            axes.append(axis)
            style_compact_curve_axis(axis, column_index, row_index)
            condition = points[
                (points["ctype"] == ctype) & (points["cdepth"] == depth)
            ]
            condition_summary = summary[
                (summary["ctype"] == ctype) & (summary["cdepth"] == depth)
            ]

            if kind == "roc":
                axis.plot(
                    [0, 1],
                    [0, 1],
                    color="#8C8C8C",
                    ls="--",
                    lw=0.38,
                    zorder=1,
                )
                x_column, y_column = "fpr_do", "tpr_sz"
            else:
                prevalence = float(
                    condition_summary.iloc[0]["true_sz_count"]
                    / condition_summary.iloc[0]["candidate_count"]
                )
                axis.axhline(
                    prevalence,
                    color="#8C8C8C",
                    ls="--",
                    lw=0.38,
                    zorder=1,
                )
                x_column, y_column = "recall_sz", "precision_sz"

            for method in METHODS:
                curve = condition[condition["method"] == method].sort_values(
                    "point_index"
                )
                axis.plot(
                    curve[x_column],
                    curve[y_column],
                    color=NATURE_COLORS[method],
                    lw=0.55,
                    zorder=3,
                )

            if row_index == 0:
                axis.set_title(depth.upper(), fontsize=FS_TICK, pad=2.0)

    xlabel_axis = figure.add_subplot(grid[-1, :])
    xlabel_axis.axis("off")
    xlabel_axis.text(
        0.5,
        0.0,
        "FPR" if kind == "roc" else "Recall",
        ha="center",
        va="bottom",
        fontsize=FS_LABEL,
    )
    return axes


def add_curve_group_labels(
    figure,
    axes: list,
    panel: str,
    kind: str,
    panel_x: float | None = None,
    cell_label_x: float | None = None,
    cell_label_weight: str = "normal",
    show_cell_labels: bool = True,
) -> None:
    """Add panel, cell-type, and shared y-axis labels around a curve group."""
    first_axis = axes[0]
    last_axis = axes[-1]
    first_position = first_axis.get_position()
    last_position = last_axis.get_position()
    x0 = first_position.x0
    y0 = last_position.y0
    y1 = first_position.y1
    ylabel = "TPR" if kind == "roc" else "Precision"

    figure.text(
        x0 - 0.027 if panel_x is None else panel_x,
        y1 + 0.004,
        panel,
        ha="left",
        va="bottom",
        fontsize=FS_PANEL,
        fontweight="bold",
        color=C_INK,
    )
    ylabel_offset = 0.020 if kind == "roc" else 0.035
    figure.text(
        x0 - ylabel_offset,
        (y0 + y1) / 2,
        ylabel,
        rotation=90,
        ha="center",
        va="center",
        fontsize=FS_LABEL,
        color=C_INK,
    )
    if not show_cell_labels:
        return
    for row_index, ctype in enumerate(CELL_TYPES):
        position = axes[row_index * len(DEPTHS)].get_position()
        figure.text(
            x0 - 0.076 if cell_label_x is None else cell_label_x,
            (position.y0 + position.y1) / 2,
            ctype,
            rotation=90,
            ha="center",
            va="center",
            fontsize=FS_TITLE,
            fontweight=cell_label_weight,
            color=C_INK,
        )


def ylabel_center_x(figure, axis) -> float:
    """Return the figure-coordinate center of an existing y-axis label."""
    renderer = figure.canvas.get_renderer()
    bbox = axis.yaxis.label.get_window_extent(renderer=renderer)
    return float(
        figure.transFigure.inverted().transform(
            ((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)
        )[0]
    )


def plot_main_figure(
    dropout_frame: pd.DataFrame,
    summary: pd.DataFrame,
    points: pd.DataFrame,
):
    """Compose panel A above compact side-by-side panels B and C."""
    figure = plt.figure(figsize=(GR_DOUBLE_COL_IN, 7.0))
    figure.subplots_adjust(left=0.112, right=0.988, top=0.972, bottom=0.018)
    layout = figure.add_gridspec(
        5,
        1,
        height_ratios=(2.42, 0.16, 0.23, 0.13, 1.68),
        hspace=0.0,
    )

    bar_grid = layout[0, 0].subgridspec(
        len(BAR_METRICS),
        3,
        wspace=0.16,
        hspace=0.28,
    )
    bar_axis_start = len(figure.axes)
    draw_hicimpute_metric_grid(
        figure,
        bar_grid,
        dropout_frame,
        BAR_METRICS,
        show_xlabel=False,
    )
    bar_axes = figure.axes[bar_axis_start:]
    for axis in bar_axes:
        axis.yaxis.label.set_fontweight("normal")
        axis.title.set_fontweight("normal")

    legend_axis = figure.add_subplot(layout[2, 0])
    legend_axis.axis("off")
    handles = curve_legend_handles()
    legend_axis.legend(
        handles,
        METHODS,
        loc="center",
        ncol=len(METHODS),
        fontsize=FS_TICK,
        handlelength=1.0,
        handletextpad=0.32,
        columnspacing=0.52,
        labelspacing=0.0,
        borderaxespad=0.0,
    )

    curve_groups = layout[4, 0].subgridspec(1, 2, wspace=0.22)
    roc_axes = draw_compact_curve_grid(
        figure,
        curve_groups[0, 0],
        summary,
        points,
        "roc",
    )
    pr_axes = draw_compact_curve_grid(
        figure,
        curve_groups[0, 1],
        summary,
        points,
        "pr",
    )

    panel_x = 0.008
    figure.text(
        panel_x,
        bar_axes[0].get_position().y1 + 0.004,
        "A",
        ha="left",
        va="bottom",
        fontsize=FS_PANEL,
        fontweight="bold",
        color=C_INK,
    )
    figure.canvas.draw()
    pcc_mae_label_x = ylabel_center_x(figure, bar_axes[0])
    add_curve_group_labels(
        figure,
        roc_axes,
        "B",
        "roc",
        panel_x=panel_x,
        cell_label_x=pcc_mae_label_x,
        cell_label_weight="normal",
    )
    add_curve_group_labels(
        figure,
        pr_axes,
        "C",
        "pr",
        panel_x=pr_axes[0].get_position().x0 - 0.065,
        show_cell_labels=False,
    )
    return figure


def save_outputs(figure, output_dir: Path, formats, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in formats:
        normalized = "tiff" if extension.lower() in {"tif", "tiff"} else extension.lower()
        path = output_dir / f"{OUTPUT_STEM}.{normalized}"
        kwargs = {"dpi": dpi}
        if normalized == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        figure.savefig(path, **kwargs)
        outputs.append(path)
    return outputs


def main() -> None:
    args = parse_args()
    apply_nature_style()
    dropout_frame = load_dropout_metrics(args.dropout_input, "hicimpute")
    summary, points = load_inputs(args.summary, args.points)
    print(f"Validated {len(dropout_frame)} PCC/MAE rows from {args.dropout_input}")
    print(f"Validated {len(summary)} AUC rows and {len(points)} curve rows")
    if args.validate_only:
        return

    figure = plot_main_figure(dropout_frame, summary, points)
    outputs = save_outputs(figure, args.output_dir, args.formats, args.raster_dpi)
    plt.close(figure)
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
