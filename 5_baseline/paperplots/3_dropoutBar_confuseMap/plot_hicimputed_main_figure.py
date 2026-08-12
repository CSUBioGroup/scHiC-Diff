#!/usr/bin/env python3
"""Plot the complete HiCImputeData main figure with continuous and SZ/DO panels."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nature_plot_style import (
    C_INK,
    FS_PANEL,
    FS_TICK,
    GR_DOUBLE_COL_IN,
    apply_nature_style,
)
from plot_dropout_pcc_mae_bar import (
    DATASET_CONFIGS,
    draw_hicimpute_metric_grid,
    legend_handles,
    load_metrics as load_dropout_metrics,
    METHODS,
)
from plot_sz_do_metric_lines import (
    DEFAULT_INPUT as DEFAULT_CLASSIFICATION_INPUT,
    draw_line_metric_grid,
    load_metrics as load_classification_metrics,
)


PLOT_ROOT = Path(__file__).resolve().parent
DEFAULT_DROPOUT_INPUT = DATASET_CONFIGS["hicimpute"]["metrics_path"]
DEFAULT_OUTPUT_DIR = PLOT_ROOT / "1_HiCImputedData/figures/main_figure"
BAR_METRICS = ("pcc", "mae")
LINE_METRICS = ("f1", "precision", "recall", "specificity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dropout-input", type=Path, default=DEFAULT_DROPOUT_INPUT)
    parser.add_argument(
        "--classification-input",
        type=Path,
        default=DEFAULT_CLASSIFICATION_INPUT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=("pdf", "png"))
    parser.add_argument("--raster-dpi", type=int, default=600)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def plot_main_figure(dropout_frame, classification_frame):
    figure = plt.figure(figsize=(GR_DOUBLE_COL_IN, 6.8))
    figure.subplots_adjust(left=0.105, right=0.985, top=0.97, bottom=0.025)
    layout = figure.add_gridspec(
        3,
        1,
        height_ratios=(2.05, 0.24, 3.25),
        hspace=0.12,
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

    line_grid = layout[2, 0].subgridspec(
        len(LINE_METRICS),
        3,
        wspace=0.16,
        hspace=0.18,
    )
    line_axis_start = len(figure.axes)
    draw_line_metric_grid(
        figure,
        line_grid,
        classification_frame,
        LINE_METRICS,
        show_xlabel=False,
        hide_zero_ytick_label=True,
    )
    line_axes = figure.axes[line_axis_start:]

    legend_axis = figure.add_subplot(layout[1, 0])
    legend_axis.axis("off")
    handles = legend_handles()
    legend_axis.legend(
        handles,
        METHODS,
        loc="center",
        ncol=len(METHODS),
        fontsize=FS_TICK,
        handlelength=0.8,
        handletextpad=0.25,
        columnspacing=0.55,
        labelspacing=0.0,
        borderaxespad=0.0,
    )
    panel_x = 0.006
    figure.text(
        panel_x,
        bar_axes[0].get_position().y1 + 0.003,
        "A",
        ha="left",
        va="bottom",
        fontsize=FS_PANEL,
        fontweight="bold",
        color=C_INK,
    )
    figure.text(
        panel_x,
        line_axes[0].get_position().y1 + 0.003,
        "B",
        ha="left",
        va="bottom",
        fontsize=FS_PANEL,
        fontweight="bold",
        color=C_INK,
    )
    return figure


def save_outputs(figure, output_dir: Path, formats, dpi: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "HiCImputeData_main_PCC_MAE_SZ_DO_classification"
    outputs = []
    for extension in formats:
        normalized = (
            "tiff" if extension.lower() in {"tif", "tiff"} else extension.lower()
        )
        path = output_dir / f"{stem}.{normalized}"
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
    classification_frame = load_classification_metrics(args.classification_input)
    print(f"Validated {len(dropout_frame)} dropout rows from {args.dropout_input}")
    print(
        f"Validated {len(classification_frame)} classification rows from "
        f"{args.classification_input}"
    )
    if args.validate_only:
        return

    figure = plot_main_figure(dropout_frame, classification_frame)
    outputs = save_outputs(figure, args.output_dir, args.formats, args.raster_dpi)
    plt.close(figure)
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
