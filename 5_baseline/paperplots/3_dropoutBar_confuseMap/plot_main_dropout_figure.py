#!/usr/bin/env python3
"""Create the main simulated-data figure with dropout PCC and MAE only."""
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
    FS_TITLE,
    GR_DOUBLE_COL_IN,
    apply_nature_style,
)
from plot_dropout_pcc_mae_bar import (
    DATASET_CONFIGS,
    METHODS,
    draw_flamingo_metric_grid,
    draw_hicimpute_metric_grid,
    legend_handles,
    load_metrics,
)


PLOT_ROOT = Path(__file__).resolve().parent
DEFAULT_HICIMPUTE_INPUT = DATASET_CONFIGS["hicimpute"]["metrics_path"]
DEFAULT_FLAMINGO_INPUT = DATASET_CONFIGS["flamingo"]["metrics_path"]
DEFAULT_OUTPUT_DIR = PLOT_ROOT / "figures/main_dropout_figure"
METRICS = ("pcc", "mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hicimpute-input", type=Path, default=DEFAULT_HICIMPUTE_INPUT
    )
    parser.add_argument("--flamingo-input", type=Path, default=DEFAULT_FLAMINGO_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=("pdf", "png"))
    parser.add_argument("--raster-dpi", type=int, default=600)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def plot_main_figure(hicimpute_frame, flamingo_frame):
    figure = plt.figure(figsize=(GR_DOUBLE_COL_IN, 6.55))
    layout = figure.add_gridspec(
        3,
        1,
        height_ratios=(1.0, 0.085, 0.82),
        hspace=0.12,
        left=0.105,
        right=0.985,
        top=0.952,
        bottom=0.05,
    )

    hicimpute_grid = layout[0, 0].subgridspec(
        len(METRICS), 3, wspace=0.16, hspace=0.28
    )
    hicimpute_start = len(figure.axes)
    draw_hicimpute_metric_grid(
        figure,
        hicimpute_grid,
        hicimpute_frame,
        METRICS,
        show_xlabel=False,
    )
    hicimpute_axes = figure.axes[hicimpute_start:]

    legend_axis = figure.add_subplot(layout[1, 0])
    legend_axis.axis("off")
    legend_axis.legend(
        legend_handles(),
        METHODS,
        loc="center",
        ncol=len(METHODS),
        fontsize=FS_TICK,
        handlelength=0.85,
        handletextpad=0.3,
        columnspacing=0.55,
        borderaxespad=0.0,
    )

    flamingo_grid = layout[2, 0].subgridspec(
        len(METRICS),
        2,
        width_ratios=(5, 3),
        wspace=0.18,
        hspace=0.28,
    )
    flamingo_start = len(figure.axes)
    draw_flamingo_metric_grid(
        figure,
        flamingo_grid,
        flamingo_frame,
        METRICS,
        show_xlabel=True,
    )
    flamingo_axes = figure.axes[flamingo_start:]

    # Retain compact panel markers while leaving the benchmark names out of
    # the artwork; the panel-specific condition labels identify each sweep.
    panel_x = 0.012
    header_offset = 0.021
    figure.text(
        panel_x,
        hicimpute_axes[0].get_position().y1 + header_offset,
        "A",
        ha="left",
        va="bottom",
        fontsize=FS_PANEL,
        fontweight="bold",
        color=C_INK,
    )
    figure.text(
        panel_x,
        flamingo_axes[0].get_position().y1 + header_offset,
        "B",
        ha="left",
        va="bottom",
        fontsize=FS_PANEL,
        fontweight="bold",
        color=C_INK,
    )
    return figure


def save_outputs(figure, output_dir: Path, formats, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    stem = "Simulated_dropout_PCC_MAE_main_figure"
    for extension in formats:
        normalized = "tiff" if extension.lower() in {"tif", "tiff"} else extension.lower()
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
    hicimpute_frame = load_metrics(args.hicimpute_input, "hicimpute")
    flamingo_frame = load_metrics(args.flamingo_input, "flamingo")
    print(f"Validated {len(hicimpute_frame)} HiCImputeData dropout rows")
    print(f"Validated {len(flamingo_frame)} FLAMINGOData dropout rows")
    if args.validate_only:
        return
    figure = plot_main_figure(hicimpute_frame, flamingo_frame)
    outputs = save_outputs(figure, args.output_dir, args.formats, args.raster_dpi)
    plt.close(figure)
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
