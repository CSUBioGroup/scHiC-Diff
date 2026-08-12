#!/usr/bin/env python3
"""Plot combined five-fold OOF HiCImputeData confusion matrices by cell type."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec


PLOT_ROOT = Path(__file__).resolve().parent
if str(PLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOT_ROOT))

from nature_plot_style import (  # noqa: E402
    FS_ANNOT,
    FS_LABEL,
    FS_TICK,
    FS_TITLE,
    GR_DOUBLE_COL_IN,
    HERO_EDGE,
    HERO_METHOD,
    apply_nature_style,
)


DEFAULT_INPUT = (
    PLOT_ROOT / "1_HiCImputedData/HiCImputeData_SZ_DO_5fold_OOF_metrics.tsv"
)
DEFAULT_OUTPUT_DIR = PLOT_ROOT / "1_HiCImputedData/figures/sz_do_confusion_matrix"
CANONICAL_METHODS = (
    "scHiCluster",
    "HiCImpute",
    "Higashi_nbr0",
    "Higashi_nbr5",
    "scVI-3D",
    "Tensor-FLAMINGO",
    "scHiC-Diff",
)
METHOD_LABELS = {
    "scHiCluster": "scHiCluster",
    "HiCImpute": "HiCImpute",
    "Higashi_nbr0": "Higashi-0",
    "Higashi_nbr5": "Higashi-5",
    "scVI-3D": "scVI-3D",
    "Tensor-FLAMINGO": "T-FLAMINGO",
    "scHiC-Diff": "scHiC-Diff",
}
CELL_TYPES = ("T1", "T2", "T3")
DEPTHS = ("1k", "2k", "4k", "7k")
MATRIX_COLUMNS = (
    "true_sz_pred_sz",
    "true_sz_pred_do",
    "true_do_pred_sz",
    "true_do_pred_do",
)
CMAP = LinearSegmentedColormap.from_list(
    "nature_green",
    ("#F7FCF5", "#C7E9C0", "#74C476", "#238B45", "#005A32"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cell-types",
        nargs="+",
        choices=CELL_TYPES,
        default=CELL_TYPES,
    )
    parser.add_argument("--formats", nargs="+", default=("pdf", "png"))
    parser.add_argument("--raster-dpi", type=int, default=600)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t")
    required = {
        "method",
        "data_name",
        "ctype",
        "cdepth",
        "positive_class",
        "evaluation_scope",
        "threshold_mode",
        "threshold_selection",
        "crossfit_folds",
        "f1_sz",
        *MATRIX_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    expected = {
        (method, f"K562_{cell_type}_{depth}")
        for method in CANONICAL_METHODS
        for cell_type in CELL_TYPES
        for depth in DEPTHS
    }
    keys = list(zip(frame["method"], frame["data_name"]))
    if pd.Series(keys).duplicated().any() or set(keys) != expected:
        raise ValueError("Expected exactly 7 methods x 12 HiCImputeData conditions")
    if set(frame["positive_class"]) != {"SZ"}:
        raise ValueError("Expected SZ as the positive class")
    if set(frame["evaluation_scope"]) != {"observed_zero"}:
        raise ValueError("Expected observed_zero evaluation scope")
    if set(frame["threshold_mode"]) != {
        "5fold_cellwise_crossfit_method_depth"
    }:
        raise ValueError("Unexpected threshold mode")
    if set(frame["threshold_selection"]) != {"max_MCC"}:
        raise ValueError("Expected max_MCC threshold selection")
    if set(frame["crossfit_folds"]) != {5}:
        raise ValueError("Expected five cross-fitting folds")
    return frame


def matrix_from_row(row: pd.Series) -> np.ndarray:
    return np.asarray(
        [
            [row.true_sz_pred_sz, row.true_sz_pred_do],
            [row.true_do_pred_sz, row.true_do_pred_do],
        ],
        dtype=float,
    )


def style_matrix_axis(
    axis,
    matrix: np.ndarray,
    method: str,
    depth: str,
    f1: float,
    row_index: int,
    column_index: int,
) -> None:
    axis.imshow(matrix, cmap=CMAP, vmin=0, vmax=1, aspect="equal")
    for true_index in range(2):
        for predicted_index in range(2):
            value = matrix[true_index, predicted_index]
            axis.text(
                predicted_index,
                true_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=FS_ANNOT,
                color="white" if value >= 0.58 else "#252525",
            )
    if row_index == len(DEPTHS) - 1:
        axis.set_xticks((0, 1), ("SZ", "DO"))
    else:
        axis.set_xticks(())
    if column_index == 0:
        axis.set_yticks((0, 1), ("SZ", "DO"))
        axis.set_ylabel(f"{depth.upper()}\nTrue class", fontweight="bold", labelpad=2)
    else:
        axis.set_yticks(())

    f1_label = "NA" if pd.isna(f1) else f"{f1:.2f}"
    title = (
        f"{METHOD_LABELS[method]}\nF1 = {f1_label}"
        if row_index == 0
        else f"F1 = {f1_label}"
    )
    axis.set_title(
        title,
        color=HERO_EDGE if method == HERO_METHOD else "#1A1A1A",
        fontweight="bold" if method == HERO_METHOD else "normal",
        fontsize=FS_TICK,
        pad=2.5,
    )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9 if method == HERO_METHOD else 0.45)
        spine.set_color(HERO_EDGE if method == HERO_METHOD else "#606060")


def plot_cell_type(frame: pd.DataFrame, cell_type: str):
    indexed = frame[frame["ctype"] == cell_type].set_index(["method", "cdepth"])
    figure = plt.figure(figsize=(GR_DOUBLE_COL_IN, 5.05))
    outer = GridSpec(1, 2, width_ratios=[1, 0.035], wspace=0.06)
    panels = outer[0, 0].subgridspec(
        len(DEPTHS),
        len(CANONICAL_METHODS),
        wspace=0.18,
        hspace=0.42,
    )
    image_artist = None
    for row_index, depth in enumerate(DEPTHS):
        for column_index, method in enumerate(CANONICAL_METHODS):
            row = indexed.loc[(method, depth)]
            axis = figure.add_subplot(panels[row_index, column_index])
            style_matrix_axis(
                axis,
                matrix_from_row(row),
                method,
                depth,
                float(row.f1_sz),
                row_index,
                column_index,
            )
            if image_artist is None:
                image_artist = axis.images[0]

    colorbar_axis = figure.add_subplot(outer[0, 1])
    colorbar = figure.colorbar(image_artist, cax=colorbar_axis)
    colorbar.set_ticks((0, 0.5, 1))
    colorbar.ax.tick_params(labelsize=FS_TICK)
    figure.text(
        0.47,
        0.026,
        "Predicted class",
        ha="center",
        va="bottom",
        fontsize=FS_LABEL,
        fontweight="bold",
    )
    figure.suptitle(
        f"{cell_type} cell type: SZ/DO classification",
        y=0.998,
        fontsize=FS_TITLE,
        fontweight="bold",
    )
    return figure


def save_outputs(fig, stem: str, output_dir: Path, formats, dpi: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in formats:
        normalized = (
            "tiff" if extension.lower() in {"tif", "tiff"} else extension.lower()
        )
        path = output_dir / f"{stem}.{normalized}"
        kwargs = {"dpi": dpi}
        if normalized == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(path)
    return outputs


def main() -> None:
    args = parse_args()
    apply_nature_style()
    frame = load_metrics(args.input)
    print(f"Validated {len(frame)} rows from {args.input}")
    if args.validate_only:
        return
    for cell_type in args.cell_types:
        figure = plot_cell_type(frame, cell_type)
        outputs = save_outputs(
            figure,
            f"HiCImputeData_SZ_DO_5fold_OOF_confusion_{cell_type}",
            args.output_dir,
            args.formats,
            args.raster_dpi,
        )
        plt.close(figure)
        for output in outputs:
            print(f"Saved {output}")


if __name__ == "__main__":
    main()
