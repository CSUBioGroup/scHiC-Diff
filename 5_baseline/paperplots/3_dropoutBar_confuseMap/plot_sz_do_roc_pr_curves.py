#!/usr/bin/env python3
"""Plot HiCImputeData SZ/DO ROC, precision-recall, and AUC supplementary figures."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_ROOT = Path(__file__).resolve().parent
if str(PLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOT_ROOT))

from nature_plot_style import (  # noqa: E402
    C_INK,
    FS_TICK,
    GR_DOUBLE_COL_IN,
    HERO_METHOD,
    NATURE_COLORS,
    apply_nature_style,
)


INPUT_ROOT = PLOT_ROOT / "1_HiCImputedData"
DEFAULT_SUMMARY = INPUT_ROOT / "HiCImputeData_SZ_DO_5fold_OOF_ROC_PR_AUC.tsv"
DEFAULT_POINTS = INPUT_ROOT / "HiCImputeData_SZ_DO_5fold_OOF_curve_plot_points.tsv"
DEFAULT_OUTPUT_DIR = INPUT_ROOT / "figures/sz_do_roc_pr"

CANONICAL_METHODS = (
    "scHiCluster",
    "HiCImpute",
    "Higashi_nbr0",
    "Higashi_nbr5",
    "scVI-3D",
    "Tensor-FLAMINGO",
    "scHiC-Diff",
)
METHOD_RENAME = {
    "Higashi_nbr0": "Higashi-nbr0",
    "Higashi_nbr5": "Higashi-nbr5",
}
METHODS = tuple(METHOD_RENAME.get(method, method) for method in CANONICAL_METHODS)
CELL_TYPES = ("T1", "T2", "T3")
DEPTHS = ("1k", "2k", "4k", "7k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formats", nargs="+", default=("pdf", "png"))
    parser.add_argument("--raster-dpi", type=int, default=600)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _expected_keys() -> set[tuple[str, str, str]]:
    return {
        (method, ctype, depth)
        for method in CANONICAL_METHODS
        for ctype in CELL_TYPES
        for depth in DEPTHS
    }


def load_inputs(summary_path: Path, points_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not points_path.exists():
        raise FileNotFoundError(points_path)
    summary = pd.read_csv(summary_path, sep="\t")
    points = pd.read_csv(points_path, sep="\t")
    summary_required = {
        "method",
        "ctype",
        "cdepth",
        "data_name",
        "positive_class",
        "evaluation_scope",
        "curve_protocol",
        "roc_auc",
        "pr_auc_linear_interpolation",
        "average_precision",
        "candidate_count",
        "true_sz_count",
        "true_do_count",
    }
    points_required = {
        "method",
        "ctype",
        "cdepth",
        "data_name",
        "point_index",
        "fpr_do",
        "tpr_sz",
        "precision_sz",
        "recall_sz",
        "full_scan_point_count",
    }
    missing_summary = sorted(summary_required - set(summary.columns))
    missing_points = sorted(points_required - set(points.columns))
    if missing_summary or missing_points:
        raise ValueError(
            f"Missing columns: summary={missing_summary}, points={missing_points}"
        )
    expected = _expected_keys()
    summary_keys = list(zip(summary["method"], summary["ctype"], summary["cdepth"]))
    if pd.Series(summary_keys).duplicated().any() or set(summary_keys) != expected:
        raise ValueError("Expected exactly 7 methods x 3 cell types x 4 depths")
    if set(summary["positive_class"]) != {"SZ"}:
        raise ValueError("Expected SZ as the positive class")
    if set(summary["evaluation_scope"]) != {"observed_zero"}:
        raise ValueError("Expected observed-zero evaluation")
    if set(summary["curve_protocol"]) != {
        "5fold_cellwise_OOF_all_unique_prediction_thresholds"
    }:
        raise ValueError("Unexpected ROC/PR curve protocol")
    point_keys = set(zip(points["method"], points["ctype"], points["cdepth"]))
    if point_keys != expected:
        raise ValueError("Plot point conditions do not match the AUC summary")
    if not np.isfinite(
        summary[
            ["roc_auc", "pr_auc_linear_interpolation", "average_precision"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("AUC summary contains non-finite values")
    if not np.isfinite(
        points[["fpr_do", "tpr_sz", "precision_sz", "recall_sz"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Curve points contain non-finite values")
    if (
        (points[["fpr_do", "tpr_sz", "precision_sz", "recall_sz"]] < 0).any().any()
        or (points[["fpr_do", "tpr_sz", "precision_sz", "recall_sz"]] > 1).any().any()
    ):
        raise ValueError("Curve points fall outside [0, 1]")

    summary = summary.copy()
    points = points.copy()
    summary["method"] = summary["method"].replace(METHOD_RENAME)
    points["method"] = points["method"].replace(METHOD_RENAME)
    return summary, points


def _style_curve(axis, column_index: int, row_index: int, kind: str) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    ticks = np.array((0.0, 0.5, 1.0))
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_facecolor("#F8F8F8")
    axis.grid(axis="both", color="#E0E0E0", lw=0.45, zorder=0)
    if column_index:
        axis.set_yticklabels([])
    elif kind == "roc":
        axis.set_ylabel("True positive rate (SZ)", labelpad=2)
    else:
        axis.set_ylabel("Precision (SZ)", labelpad=2)
    if row_index < len(CELL_TYPES) - 1:
        axis.set_xticklabels([])
    else:
        # Adjacent panels share a boundary. Suppressing duplicate endpoint
        # labels makes the compact four-column layout readable at print size.
        labels = ["0", "0.5", "1"]
        if column_index:
            labels[0] = ""
        if column_index < len(DEPTHS) - 1:
            labels[-1] = ""
        axis.set_xticklabels(labels)
    axis.tick_params(axis="both", pad=1.5)


def plot_curve_grid(summary: pd.DataFrame, points: pd.DataFrame, kind: str):
    if kind not in {"roc", "pr"}:
        raise ValueError(kind)
    figure = plt.figure(figsize=(GR_DOUBLE_COL_IN, 5.50))
    outer = figure.add_gridspec(
        3,
        1,
        height_ratios=(1.0, 0.06, 0.11),
        hspace=0.01,
        left=0.125,
        right=0.99,
        top=0.965,
        bottom=0.035,
    )
    panels = outer[0, 0].subgridspec(3, 4, wspace=0.14, hspace=0.18)
    handles = []
    row_axes = []

    for row_index, ctype in enumerate(CELL_TYPES):
        for column_index, depth in enumerate(DEPTHS):
            axis = figure.add_subplot(panels[row_index, column_index])
            _style_curve(axis, column_index, row_index, kind)
            if column_index == 0:
                row_axes.append(axis)
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
                    lw=0.55,
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
                    lw=0.55,
                    zorder=1,
                )
                x_column, y_column = "recall_sz", "precision_sz"

            for method in METHODS:
                curve = condition[condition["method"] == method].sort_values(
                    "point_index"
                )
                is_hero = method == HERO_METHOD
                line, = axis.plot(
                    curve[x_column],
                    curve[y_column],
                    color=NATURE_COLORS[method],
                    lw=1.7 if is_hero else 0.8,
                    zorder=5 if is_hero else 3,
                    label=method,
                )
                if row_index == 0 and column_index == 0:
                    handles.append(line)

            if row_index == 0:
                axis.set_title(depth.upper(), pad=3)
    for ctype, axis in zip(CELL_TYPES, row_axes):
        position = axis.get_position()
        figure.text(
            0.024,
            (position.y0 + position.y1) / 2,
            ctype,
            rotation=90,
            ha="center",
            va="center",
            fontsize=FS_TICK,
            fontweight="bold",
            color=C_INK,
        )

    xlabel_axis = figure.add_subplot(outer[1, 0])
    xlabel_axis.axis("off")
    xlabel_axis.text(
        0.5,
        0.20,
        "False positive rate (DO)" if kind == "roc" else "Recall (SZ)",
        ha="center",
        va="bottom",
    )
    legend_axis = figure.add_subplot(outer[2, 0])
    legend_axis.axis("off")
    legend_axis.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="center",
        ncol=len(handles),
        fontsize=FS_TICK,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.65,
        borderaxespad=0.0,
    )
    return figure


def plot_auc_heatmap(summary: pd.DataFrame):
    condition_keys = [f"{ctype}\n{depth.upper()}" for ctype in CELL_TYPES for depth in DEPTHS]
    figure = plt.figure(figsize=(GR_DOUBLE_COL_IN, 3.45))
    layout = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.045),
        wspace=0.24,
        left=0.19,
        right=0.94,
        top=0.91,
        bottom=0.19,
    )
    images = []
    for panel_index, (column, title) in enumerate(
        (
            ("roc_auc", "ROC-AUC"),
            ("average_precision", "Average precision (AP)"),
        )
    ):
        axis = figure.add_subplot(layout[0, panel_index])
        values = np.full((len(METHODS), len(condition_keys)), np.nan, dtype=float)
        for method_index, method in enumerate(METHODS):
            method_frame = summary[summary["method"] == method]
            index = method_frame.set_index(["ctype", "cdepth"])
            for condition_index, ctype in enumerate(CELL_TYPES):
                for depth_index, depth in enumerate(DEPTHS):
                    values[method_index, condition_index * len(DEPTHS) + depth_index] = index.loc[
                        (ctype, depth), column
                    ]
        image = axis.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
        images.append(image)
        axis.set_title(title, pad=4)
        axis.set_xticks(np.arange(len(condition_keys)), condition_keys)
        axis.set_yticks(np.arange(len(METHODS)))
        if panel_index == 0:
            axis.set_yticklabels(METHODS)
        else:
            axis.set_yticklabels([])
        axis.tick_params(axis="x", length=0)
        axis.tick_params(axis="y", length=0, pad=3)
        for label in axis.get_yticklabels():
            if label.get_text() == HERO_METHOD:
                label.set_fontweight("bold")
                label.set_color(NATURE_COLORS[HERO_METHOD])
        for row_index, row in enumerate(values):
            for column_index, value in enumerate(row):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="white" if value >= 0.62 else C_INK,
                )
        for boundary in (3.5, 7.5):
            axis.axvline(boundary, color="white", lw=1.2)
        for spine in axis.spines.values():
            spine.set_visible(False)

    color_axis = figure.add_subplot(layout[0, 2])
    colorbar = figure.colorbar(images[0], cax=color_axis)
    colorbar.set_ticks(np.arange(0.0, 1.01, 0.2))
    colorbar.outline.set_linewidth(0.4)
    return figure


def save_outputs(figure, stem: str, output_dir: Path, formats, dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
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
    summary, points = load_inputs(args.summary, args.points)
    print(f"Validated {len(summary)} AUC rows and {len(points)} curve plot rows")
    if args.validate_only:
        return

    figures = (
        (plot_curve_grid(summary, points, "roc"), "HiCImputeData_SZ_DO_5fold_OOF_ROC_curves"),
        (plot_curve_grid(summary, points, "pr"), "HiCImputeData_SZ_DO_5fold_OOF_PR_curves"),
        (plot_auc_heatmap(summary), "HiCImputeData_SZ_DO_5fold_OOF_ROC_PR_AUC"),
    )
    for figure, stem in figures:
        outputs = save_outputs(
            figure, stem, args.output_dir, args.formats, args.raster_dpi
        )
        plt.close(figure)
        for output in outputs:
            print(f"Saved {output}")


if __name__ == "__main__":
    main()
