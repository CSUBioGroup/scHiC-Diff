#!/usr/bin/env python3
"""Plot five-fold cell-wise OOF HiCImputeData SZ/DO metric lines."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


PLOT_ROOT = Path(__file__).resolve().parent
if str(PLOT_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOT_ROOT))

from nature_plot_style import (  # noqa: E402
    FS_TICK,
    GR_DOUBLE_COL_IN,
    HERO_METHOD,
    NATURE_COLORS,
    apply_nature_style,
)


DEFAULT_INPUT = (
    PLOT_ROOT / "1_HiCImputedData/HiCImputeData_SZ_DO_5fold_OOF_metrics.tsv"
)
DEFAULT_OUTPUT_DIR = PLOT_ROOT / "1_HiCImputedData/figures/sz_do_metric_lines"
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
LINE_METRICS = {
    "f1": ("f1_sz", "SZ F1 score"),
    "precision": ("precision_sz", "SZ precision"),
    "recall": ("recall_sz", "SZ recall"),
    "specificity": ("specificity_do", "DO recall"),
    "accuracy": ("accuracy", "Accuracy"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=tuple(LINE_METRICS),
        default=("f1", "precision", "recall", "specificity"),
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
        *[column for column, _ in LINE_METRICS.values()],
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
    frame = frame.copy()
    frame["method"] = frame["method"].replace(METHOD_RENAME)
    return frame


def draw_line_metric_grid(
    container,
    panels,
    frame: pd.DataFrame,
    metrics,
    show_xlabel: bool = True,
    hide_zero_ytick_label: bool = False,
):
    x = np.arange(len(DEPTHS)) * 0.72
    handles = []

    for row_index, metric in enumerate(metrics):
        column, label = LINE_METRICS[metric]
        for column_index, cell_type in enumerate(CELL_TYPES):
            axis = container.add_subplot(panels[row_index, column_index])
            axis.set_facecolor("#F6F6F6")
            indexed = frame[frame["ctype"] == cell_type].set_index(
                ["method", "cdepth"]
            )
            for method in METHODS:
                values = indexed.loc[method].reindex(DEPTHS)[column].to_numpy(dtype=float)
                is_hero = method == HERO_METHOD
                line, = axis.plot(
                    x,
                    values,
                    marker="o",
                    color=NATURE_COLORS[method],
                    lw=2.0 if is_hero else 1.1,
                    markersize=4.5 if is_hero else 3.0,
                    markeredgecolor="white" if is_hero else "none",
                    markeredgewidth=0.6 if is_hero else 0,
                    zorder=5 if is_hero else 2,
                    label=method,
                )
                if row_index == 0 and column_index == 0:
                    handles.append(line)
            axis.set_ylim(0, 1.05)
            axis.set_yticks(np.arange(0, 1.01, 0.2))
            for ytick in axis.get_yticks():
                axis.axhline(ytick, color="#DBDBDB", ls="--", lw=0.5, zorder=0)
            axis.set_xticks(x, [depth.upper() for depth in DEPTHS])
            if row_index < len(metrics) - 1:
                axis.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    labelbottom=False,
                )
            axis.margins(x=0.06)
            if column_index == 0:
                axis.set_ylabel(label, fontweight="bold")
                if hide_zero_ytick_label:
                    axis.set_yticklabels(
                        [
                            "" if np.isclose(value, 0) else f"{value:.1f}"
                            for value in axis.get_yticks()
                        ]
                    )
            else:
                axis.set_yticklabels([])
            if row_index == 0:
                axis.set_title(f"{cell_type} cell type", pad=3)
            if show_xlabel and row_index == len(metrics) - 1 and column_index == 1:
                axis.set_xlabel("Sequencing depth")

    return handles


def add_line_legend(figure, outer, handles) -> None:
    legend_axis = figure.add_subplot(outer[0, 1])
    legend_axis.axis("off")
    legend_axis.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="center left",
        fontsize=FS_TICK,
        handlelength=1.4,
        handletextpad=0.5,
        labelspacing=0.55,
        borderaxespad=0.0,
    )
def plot_lines(frame: pd.DataFrame, metrics: tuple[str, ...]):
    figure = plt.figure(figsize=(GR_DOUBLE_COL_IN, 1.15 + 1.1 * len(metrics)))
    outer = GridSpec(1, 2, width_ratios=[1, 0.17], wspace=0.04)
    panels = outer[0, 0].subgridspec(len(metrics), 3, wspace=0.16, hspace=0.3)
    handles = draw_line_metric_grid(figure, panels, frame, metrics)

    add_line_legend(figure, outer, handles)
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
    metrics = tuple(args.metrics)
    figure = plot_lines(frame, metrics)
    metric_names = "_".join(metric.upper() for metric in metrics)
    outputs = save_outputs(
        figure,
        f"HiCImputeData_SZ_DO_5fold_OOF_{metric_names}_lines",
        args.output_dir,
        args.formats,
        args.raster_dpi,
    )
    plt.close(figure)
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
