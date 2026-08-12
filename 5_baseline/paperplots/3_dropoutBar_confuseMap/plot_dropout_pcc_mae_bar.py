#!/usr/bin/env python3
"""Plot Nature-style dropout PCC/MAE/SCC bars for either benchmark."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from nature_plot_style import (
    FS_ANNOT,
    FS_TICK,
    GR_DOUBLE_COL_IN,
    HERO_EDGE,
    HERO_METHOD,
    NATURE_COLORS,
    apply_nature_style,
    grouped_bars,
    soft_grid,
)


PLOT_ROOT = Path(__file__).resolve().parent
PAPERPLOTS_DIR = PLOT_ROOT.parent

DATASET_CONFIGS = {
    "hicimpute": {
        "label": "HiCImputeData",
        "metrics_path": PAPERPLOTS_DIR
        / "1_pccAndMae_all/1_HiCImputedData/HiCImputeData_PCC_MAE_SCC_metrics.csv",
        "output_dir": PLOT_ROOT / "1_HiCImputedData/figures/dropout_bar",
    },
    "flamingo": {
        "label": "FLAMINGOData",
        "metrics_path": PAPERPLOTS_DIR
        / "1_pccAndMae_all/2_FLAMINGOData/FLAMINGOData_PCC_MAE_SCC_metrics.csv",
        "output_dir": PLOT_ROOT / "2_FLAMINGOData/figures/dropout_bar",
    },
}

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
METRIC_COLUMNS = {
    "pcc": ("pcc_held_mean", "pcc_held_std", "PCC", "up"),
    "mae": ("mae_held_mean", "mae_held_std", "MAE", "down"),
    "scc": ("scc_held_mean", "scc_held_std", "SCC", "up"),
}

W_CONDITIONS = (
    ("0.5", "v3_hybrid_W0p5_500cells_level0"),
    ("0.6", "v3_hybrid_W0p6_500cells_level0"),
    ("0.7", "v3_hybrid_W0p7_500cells_level0"),
    ("0.8", "v3_hybrid_W0p8_500cells_level0"),
    ("0.9", "v3_hybrid_W0p9_500cells_level0"),
)
P_CONDITIONS = (
    ("0.5%", "v3_hybrid_W0p7_500cells_level0"),
    ("1%", "v3_hybrid_W0p7_500cells_level0_r0p01"),
    ("5%", "v3_hybrid_W0p7_500cells_level0_r0p05"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot heldout/dropout PCC, MAE, and SCC for one benchmark."
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_CONFIGS), required=True)
    parser.add_argument(
        "--input",
        type=Path,
        help="Override the authoritative metric CSV selected by --dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the dataset-specific figures/dropout_bar directory.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=tuple(METRIC_COLUMNS),
        default=("pcc", "mae"),
    )
    parser.add_argument("--formats", nargs="+", default=("pdf", "png"))
    parser.add_argument("--raster-dpi", type=int, default=600)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _check_common_columns(df: pd.DataFrame, dataset_columns: set[str]) -> None:
    required = {
        "method",
        "transform",
        *dataset_columns,
        *[column for spec in METRIC_COLUMNS.values() for column in spec[:2]],
    }
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")


def _check_metric_keys(
    df: pd.DataFrame,
    key_column: str,
    expected: set[tuple[str, str]],
) -> None:
    keys = list(zip(df["method"], df[key_column]))
    if pd.Series(keys).duplicated().any():
        raise ValueError(f"Duplicate method/{key_column} rows in metric table")
    actual = set(keys)
    if actual != expected:
        raise ValueError(
            f"Metric-key mismatch; missing={sorted(expected - actual)[:10]}, "
            f"extra={sorted(actual - expected)[:10]}"
        )


def load_metrics(path: Path, dataset: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    if dataset == "hicimpute":
        _check_common_columns(df, {"data_name", "ctype", "cdepth"})
        expected = {
            (method, f"K562_{ctype}_{depth}")
            for method in CANONICAL_METHODS
            for ctype in CELL_TYPES
            for depth in DEPTHS
        }
        _check_metric_keys(df, "data_name", expected)
        expected_transform = "raw"
    else:
        _check_common_columns(df, {"dataset"})
        datasets = {condition[1] for condition in W_CONDITIONS + P_CONDITIONS}
        expected = {
            (method, condition)
            for method in CANONICAL_METHODS
            for condition in datasets
        }
        _check_metric_keys(df, "dataset", expected)
        expected_transform = "raw"

    transforms = set(df["transform"])
    if transforms != {expected_transform}:
        raise ValueError(f"Unexpected transforms: {sorted(transforms)}")

    df = df.copy()
    df["method"] = df["method"].replace(METHOD_RENAME)
    return df


def metric_axis(df: pd.DataFrame, metric: str):
    mean_col, std_col, _, _ = METRIC_COLUMNS[metric]
    if metric in {"pcc", "scc"}:
        low = float((df[mean_col] - df[std_col].fillna(0)).min(skipna=True))
        lower = max(-1.0, min(0.0, math.floor(low * 10) / 10))
        return (lower, 1.0), np.arange(0, 1.01, 0.2)

    high = float((df[mean_col] + df[std_col].fillna(0)).max(skipna=True))
    tick_step = 1.0 if high > 3.0 else 0.5
    upper = max(tick_step, math.ceil(high / tick_step) * tick_step)
    return (0.0, upper), np.arange(0, upper + 0.01, tick_step)


def extract_hicimpute_panel(df: pd.DataFrame, ctype: str, metric: str):
    mean_col, std_col, _, _ = METRIC_COLUMNS[metric]
    indexed = df[df["ctype"] == ctype].set_index(["method", "cdepth"])
    values = np.full((len(DEPTHS), len(METHODS)), np.nan, dtype=float)
    errors = np.full_like(values, np.nan)
    for depth_idx, depth in enumerate(DEPTHS):
        for method_idx, method in enumerate(METHODS):
            row = indexed.loc[(method, depth)]
            values[depth_idx, method_idx] = row[mean_col]
            errors[depth_idx, method_idx] = row[std_col]
    return values, errors


def extract_flamingo_panel(df: pd.DataFrame, metric: str, conditions):
    mean_col, std_col, _, _ = METRIC_COLUMNS[metric]
    datasets = [dataset for _, dataset in conditions]
    indexed = df.set_index(["method", "dataset"])
    values = np.full((len(datasets), len(METHODS)), np.nan, dtype=float)
    errors = np.full_like(values, np.nan)
    for dataset_idx, condition in enumerate(datasets):
        for method_idx, method in enumerate(METHODS):
            row = indexed.loc[(method, condition)]
            values[dataset_idx, method_idx] = row[mean_col]
            errors[dataset_idx, method_idx] = row[std_col]
    return values, errors


def annotate_missing(
    ax,
    values: np.ndarray,
    x: np.ndarray,
    bar_width: float,
    y: float,
) -> None:
    for group_idx, method_idx in np.argwhere(~np.isfinite(values)):
        offset = (method_idx - (len(METHODS) - 1) / 2) * bar_width
        ax.text(
            x[group_idx] + offset,
            y,
            "NA",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=FS_ANNOT,
            color="#686868",
            clip_on=True,
        )


def legend_handles():
    return [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=NATURE_COLORS[method],
            ec=HERO_EDGE if method == HERO_METHOD else "white",
            lw=0.5 if method == HERO_METHOD else 0.4,
        )
        for method in METHODS
    ]


def add_legend(fig, outer) -> None:
    legend_ax = fig.add_subplot(outer[0, 1])
    legend_ax.axis("off")
    legend_ax.legend(
        legend_handles(),
        METHODS,
        loc="center left",
        fontsize=FS_TICK,
        handlelength=1.1,
        handletextpad=0.5,
        labelspacing=0.5,
        borderaxespad=0.0,
    )
def draw_hicimpute_metric_grid(
    container,
    plot_gs,
    df: pd.DataFrame,
    metrics: tuple[str, ...],
    show_xlabel: bool = True,
) -> bool:
    n_rows = len(metrics)
    contains_missing = False

    for row_idx, metric in enumerate(metrics):
        _, _, label, direction = METRIC_COLUMNS[metric]
        ylim, yticks = metric_axis(df, metric)
        for col_idx, ctype in enumerate(CELL_TYPES):
            ax = container.add_subplot(plot_gs[row_idx, col_idx])
            values, errors = extract_hicimpute_panel(df, ctype, metric)
            safe_errors = np.where(np.isfinite(values), errors, 0.0)
            ax.set_ylim(*ylim)
            ax.set_yticks(yticks)
            soft_grid(ax, yticks)
            x, bar_width = grouped_bars(ax, values, METHODS, errors=safe_errors)
            if ylim[0] < 0:
                ax.axhline(0, color="#666666", lw=0.55, zorder=4)
            if (~np.isfinite(values)).any():
                contains_missing = True
                missing_y = ylim[0] + 0.02 * (ylim[1] - ylim[0])
                annotate_missing(ax, values, x, bar_width, missing_y)

            ax.set_xticklabels([depth.upper() for depth in DEPTHS])
            ax.margins(x=0.035)
            if col_idx == 0:
                arrow = "↑" if direction == "up" else "↓"
                ax.set_ylabel(f"{label} {arrow}", fontweight="bold")
            else:
                ax.set_yticklabels([])
            if row_idx == 0:
                ax.set_title(f"{ctype} cell type", pad=3)
            if show_xlabel and row_idx == n_rows - 1 and col_idx == 1:
                ax.set_xlabel("Sequencing depth")

    return contains_missing


def plot_hicimpute_metrics(df: pd.DataFrame, metrics: tuple[str, ...]):
    n_rows = len(metrics)
    fig = plt.figure(figsize=(GR_DOUBLE_COL_IN, 1.0 + 1.2 * n_rows))
    outer = GridSpec(1, 2, width_ratios=[1, 0.17], wspace=0.04)
    plot_gs = outer[0, 0].subgridspec(n_rows, 3, wspace=0.16, hspace=0.28)
    contains_missing = draw_hicimpute_metric_grid(fig, plot_gs, df, metrics)

    add_legend(fig, outer)
    if contains_missing:
        fig.text(
            0.995,
            0.005,
            "NA: correlation undefined because prediction variance is zero",
            ha="right",
            va="bottom",
            fontsize=FS_ANNOT,
            color="#686868",
        )
    return fig


def draw_flamingo_metric_grid(
    container,
    plot_gs,
    df: pd.DataFrame,
    metrics: tuple[str, ...],
    show_xlabel: bool = True,
) -> None:
    """Draw FLAMINGOData metric panels into a caller-provided grid."""
    n_rows = len(metrics)
    panel_specs = (
        (W_CONDITIONS, "W sweep (P = 0.5%)", "W"),
        (P_CONDITIONS, "P sweep (W = 0.7)", "P"),
    )

    for row_idx, metric in enumerate(metrics):
        _, _, label, direction = METRIC_COLUMNS[metric]
        ylim, yticks = metric_axis(df, metric)
        for col_idx, (conditions, title, xlabel) in enumerate(panel_specs):
            ax = container.add_subplot(plot_gs[row_idx, col_idx])
            values, errors = extract_flamingo_panel(df, metric, conditions)
            ax.set_ylim(*ylim)
            ax.set_yticks(yticks)
            soft_grid(ax, yticks)
            grouped_bars(ax, values, METHODS, errors=errors)
            ax.set_xticklabels([condition[0] for condition in conditions])
            ax.margins(x=0.035)
            if col_idx == 0:
                arrow = "↑" if direction == "up" else "↓"
                ax.set_ylabel(f"{label} {arrow}", fontweight="bold")
            else:
                ax.set_yticklabels([])
            if row_idx == 0:
                ax.set_title(title, pad=3)
            if show_xlabel and row_idx == n_rows - 1:
                ax.set_xlabel(xlabel)


def plot_flamingo_metrics(df: pd.DataFrame, metrics: tuple[str, ...]):
    n_rows = len(metrics)
    fig = plt.figure(figsize=(GR_DOUBLE_COL_IN, 1.0 + 1.2 * n_rows))
    outer = GridSpec(1, 2, width_ratios=[1, 0.17], wspace=0.04)
    plot_gs = outer[0, 0].subgridspec(
        n_rows,
        2,
        width_ratios=[5, 3],
        wspace=0.18,
        hspace=0.28,
    )
    draw_flamingo_metric_grid(fig, plot_gs, df, metrics)

    add_legend(fig, outer)
    return fig


def save_outputs(fig, stem: str, output_dir: Path, formats, dpi: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in formats:
        normalized = "tiff" if ext.lower() in {"tif", "tiff"} else ext.lower()
        path = output_dir / f"{stem}.{normalized}"
        kwargs = {"dpi": dpi}
        if normalized == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(path)
    return outputs


def main() -> None:
    args = parse_args()
    config = DATASET_CONFIGS[args.dataset]
    input_path = args.input or config["metrics_path"]
    output_dir = args.output_dir or config["output_dir"]

    apply_nature_style()
    df = load_metrics(input_path, args.dataset)
    print(f"Validated {len(df)} rows from {input_path}")
    if args.validate_only:
        return

    metrics = tuple(args.metrics)
    plotter = (
        plot_hicimpute_metrics
        if args.dataset == "hicimpute"
        else plot_flamingo_metrics
    )
    fig = plotter(df, metrics)
    metric_names = "_".join(metric.upper() for metric in metrics)
    outputs = save_outputs(
        fig,
        f"{config['label']}_dropout_{metric_names}_bar",
        output_dir,
        args.formats,
        args.raster_dpi,
    )
    plt.close(fig)
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
