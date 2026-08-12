#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce the two held-out ≥600 kb support panels including FLAMINGO."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "baseline_schicdiff",
    "flamingo",
    "scvi3d",
    "schicluster",
    "higashi_nbr0",
    "higashi_nbr5",
]
METHOD_LABELS = {
    "baseline_schicdiff": "scHiC-Diff",
    "flamingo": "FLAMINGO",
    "scvi3d": "scVI-3D",
    "schicluster": "scHiCluster",
    "higashi_nbr0": "Higashi-0",
    "higashi_nbr5": "Higashi-5",
}
COLORS = {
    "baseline_schicdiff": "#D55E00",
    "flamingo": "#0072B2",
    "scvi3d": "#009E73",
    "schicluster": "#CC79A7",
    "higashi_nbr0": "#56B4E9",
    "higashi_nbr5": "#E69F00",
}
TOP_N_VALUES = [10, 20, 50, 100, 200]
FRACTION_SOURCE_NAME = "support_fraction_600kb_data_with_flamingo.csv"
COUNT_SOURCE_NAME = "panelB_600kb_raw_supported_counts_with_flamingo.csv"
MANIFEST_NAME = "heldout_600kb_source_manifest_with_flamingo.csv"
FRACTION_STEM = "support_fraction_600kb_with_flamingo"
COUNT_STEM = "panelB_600kb_raw_supported_counts_with_flamingo"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def discover_experiment_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        script_dir = Path(__file__).resolve().parent
        candidates = [
            script_dir.parent,
            script_dir.parent / "2_callLoop_apa",
            script_dir.parent.parent / "2_callLoop_apa",
            Path.cwd().resolve(),
        ]
    marker = Path(
        "4_test_corrected_benchmark/results_diagnostics/"
        "heldout_raw_support_sensitivity_with_flamingo/topn_summary.csv"
    )
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate
    raise FileNotFoundError(f"cannot locate FLAMINGO support results; checked {[str(path) for path in candidates]}")


def _validate_method_set(frame: pd.DataFrame, label: str) -> None:
    observed = set(frame["method"].astype(str))
    expected = set(METHOD_ORDER)
    if observed != expected:
        raise ValueError(
            f"{label} method set mismatch: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def build_sources(experiment_root: Path, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_root = (
        experiment_root
        / "4_test_corrected_benchmark/results_diagnostics/"
        "heldout_raw_support_sensitivity_with_flamingo"
    )
    topn_path = source_root / "topn_summary.csv"
    count_path = source_root / "panelB_600kb_raw_supported_counts.csv"
    for required in (topn_path, count_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    topn = pd.read_csv(topn_path)
    topn = topn.loc[topn["min_distance_bins"].astype(int).eq(30)].copy()
    columns = [
        "method",
        "method_name",
        "min_distance_bins",
        "branch_label",
        "requested_n",
        "actual_n_mean",
        "actual_n_std",
        "eligible_n_mean",
        "eligible_n_std",
        "supported_count_mean",
        "supported_count_std",
        "supported_fraction_mean",
        "supported_fraction_std",
        "median_center_oe_mean",
        "median_center_oe_std",
    ]
    missing = sorted(set(columns).difference(topn.columns))
    if missing:
        raise ValueError(f"topn_summary.csv is missing columns: {missing}")
    topn = topn[columns]
    counts = pd.read_csv(count_path)
    _validate_method_set(topn, "Top-N source")
    _validate_method_set(counts, "all-eligible count source")
    if set(topn["requested_n"].astype(int)) != set(TOP_N_VALUES):
        raise ValueError("Top-N source must contain 10/20/50/100/200")
    if not topn.groupby("method")["requested_n"].nunique().eq(len(TOP_N_VALUES)).all():
        raise ValueError("every method must contain all five Top-N prefixes")
    if not counts["n_splits"].astype(int).eq(3).all():
        raise ValueError("all count summaries must use three held-out splits")
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    topn["method_name"] = topn["method"].map(METHOD_LABELS)
    counts["method_name"] = counts["method"].map(METHOD_LABELS)
    topn["_order"] = topn["method"].map(order)
    counts["_order"] = counts["method"].map(order)
    topn = topn.sort_values(["_order", "requested_n"]).drop(columns="_order").reset_index(drop=True)
    counts = counts.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    topn.to_csv(data_dir / FRACTION_SOURCE_NAME, index=False)
    counts.to_csv(data_dir / COUNT_SOURCE_NAME, index=False)
    pd.DataFrame(
        [
            {"role": "topn_summary", "path": str(topn_path.resolve()), "sha256": sha256_file(topn_path)},
            {"role": "all_eligible_summary", "path": str(count_path.resolve()), "sha256": sha256_file(count_path)},
        ]
    ).to_csv(data_dir / MANIFEST_NAME, index=False)
    return topn, counts


def load_sources(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    topn = pd.read_csv(data_dir / FRACTION_SOURCE_NAME)
    counts = pd.read_csv(data_dir / COUNT_SOURCE_NAME)
    _validate_method_set(topn, "frozen Top-N source")
    _validate_method_set(counts, "frozen all-eligible source")
    return topn, counts


def _save_all(figure: plt.Figure, output_dir: Path, stem: str, dpi: int) -> dict[str, Path]:
    outputs = {extension: output_dir / f"{stem}.{extension}" for extension in ("png", "pdf", "svg")}
    figure.savefig(outputs["png"], dpi=dpi, facecolor="white")
    figure.savefig(outputs["pdf"], facecolor="white")
    figure.savefig(outputs["svg"], facecolor="white")
    return outputs


def draw_fraction_axis(axis: plt.Axes, topn: pd.DataFrame) -> None:
    """Draw the support curve using filled-circle lines and mean ± SD ribbons."""
    _validate_method_set(topn, "support-fraction plotting data")
    positions = np.arange(len(TOP_N_VALUES), dtype=float)
    for method in METHOD_ORDER:
        selected = topn.loc[topn["method"].eq(method)].sort_values("requested_n")
        if selected["requested_n"].astype(int).tolist() != TOP_N_VALUES:
            raise ValueError(f"support-fraction data lacks ordered Top-N values for {method}")
        means = selected["supported_fraction_mean"].to_numpy(dtype=float)
        deviations = selected["supported_fraction_std"].fillna(0).to_numpy(dtype=float)
        lower = np.clip(means - deviations, 0.0, 1.0)
        upper = np.clip(means + deviations, 0.0, 1.0)
        axis.fill_between(
            positions,
            lower,
            upper,
            color=COLORS[method],
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            positions,
            means,
            color=COLORS[method],
            linestyle="-",
            linewidth=1.8 if method == "baseline_schicdiff" else 1.2,
            marker="o",
            markersize=5.2,
            markerfacecolor=COLORS[method],
            markeredgecolor=COLORS[method],
            markeredgewidth=0.6,
            label=METHOD_LABELS[method],
            zorder=3 if method == "baseline_schicdiff" else 2,
        )

    axis.set_xticks(positions, [str(value) for value in TOP_N_VALUES])
    axis.set_xlim(-0.22, len(TOP_N_VALUES) - 0.78)
    axis.set_ylim(0, 0.95)
    axis.set_xlabel("Rank prefix N")
    axis.set_ylabel("Held-out raw-supported fraction")
    axis.set_title("Held-out raw support among ≥600 kb loops", pad=8)
    axis.grid(True, which="major", axis="both", color="#C9CDD2", linewidth=0.55, alpha=0.55)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#20252B")
        spine.set_linewidth(0.7)
    axis.tick_params(direction="out", width=0.7, length=3.0, color="#20252B")
    axis.legend(loc="upper right", ncol=2, frameon=False, handlelength=2.0)


def plot_fraction(topn: pd.DataFrame, output_dir: Path, dpi: int) -> dict[str, Path]:
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        figure, axis = plt.subplots(figsize=(6.85, 3.75))
        draw_fraction_axis(axis, topn)
        figure.text(
            0.105,
            0.025,
            "Mean ± SD across three mutually exclusive 238/238-cell splits.",
            color="#5C6570",
            fontsize=7.5,
        )
        figure.subplots_adjust(left=0.105, right=0.985, top=0.88, bottom=0.19)
        outputs = _save_all(figure, output_dir, FRACTION_STEM, dpi)
        plt.close(figure)
    return outputs


def plot_counts(counts: pd.DataFrame, output_dir: Path, dpi: int) -> dict[str, Path]:
    x = np.arange(len(counts), dtype=float)
    supported = counts["supported_mean"].to_numpy(dtype=float)
    deviations = counts["supported_sd"].fillna(0).to_numpy(dtype=float)
    eligible = counts["eligible_mean"].to_numpy(dtype=float)
    colors = ["#176B87" if method == "baseline_schicdiff" else "#B7BDC5" for method in counts["method"]]
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        figure, axis = plt.subplots(figsize=(6.85, 3.85))
        bars = axis.bar(x, supported, width=0.66, color=colors, edgecolor="none", zorder=3)
        axis.errorbar(
            x,
            supported,
            yerr=deviations,
            fmt="none",
            ecolor="#20252B",
            elinewidth=0.8,
            capsize=2.8,
            capthick=0.8,
            zorder=4,
        )
        axis.set_ylim(0, float(np.max(supported + deviations)) + 9)
        for index, (bar, mean, deviation, eligible_mean) in enumerate(
            zip(bars, supported, deviations, eligible)
        ):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                mean + deviation + 0.9,
                f"{mean:.1f}\nEligible {eligible_mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color="#5C6570",
            )
        axis.set_xticks(x, [METHOD_LABELS[method] for method in counts["method"]])
        axis.set_ylabel("Held-out raw-supported loops")
        axis.set_title("Independent raw support among all eligible ≥600 kb loops", loc="left")
        axis.grid(axis="y", color="#D9DDE2", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        figure.text(
            0.105,
            0.025,
            "Bars show mean supported loops; labels also report each method's native eligible total. Error bars are SD across three splits.",
            color="#5C6570",
            fontsize=6.7,
        )
        figure.subplots_adjust(left=0.105, right=0.985, top=0.86, bottom=0.20)
        outputs = _save_all(figure, output_dir, COUNT_STEM, dpi)
        plt.close(figure)
    return outputs


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--data-dir", type=Path, default=script_dir)
    parser.add_argument("--out-dir", type=Path, default=script_dir)
    parser.add_argument("--rebuild-source", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dpi < 300:
        raise ValueError("--dpi must be at least 300")
    data_dir = args.data_dir.resolve()
    if args.rebuild_source or not (data_dir / FRACTION_SOURCE_NAME).is_file():
        root = discover_experiment_root(args.experiment_root)
        topn, counts = build_sources(root, data_dir)
        print(f"source: rebuilt ({data_dir})")
    else:
        topn, counts = load_sources(data_dir)
        print(f"source: frozen ({data_dir})")
    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, outputs in [
        ("supported_fraction", plot_fraction(topn, output_dir, args.dpi)),
        ("supported_counts", plot_counts(counts, output_dir, args.dpi)),
    ]:
        print(name)
        for extension, path in outputs.items():
            print(f"  {extension}: {path}")


if __name__ == "__main__":
    main()
