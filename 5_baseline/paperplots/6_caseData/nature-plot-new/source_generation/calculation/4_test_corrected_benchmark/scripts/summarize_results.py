#!/usr/bin/env python3
"""Summarize repeated loop and corrected Map2 results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(BENCHMARK_DIR / ".mplconfig")
os.environ["XDG_CACHE_HOME"] = str(BENCHMARK_DIR / ".cache")
(BENCHMARK_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)
(BENCHMARK_DIR / ".cache").mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_benchmark import load_resolved_config


def summarize_loops(runs: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "group", "cell_count", "seed", "loop_count", "summit_count"}
    missing = sorted(required.difference(runs.columns))
    if missing:
        raise ValueError(f"loop runs are missing columns: {missing}")
    return (
        runs.groupby(["method", "group", "cell_count"], as_index=False)
        .agg(
            repeat_count=("seed", "count"),
            loop_count_mean=("loop_count", "mean"),
            loop_count_std=("loop_count", "std"),
            loop_count_min=("loop_count", "min"),
            loop_count_max=("loop_count", "max"),
            summit_count_mean=("summit_count", "mean"),
            summit_count_std=("summit_count", "std"),
            summit_count_min=("summit_count", "min"),
            summit_count_max=("summit_count", "max"),
        )
        .sort_values(["group", "cell_count", "method"])
        .reset_index(drop=True)
    )


def summarize_map2(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "method",
        "group",
        "cell_count",
        "seed",
        "transform",
        "background",
        "ratio",
        "log2_enrichment",
        "percentile",
        "empirical_p_upper",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"Map2 metrics are missing columns: {missing}")
    return (
        metrics.groupby(
            ["method", "group", "cell_count", "transform", "background"],
            as_index=False,
        )
        .agg(
            repeat_count=("seed", "count"),
            ratio_mean=("ratio", "mean"),
            ratio_std=("ratio", "std"),
            ratio_min=("ratio", "min"),
            ratio_max=("ratio", "max"),
            log2_enrichment_mean=("log2_enrichment", "mean"),
            percentile_mean=("percentile", "mean"),
            empirical_p_upper_mean=("empirical_p_upper", "mean"),
        )
        .sort_values(["group", "cell_count", "transform", "background", "method"])
        .reset_index(drop=True)
    )


def write_summary_plots(
    loop_summary: pd.DataFrame,
    map2_summary: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for group in sorted(loop_summary["group"].unique()):
        subset = loop_summary[loop_summary["group"] == group]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for method in sorted(subset["method"].unique()):
            method_data = subset[subset["method"] == method].sort_values("cell_count")
            x = method_data["cell_count"].to_numpy()
            loop_y = method_data["loop_count_mean"].to_numpy()
            summit_y = method_data["summit_count_mean"].to_numpy()
            loop_std = method_data.get("loop_count_std", pd.Series(np.zeros(len(method_data)))).fillna(0).to_numpy()
            axes[0].errorbar(x, loop_y, yerr=loop_std, marker="o", linewidth=1.2, label=method)
            axes[1].plot(x, summit_y, marker="o", linewidth=1.2, label=method)
        for axis, ylabel in zip(axes, ["Called loops", "Summits"]):
            axis.set_xscale("log")
            axis.set_xlabel("Aggregated cells")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
        axes[0].legend(frameon=False, fontsize=8)
        figure.suptitle(f"Repeated loop calling: {group}")
        figure.tight_layout()
        path = output_dir / f"loop_counts_{group}.png"
        figure.savefig(path, dpi=220)
        plt.close(figure)
        outputs.append(path)

    if not map2_summary.empty:
        preferred_group = "earlyNeurons" if "earlyNeurons" in set(map2_summary["group"]) else map2_summary.iloc[0]["group"]
        preferred_count = int(map2_summary[map2_summary["group"] == preferred_group]["cell_count"].max())
        selected = map2_summary[
            (map2_summary["group"] == preferred_group)
            & (map2_summary["cell_count"] == preferred_count)
        ]
        for (transform, background), subset in selected.groupby(["transform", "background"]):
            subset = subset.sort_values("ratio_mean", ascending=False)
            figure, axis = plt.subplots(figsize=(8, 4.5))
            errors = subset.get("ratio_std", pd.Series(np.zeros(len(subset)))).fillna(0).to_numpy()
            axis.bar(subset["method"], subset["ratio_mean"], yerr=errors, capsize=3)
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
            axis.set_ylabel("Corrected center/background ratio")
            axis.set_title(
                f"Map2 {preferred_group}, {preferred_count} cells | {transform}, {background}"
            )
            axis.tick_params(axis="x", rotation=30)
            figure.tight_layout()
            path = output_dir / f"map2_{preferred_group}_{preferred_count}_{transform}_{background}.png"
            figure.savefig(path, dpi=220)
            plt.close(figure)
            outputs.append(path)
    return outputs


def write_summaries(config: dict) -> tuple[Path, Path]:
    loop_path = config["output_root"] / "manifests" / "loop_runs.csv"
    map2_path = config["output_root"] / "map2" / "map2_metrics.csv"
    if not loop_path.exists() or not map2_path.exists():
        raise FileNotFoundError(f"required full-run outputs are missing: {loop_path}, {map2_path}")
    output_dir = config["output_root"] / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    loop_output = output_dir / "loop_summary.csv"
    map2_output = output_dir / "map2_summary.csv"
    loop_summary = summarize_loops(pd.read_csv(loop_path))
    map2_summary = summarize_map2(pd.read_csv(map2_path))
    loop_summary.to_csv(loop_output, index=False)
    map2_summary.to_csv(map2_output, index=False)
    write_summary_plots(loop_summary, map2_summary, output_dir / "figures")
    return loop_output, map2_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_resolved_config(args.config)
    outputs = write_summaries(config)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
