#!/usr/bin/env python3
"""Held-out raw support–sensitivity analysis for ranked chromatin loops."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adapters import load_csr_npz
from map2_metrics import expected_by_distance, observed_over_expected


NATIVE_LOOP_COLUMNS = ["start1", "end1", "start2", "end2", "score"]
BENCHMARK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = BENCHMARK_DIR / "results_diagnostics/heldout_raw_support_sensitivity"
PANEL_B_METHOD_ORDER = [
    "baseline_schicdiff",
    "ramani_ckpt_ep0999",
    "scvi3d",
    "schicluster",
    "higashi_nbr0",
    "higashi_nbr5",
]


def load_native_loops(
    path: str | Path,
    *,
    resolution: int,
    n_bins: int,
) -> pd.DataFrame:
    """Load and validate the loop caller's native five-column local BEDPE."""
    path = Path(path)
    if resolution <= 0 or n_bins <= 0:
        raise ValueError("resolution and n_bins must be positive")
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep=r"\s+", header=None, comment="#")
    if frame.shape[1] != 5:
        raise ValueError(f"native loop file must contain five columns: {path}")
    frame.columns = NATIVE_LOOP_COLUMNS
    for column in NATIVE_LOOP_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    coordinates = frame[["start1", "end1", "start2", "end2"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all() or not np.isfinite(frame["score"]).all():
        raise ValueError(f"loop file contains non-finite values: {path}")
    if not np.equal(coordinates, np.floor(coordinates)).all():
        raise ValueError("loop coordinates must be integers")
    coordinates = coordinates.astype(np.int64)
    frame[["start1", "end1", "start2", "end2"]] = coordinates
    if np.any(coordinates % resolution):
        raise ValueError(f"loop coordinates must be aligned to {resolution} bp bins")
    if np.any(frame["end1"] - frame["start1"] != resolution) or np.any(
        frame["end2"] - frame["start2"] != resolution
    ):
        raise ValueError("every loop anchor must span exactly one resolution bin")
    locus_size = n_bins * resolution
    if np.any(coordinates < 0) or np.any(coordinates[:, [1, 3]] > locus_size):
        raise ValueError(f"loop coordinates fall outside the {locus_size}-bp locus")

    swap = frame["start2"].to_numpy() < frame["start1"].to_numpy()
    if swap.any():
        for first, second in [("start1", "start2"), ("end1", "end2")]:
            saved = frame.loc[swap, first].copy()
            frame.loc[swap, first] = frame.loc[swap, second].to_numpy()
            frame.loc[swap, second] = saved.to_numpy()
    frame["bin1"] = frame["start1"] // resolution
    frame["bin2"] = frame["start2"] // resolution
    frame["distance_bins"] = frame["bin2"] - frame["bin1"]
    frame["_input_order"] = np.arange(len(frame), dtype=np.int64)
    frame = (
        frame.sort_values(["score", "_input_order"], ascending=[False, True], kind="mergesort")
        .drop_duplicates(["bin1", "bin2"], keep="first")
        .sort_values("_input_order", kind="mergesort")
        .drop(columns="_input_order")
        .reset_index(drop=True)
    )
    return frame


def prepare_ranked_loops(frame: pd.DataFrame, *, min_distance_bins: int) -> pd.DataFrame:
    """Filter by distance, then rank high-to-low by caller score."""
    if min_distance_bins < 0:
        raise ValueError("min_distance_bins must be non-negative")
    eligible = frame.loc[frame["distance_bins"] >= min_distance_bins].copy()
    eligible = eligible.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    eligible["rank"] = np.arange(1, len(eligible) + 1, dtype=np.int64)
    return eligible


def complete_center_mean(matrix: np.ndarray, bin1: int, bin2: int, radius: int = 1) -> float:
    """Mean a complete square center window, returning NaN at locus edges."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    n_bins = matrix.shape[0]
    if bin1 - radius < 0 or bin2 - radius < 0:
        return float("nan")
    if bin1 + radius >= n_bins or bin2 + radius >= n_bins:
        return float("nan")
    return float(matrix[bin1 - radius : bin1 + radius + 1, bin2 - radius : bin2 + radius + 1].mean())


def _upper_tail_empirical_p(observed: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed):
        return float("nan")
    return float((1 + np.count_nonzero(null >= observed)) / (1 + null.size))


def exact_distance_statistics(oe_matrix: np.ndarray, *, bin1: int, bin2: int) -> dict[str, float | int]:
    """Score one loop against every other pixel at its exact separation."""
    oe_matrix = np.asarray(oe_matrix, dtype=float)
    if oe_matrix.ndim != 2 or oe_matrix.shape[0] != oe_matrix.shape[1]:
        raise ValueError("O/E matrix must be square")
    n_bins = oe_matrix.shape[0]
    if not (0 <= bin1 < bin2 < n_bins):
        raise ValueError(f"invalid upper-triangle loop bins: ({bin1}, {bin2})")
    distance = bin2 - bin1
    starts = np.arange(n_bins - distance, dtype=np.int64)
    exact_values = oe_matrix[starts, starts + distance]
    keep = starts != bin1
    exact_null = exact_values[keep]
    center_oe = float(oe_matrix[bin1, bin2])

    center_3x3_oe = complete_center_mean(oe_matrix, bin1, bin2)
    window_values = np.asarray(
        [complete_center_mean(oe_matrix, int(start), int(start + distance)) for start in starts],
        dtype=float,
    )
    window_null = window_values[keep]
    window_null = window_null[np.isfinite(window_null)]
    return {
        "center_oe": center_oe,
        "center_3x3_oe": center_3x3_oe,
        "null_count": int(np.isfinite(exact_null).sum()),
        "null_3x3_count": int(window_null.size),
        "empirical_p": _upper_tail_empirical_p(center_oe, exact_null),
        "empirical_p_3x3": _upper_tail_empirical_p(center_3x3_oe, window_null),
    }


def benjamini_hochberg(pvalues: np.ndarray | pd.Series) -> np.ndarray:
    """Return BH-adjusted q-values in the input order, preserving NaNs."""
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(values))
    if not valid_indices.size:
        return adjusted
    valid = values[valid_indices]
    if np.any((valid < 0) | (valid > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(valid, kind="mergesort")
    ranked = valid[order]
    scale = valid.size / np.arange(1, valid.size + 1, dtype=float)
    ranked_adjusted = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0.0, 1.0)
    restored = np.empty_like(ranked_adjusted)
    restored[order] = ranked_adjusted
    adjusted[valid_indices] = restored
    return adjusted


def score_loops_against_reference(
    loops: pd.DataFrame,
    oe_matrix: np.ndarray,
    *,
    support_oe_threshold: float = 1.0,
    support_p_threshold: float = 0.05,
) -> pd.DataFrame:
    """Attach exact-distance held-out statistics and stable branch-wide BH q-values."""
    scored = loops.copy().reset_index(drop=True)
    statistics = [
        exact_distance_statistics(oe_matrix, bin1=int(row.bin1), bin2=int(row.bin2))
        for row in scored.itertuples(index=False)
    ]
    if statistics:
        scored = pd.concat([scored, pd.DataFrame(statistics)], axis=1)
    else:
        for column in [
            "center_oe", "center_3x3_oe", "null_count", "null_3x3_count",
            "empirical_p", "empirical_p_3x3",
        ]:
            scored[column] = pd.Series(dtype=float)
    scored["q_all"] = benjamini_hochberg(scored["empirical_p"].to_numpy())
    scored["raw_supported"] = (
        (scored["center_oe"] > support_oe_threshold)
        & (scored["empirical_p"] <= support_p_threshold)
    )
    return scored


def summarize_prefixes(
    scored: pd.DataFrame,
    *,
    top_n_values: list[int] | tuple[int, ...] = (10, 20, 50, 100, 200),
    q_threshold: float = 0.10,
) -> pd.DataFrame:
    """Summarize pre-ranked held-out support at requested Top-N prefixes."""
    if not 0 <= q_threshold <= 1:
        raise ValueError("q_threshold must lie in [0, 1]")
    ranked = scored.sort_values("rank", kind="mergesort").reset_index(drop=True)
    rows: list[dict[str, float | int]] = []
    for requested in top_n_values:
        if int(requested) <= 0:
            raise ValueError("Top-N values must be positive")
        prefix = ranked.head(int(requested)).copy()
        actual = len(prefix)
        q_prefix = benjamini_hochberg(prefix["empirical_p"].to_numpy())
        supported_count = int(prefix["raw_supported"].sum()) if actual else 0
        rows.append(
            {
                "requested_n": int(requested),
                "actual_n": actual,
                "eligible_n": len(ranked),
                "supported_count": supported_count,
                "supported_fraction": float(supported_count / actual) if actual else np.nan,
                "median_center_oe": float(prefix["center_oe"].median()) if actual else np.nan,
                "median_center_3x3_oe": (
                    float(prefix["center_3x3_oe"].median()) if actual else np.nan
                ),
                "median_empirical_p": (
                    float(prefix["empirical_p"].median()) if actual else np.nan
                ),
                "q_all_le_0_10_count": (
                    int((prefix["q_all"] <= q_threshold).sum()) if actual else 0
                ),
                "q_prefix_le_0_10_count": (
                    int(np.count_nonzero(q_prefix <= q_threshold)) if actual else 0
                ),
                "median_q_all": float(prefix["q_all"].median()) if actual else np.nan,
                "median_q_prefix": (
                    float(np.nanmedian(q_prefix)) if np.isfinite(q_prefix).any() else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def support_operating_point(
    scored: pd.DataFrame,
    *,
    target_fraction: float = 0.80,
    min_prefix: int = 10,
) -> dict[str, float | int]:
    """Return the largest ranked prefix meeting a cumulative support target."""
    if not 0 < target_fraction <= 1:
        raise ValueError("target_fraction must lie in (0, 1]")
    if min_prefix <= 0:
        raise ValueError("min_prefix must be positive")
    ranked = scored.sort_values("rank", kind="mergesort").reset_index(drop=True)
    eligible_n = len(ranked)
    if eligible_n < min_prefix:
        return {
            "support80_k": 0,
            "support80_supported_count": 0,
            "support80_fraction": np.nan,
            "eligible_n": eligible_n,
        }
    cumulative = ranked["raw_supported"].astype(int).cumsum().to_numpy()
    prefix_sizes = np.arange(1, eligible_n + 1, dtype=np.int64)
    fractions = cumulative / prefix_sizes
    qualifying = np.flatnonzero(
        (prefix_sizes >= min_prefix) & (fractions >= target_fraction)
    )
    if not qualifying.size:
        return {
            "support80_k": 0,
            "support80_supported_count": 0,
            "support80_fraction": np.nan,
            "eligible_n": eligible_n,
        }
    index = int(qualifying[-1])
    return {
        "support80_k": int(prefix_sizes[index]),
        "support80_supported_count": int(cumulative[index]),
        "support80_fraction": float(fractions[index]),
        "eligible_n": eligible_n,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_oe(path: str | Path, *, n_bins: int) -> np.ndarray:
    """Load one upper-triangle CSR vector and return its symmetric diagonal O/E."""
    source = load_csr_npz(path)
    expected_features = n_bins * (n_bins + 1) // 2
    if source.shape != (1, expected_features):
        raise ValueError(
            f"raw reference must have shape (1, {expected_features}), got {source.shape}: {path}"
        )
    vector = source.toarray().reshape(-1)
    rows, columns = np.triu_indices(n_bins)
    matrix = np.zeros((n_bins, n_bins), dtype=float)
    matrix[rows, columns] = vector
    matrix[columns, rows] = vector
    return observed_over_expected(matrix, expected_by_distance(matrix))


def _aggregate_topn(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "method_name", "min_distance_bins", "branch_label", "requested_n"]
    metrics = [
        "actual_n", "eligible_n", "supported_count", "supported_fraction",
        "median_center_oe", "median_center_3x3_oe", "median_empirical_p",
        "q_all_le_0_10_count", "q_prefix_le_0_10_count", "median_q_all",
        "median_q_prefix",
    ]
    summary = frame.groupby(keys, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(part for part in column if part) for column in summary.columns]
    return summary


def aggregate_support80(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "method_name", "min_distance_bins", "branch_label"]
    metrics = [
        "support80_k", "support80_supported_count", "support80_fraction", "eligible_n"
    ]
    summary = frame.groupby(keys, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(part for part in column if part) for column in summary.columns]
    qualification = frame.assign(_qualifies=frame["support80_k"] > 0).groupby(
        keys, dropna=False
    ).agg(
        seed_count=("support80_k", "size"),
        qualifying_seed_count=("_qualifies", "sum"),
    ).reset_index()
    qualification["qualifying_seed_fraction"] = (
        qualification["qualifying_seed_count"] / qualification["seed_count"]
    )
    return summary.merge(qualification, on=keys, how="left", validate="one_to_one")


def aggregate_raw_supported_counts(
    frame: pd.DataFrame,
    *,
    min_distance_bins: int = 30,
    method_order: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Aggregate all eligible and raw-supported loops across held-out splits."""
    required = {"seed", "method", "method_name", "min_distance_bins", "raw_supported"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"per-loop support table is missing columns: {', '.join(missing)}")
    selected = frame.loc[frame["min_distance_bins"] == int(min_distance_bins)].copy()
    if selected.empty:
        raise ValueError(f"no loops found for min_distance_bins={int(min_distance_bins)}")

    seed_counts = selected.groupby(
        ["seed", "method", "method_name"], sort=False, as_index=False
    ).agg(
        eligible=("raw_supported", "size"),
        supported=("raw_supported", "sum"),
    )
    summary = seed_counts.groupby(
        ["method", "method_name"], sort=False, as_index=False
    ).agg(
        supported_mean=("supported", "mean"),
        supported_sd=("supported", "std"),
        eligible_mean=("eligible", "mean"),
        eligible_sd=("eligible", "std"),
        n_splits=("seed", "nunique"),
    )
    summary.insert(2, "min_distance_bins", int(min_distance_bins))

    observed_order = selected["method"].drop_duplicates().tolist()
    if method_order is None:
        ordered_methods = observed_order
    else:
        ordered_methods = [method for method in method_order if method in observed_order]
        ordered_methods.extend(method for method in observed_order if method not in ordered_methods)
    positions = {method: index for index, method in enumerate(ordered_methods)}
    summary["_order"] = summary["method"].map(positions)
    summary = summary.sort_values("_order", kind="mergesort").drop(columns="_order")
    return summary.reset_index(drop=True)


def plot_raw_supported_counts_panel(
    summary: pd.DataFrame,
    output_root: str | Path,
    *,
    stem: str = "panelB_600kb_raw_supported_counts",
) -> dict[str, Path]:
    """Export Panel B and the exact summary table used to draw it."""
    required = {
        "method", "method_name", "min_distance_bins", "supported_mean",
        "supported_sd", "eligible_mean", "eligible_sd", "n_splits",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Panel B summary is missing columns: {', '.join(missing)}")
    if summary.empty:
        raise ValueError("Panel B summary must contain at least one method")
    if summary["min_distance_bins"].nunique() != 1 or int(summary["min_distance_bins"].iloc[0]) != 30:
        raise ValueError("Panel B requires the 30-bin (600 kb) branch")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {extension: output_root / f"{stem}.{extension}" for extension in ["csv", "png", "pdf", "svg"]}
    plotted = summary.copy().reset_index(drop=True)
    plotted.to_csv(paths["csv"], index=False, float_format="%.6f")

    method_label = {
        "baseline_schicdiff": "Baseline\nscHiC-Diff",
        "ramani_ckpt_ep0999": "Ramani\nckpt ep999",
        "scvi3d": "scVI-3D",
        "schicluster": "scHiCluster",
        "higashi_nbr0": "Higashi\nnbr0",
        "higashi_nbr5": "Higashi\nnbr5",
    }
    highlight_colors = {
        "baseline_schicdiff": "#176B87",
        "ramani_ckpt_ep0999": "#2A9488",
    }
    neutral_color = "#B7BDC5"
    text_color = "#20252B"
    muted_color = "#5C6570"
    grid_color = "#D9DDE2"

    x = np.arange(len(plotted), dtype=float)
    supported = plotted["supported_mean"].to_numpy(dtype=float)
    supported_sd = plotted["supported_sd"].fillna(0).to_numpy(dtype=float)
    eligible = plotted["eligible_mean"].to_numpy(dtype=float)
    colors = [highlight_colors.get(str(method), neutral_color) for method in plotted["method"]]
    labels = [
        method_label.get(str(method), str(name))
        for method, name in plotted[["method", "method_name"]].itertuples(index=False)
    ]

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        figure, axis = plt.subplots(figsize=(9.2, 5.8))
        bars = axis.bar(x, supported, width=0.66, color=colors, edgecolor="none", zorder=3)
        axis.errorbar(
            x,
            supported,
            yerr=supported_sd,
            fmt="none",
            ecolor=text_color,
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            zorder=4,
        )

        annotation_top = supported + supported_sd + 2.1
        upper_limit = max(10.0, float(np.nanmax(annotation_top)) + 5.0)
        axis.set_ylim(0, upper_limit)
        for index, (bar, mean, deviation, eligible_mean, color) in enumerate(
            zip(bars, supported, supported_sd, eligible, colors)
        ):
            highlighted = color != neutral_color
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                max(0.8, mean - 1.8),
                f"{mean:.1f}",
                ha="center",
                va="top",
                color="white" if highlighted else text_color,
                fontsize=10,
                fontweight="medium",
                zorder=5,
            )
            axis.text(
                index,
                mean + deviation + 1.5,
                f"Eligible = {eligible_mean:.1f}",
                ha="center",
                va="bottom",
                color=muted_color,
                fontsize=8.5,
                zorder=5,
            )

        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_ylabel("Held-out raw-supported loops")
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color=grid_color, linewidth=0.8)
        axis.xaxis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#7A828B")
        axis.spines["bottom"].set_color("#7A828B")
        axis.tick_params(axis="x", length=0, pad=8, colors=text_color)
        axis.tick_params(axis="y", colors=text_color)

        figure.text(0.065, 0.955, "B", ha="left", va="top", fontsize=16, fontweight="medium", color=text_color)
        figure.text(
            0.105,
            0.955,
            "Independent raw support among all eligible long-range loops",
            ha="left",
            va="top",
            fontsize=13,
            fontweight="medium",
            color=text_color,
        )
        figure.text(
            0.105,
            0.912,
            "≥600 kb; mean ± SD across three mutually exclusive 238/238-cell splits",
            ha="left",
            va="top",
            fontsize=9.5,
            color=muted_color,
        )
        figure.text(
            0.105,
            0.028,
            "Eligible totals follow each method's native caller threshold; this is a descriptive sensitivity view, not a matched-N precision comparison.",
            ha="left",
            va="bottom",
            fontsize=8.3,
            color=muted_color,
        )
        figure.subplots_adjust(left=0.105, right=0.985, top=0.84, bottom=0.22)
        figure.savefig(paths["png"], dpi=400, facecolor="white")
        figure.savefig(paths["pdf"], facecolor="white")
        figure.savefig(paths["svg"], facecolor="white")
        plt.close(figure)
    return paths


def _plot_topn_summary(summary: pd.DataFrame, output_root: Path) -> None:
    for minimum, branch in summary[["min_distance_bins", "branch_label"]].drop_duplicates().itertuples(index=False):
        selected = summary.loc[summary["min_distance_bins"] == minimum]
        for metric, ylabel, stem in [
            ("supported_fraction", "held-out raw-supported fraction", "support_fraction"),
            ("median_center_oe", "median exact-pixel center O/E", "median_center_oe"),
        ]:
            figure, axis = plt.subplots(figsize=(7.0, 4.5))
            for method_name, group in selected.groupby("method_name", sort=False):
                group = group.sort_values("requested_n")
                y = group[f"{metric}_mean"].to_numpy(dtype=float)
                yerr = group[f"{metric}_std"].fillna(0).to_numpy(dtype=float)
                axis.errorbar(
                    group["requested_n"], y, yerr=yerr, marker="o", capsize=3,
                    label=method_name,
                )
            axis.set_xscale("log")
            axis.set_xticks([10, 20, 50, 100, 200])
            axis.set_xticklabels(["10", "20", "50", "100", "200"])
            axis.set_xlabel("rank prefix N")
            axis.set_ylabel(ylabel)
            axis.set_title(f"Held-out raw support: {branch}")
            axis.legend(fontsize=8)
            figure.tight_layout()
            figure.savefig(output_root / f"{stem}_{branch}.png", dpi=180)
            plt.close(figure)


def _plot_support80(summary: pd.DataFrame, output_root: Path) -> None:
    for minimum, branch in summary[["min_distance_bins", "branch_label"]].drop_duplicates().itertuples(index=False):
        selected = summary.loc[summary["min_distance_bins"] == minimum].copy()
        selected = selected.sort_values("support80_k_mean", ascending=False, kind="mergesort")
        figure, axis = plt.subplots(figsize=(7.0, 4.5))
        axis.bar(
            selected["method_name"],
            selected["support80_k_mean"],
            yerr=selected["support80_k_std"].fillna(0),
            capsize=3,
        )
        axis.set_ylabel("largest K with cumulative support ≥ 80%")
        axis.set_title(f"Held-out raw support operating point: {branch}")
        axis.tick_params(axis="x", rotation=30)
        figure.tight_layout()
        figure.savefig(output_root / f"support80_{branch}.png", dpi=180)
        plt.close(figure)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_report(
    topn_summary: pd.DataFrame,
    support80_summary: pd.DataFrame,
    output_root: Path,
) -> None:
    focus = topn_summary.loc[topn_summary["min_distance_bins"] == 30].copy()
    focus = focus.sort_values(["requested_n", "supported_fraction_mean"], ascending=[True, False])
    top_rows = [
        [
            str(row.method_name),
            f"Top{int(row.requested_n)}",
            f"{row.actual_n_mean:.1f}",
            f"{row.supported_count_mean:.2f}",
            f"{row.supported_fraction_mean:.3f}",
            f"{row.median_center_oe_mean:.3f}",
            f"{row.q_prefix_le_0_10_count_mean:.2f}",
        ]
        for row in focus.itertuples(index=False)
    ]
    operating = support80_summary.loc[support80_summary["min_distance_bins"] == 30].copy()
    operating = operating.sort_values("support80_k_mean", ascending=False, kind="mergesort")
    operating_rows = [
        [
            str(row.method_name),
            f"{row.support80_k_mean:.2f}",
            f"{row.support80_k_std:.2f}" if np.isfinite(row.support80_k_std) else "NA",
            f"{int(row.qualifying_seed_count)}/{int(row.seed_count)}",
            f"{row.eligible_n_mean:.1f}",
        ]
        for row in operating.itertuples(index=False)
    ]
    text = f"""# Held-out raw support–sensitivity report

## Protocol

- Three deterministic seeds (42, 43, 44), each with mutually exclusive 238-cell caller and 238-cell raw-reference halves.
- Exact-pixel diagonal O/E in the raw-reference half.
- Exact-distance exhaustive null with the tested coordinate excluded.
- A loop is raw-supported when center O/E > 1 and empirical upper-tail p <= 0.05.
- BH q-values are reported but are not the binary support rule because this 2 Mb locus gives a coarse exact-distance null.
- Distance filtering precedes score ranking; actual N is always recorded.

## 600 kb Top-N results

{_markdown_table(
    ["Method", "Prefix", "Actual N", "Supported", "Supported fraction", "Median O/E", "BH q<=0.10"],
    top_rows,
)}

## 600 kb 80% support operating point

{_markdown_table(["Method", "K mean", "K SD", "Qualifying seeds", "Eligible N"], operating_rows)}

## Interpretation limits

This is internal held-out validation, not external bulk Hi-C validation: loop ranking and raw support use mutually exclusive cells, but the imputation models may have been trained globally. The analysis covers one 2 Mb locus, exact-distance empirical p-values are coarse, and loops reuse anchors. Conclusions should therefore emphasize agreement across seeds and rank curves rather than treating individual p-values as definitive biological proof.
"""
    (output_root / "HELDOUT_RAW_SUPPORT_REPORT.md").write_text(text)


def run_analysis(
    *,
    methods: list[dict],
    reference_paths: dict[int, list[str | Path]],
    output_root: str | Path,
    seeds: list[int] | tuple[int, ...] = (42, 43, 44),
    resolution: int = 20_000,
    n_bins: int = 100,
    min_distance_bins_values: list[int] | tuple[int, ...] = (4, 30),
    top_n_values: list[int] | tuple[int, ...] = (10, 20, 50, 100, 200),
    support_oe_threshold: float = 1.0,
    support_p_threshold: float = 0.05,
    support_target_fraction: float = 0.80,
    support_min_prefix: int = 10,
) -> dict[str, pd.DataFrame]:
    """Run the full held-out raw support analysis and write auditable artifacts."""
    if resolution <= 0 or n_bins <= 0:
        raise ValueError("resolution and n_bins must be positive")
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    per_loop_parts: list[pd.DataFrame] = []
    topn_parts: list[pd.DataFrame] = []
    support80_rows: list[dict[str, object]] = []
    provenance_inputs: list[dict[str, object]] = []

    for seed in seeds:
        candidates = [Path(path).resolve() for path in reference_paths[int(seed)]]
        if not candidates:
            raise ValueError(f"no raw-reference paths configured for seed {seed}")
        missing = [str(path) for path in candidates if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        reference_hashes = [sha256_file(path) for path in candidates]
        if len(set(reference_hashes)) != 1:
            raise ValueError(f"raw-reference hash disagreement within seed {seed}")
        oe_matrix = load_reference_oe(candidates[0], n_bins=n_bins)
        for path, digest in zip(candidates, reference_hashes):
            provenance_inputs.append(
                {"kind": "raw_reference", "seed": int(seed), "path": str(path), "sha256": digest}
            )

        for method in methods:
            loop_paths = method["loop_paths"]
            loop_path = Path(loop_paths[int(seed)]).resolve()
            loops = load_native_loops(loop_path, resolution=resolution, n_bins=n_bins)
            provenance_inputs.append(
                {
                    "kind": "ranked_loops",
                    "seed": int(seed),
                    "method": method["method"],
                    "path": str(loop_path),
                    "sha256": sha256_file(loop_path),
                }
            )
            for minimum in min_distance_bins_values:
                ranked = prepare_ranked_loops(loops, min_distance_bins=int(minimum))
                scored = score_loops_against_reference(
                    ranked,
                    oe_matrix,
                    support_oe_threshold=support_oe_threshold,
                    support_p_threshold=support_p_threshold,
                )
                branch = f"{int(minimum) * int(resolution) // 1000}kb"
                metadata = {
                    "seed": int(seed),
                    "method": method["method"],
                    "method_name": method["method_name"],
                    "min_distance_bins": int(minimum),
                    "branch_label": branch,
                }
                for key, value in reversed(list(metadata.items())):
                    scored.insert(0, key, value)
                per_loop_parts.append(scored)

                topn = summarize_prefixes(scored, top_n_values=top_n_values)
                for key, value in reversed(list(metadata.items())):
                    topn.insert(0, key, value)
                topn_parts.append(topn)

                operating = support_operating_point(
                    scored,
                    target_fraction=support_target_fraction,
                    min_prefix=support_min_prefix,
                )
                support80_rows.append({**metadata, **operating})

    per_loop = pd.concat(per_loop_parts, ignore_index=True) if per_loop_parts else pd.DataFrame()
    topn_per_seed = pd.concat(topn_parts, ignore_index=True) if topn_parts else pd.DataFrame()
    support80_per_seed = pd.DataFrame.from_records(support80_rows)
    topn_summary = _aggregate_topn(topn_per_seed)
    support80_summary = aggregate_support80(support80_per_seed)
    panel_b_summary = pd.DataFrame()

    per_loop.to_csv(output_root / "per_loop_support.csv", index=False)
    topn_per_seed.to_csv(output_root / "topn_per_seed.csv", index=False)
    topn_summary.to_csv(output_root / "topn_summary.csv", index=False)
    support80_per_seed.to_csv(output_root / "support80_per_seed.csv", index=False)
    support80_summary.to_csv(output_root / "support80_summary.csv", index=False)
    if not per_loop.empty and (per_loop["min_distance_bins"] == 30).any():
        panel_b_summary = aggregate_raw_supported_counts(
            per_loop,
            min_distance_bins=30,
            method_order=PANEL_B_METHOD_ORDER,
        )
        plot_raw_supported_counts_panel(panel_b_summary, output_root)

    provenance = {
        "analysis": "heldout_raw_support_sensitivity",
        "seeds": [int(seed) for seed in seeds],
        "resolution": int(resolution),
        "n_bins": int(n_bins),
        "min_distance_bins_values": [int(value) for value in min_distance_bins_values],
        "top_n_values": [int(value) for value in top_n_values],
        "support_definition": {
            "center_oe_gt": float(support_oe_threshold),
            "empirical_p_lte": float(support_p_threshold),
            "null": "all other exact-distance upper-triangle pixels",
        },
        "support_operating_point": {
            "target_fraction": float(support_target_fraction),
            "minimum_prefix": int(support_min_prefix),
        },
        "methods": [
            {"method": method["method"], "method_name": method["method_name"]}
            for method in methods
        ],
        "inputs": provenance_inputs,
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    _plot_topn_summary(topn_summary, output_root)
    _plot_support80(support80_summary, output_root)
    _write_report(topn_summary, support80_summary, output_root)
    return {
        "per_loop": per_loop,
        "topn_per_seed": topn_per_seed,
        "topn_summary": topn_summary,
        "support80_per_seed": support80_per_seed,
        "support80_summary": support80_summary,
        "panel_b_summary": panel_b_summary,
    }


def default_inputs(seeds: list[int] | tuple[int, ...]) -> tuple[list[dict], dict[int, list[Path]]]:
    diagnostics = BENCHMARK_DIR / "results_diagnostics"
    definitions = [
        ("baseline_schicdiff", "Baseline scHiC-Diff", diagnostics / "heldout_raw_reference", "schicdiff"),
        ("ramani_ckpt_ep0999", "Ramani ckpt ep999", diagnostics / "heldout_ramani_v3/ramani_ckpt_ep0999", "schicdiff"),
        ("scvi3d", "scVI-3D", diagnostics / "heldout_raw_reference", "scvi3d"),
        ("schicluster", "scHiCluster", diagnostics / "heldout_comparator_methods/scHiCluster", "schicdiff"),
        ("higashi_nbr0", "Higashi nbr0", diagnostics / "heldout_comparator_methods/Higashi_nbr0", "schicdiff"),
        ("higashi_nbr5", "Higashi nbr5", diagnostics / "heldout_comparator_methods/Higashi_nbr5", "schicdiff"),
    ]
    methods = []
    for slug, label, root, loop_slug in definitions:
        methods.append(
            {
                "method": slug,
                "method_name": label,
                "loop_paths": {
                    int(seed): root / f"seed{seed}/loops/{loop_slug}/loops.loop.bedpe"
                    for seed in seeds
                },
            }
        )
    roots = [definition[2] for definition in definitions]
    reference_paths = {
        int(seed): [root / f"seed{seed}/reference/raw_reference_sum.npz" for root in roots]
        for seed in seeds
    }
    return methods, reference_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--top-n", nargs="+", type=int, default=[10, 20, 50, 100, 200])
    parser.add_argument("--support-p", type=float, default=0.05)
    parser.add_argument("--support-oe", type=float, default=1.0)
    parser.add_argument("--support-target", type=float, default=0.80)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    methods, reference_paths = default_inputs(args.seeds)
    results = run_analysis(
        methods=methods,
        reference_paths=reference_paths,
        output_root=args.output_root,
        seeds=args.seeds,
        top_n_values=args.top_n,
        support_p_threshold=args.support_p,
        support_oe_threshold=args.support_oe,
        support_target_fraction=args.support_target,
    )
    print(f"Wrote {len(results['per_loop'])} per-loop records to {Path(args.output_root).resolve()}")


if __name__ == "__main__":
    main()
