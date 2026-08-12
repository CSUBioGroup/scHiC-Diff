#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot repeat mean ± sample SD for corrected early-neuron loop calls.

The recommended figure reports distinct loop summits, matching the circles in
``plot_call_loops_seed42.py``. ``--metric loop_count`` renders the caller's
significant loop pixels from the same frozen source tables.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_call_loops_seed42 import (
    CELL_COUNTS,
    GROUP,
    METHOD_LABELS,
    METHOD_LABEL_BY_SLUG,
    METHOD_SLUGS,
    discover_experiment_root,
    discover_style_dir,
    sha256_file,
)


RAW_COUNTS_NAME = "call_loop_counts_three_seed_raw_with_flamingo.csv"
SUMMARY_COUNTS_NAME = "call_loop_counts_three_seed_summary_with_flamingo.csv"
SOURCE_MANIFEST_NAME = "call_loop_counts_three_seed_source_manifest_with_flamingo.csv"
OUTPUT_STEMS = {
    "summit_count": "call_loop_summit_counts_three_seed_with_flamingo",
    "loop_count": "call_loop_significant_pixels_three_seed_with_flamingo",
}
Y_LABELS = {
    "summit_count": "Distinct loop summits",
    "loop_count": "Significant loop pixels",
}


def _add_import_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def normalize_run_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter, order, and validate the approved per-run comparison rows."""
    required = {
        "status",
        "method",
        "group",
        "cell_count",
        "seed",
        "loop_count",
        "summit_count",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"loop run manifest is missing columns: {', '.join(missing)}")
    selected = frame.loc[
        frame["status"].eq("completed")
        & frame["method"].isin(METHOD_SLUGS)
        & frame["group"].eq(GROUP)
        & frame["cell_count"].astype(int).isin(CELL_COUNTS)
    ].copy()
    selected["cell_count"] = selected["cell_count"].astype(int)
    selected["seed"] = selected["seed"].astype(int)
    selected["loop_count"] = selected["loop_count"].astype(int)
    selected["summit_count"] = selected["summit_count"].astype(int)
    if (selected[["loop_count", "summit_count"]] < 0).any().any():
        raise ValueError("loop counts must be non-negative")

    for method in METHOD_SLUGS:
        for count in CELL_COUNTS:
            observed = selected.loc[
                selected["method"].eq(method) & selected["cell_count"].eq(count), "seed"
            ].sort_values().tolist()
            expected = [42] if count == 476 else [42, 43, 44]
            if observed != expected:
                raise ValueError(f"unexpected repeats for {method}/{count} cells: {observed}; expected {expected}")
    if selected.duplicated(["method", "cell_count", "seed"]).any():
        raise ValueError("loop run manifest contains duplicate method/count/seed rows")

    selected["method_name"] = selected["method"].map(METHOD_LABEL_BY_SLUG)
    method_order = {method: index for index, method in enumerate(METHOD_SLUGS)}
    count_order = {count: index for index, count in enumerate(CELL_COUNTS)}
    selected["_method_order"] = selected["method"].map(method_order)
    selected["_count_order"] = selected["cell_count"].map(count_order)
    keep = [
        "method",
        "method_name",
        "group",
        "cell_count",
        "seed",
        "loop_count",
        "summit_count",
    ]
    return selected.sort_values(["_method_order", "_count_order", "seed"])[keep].reset_index(drop=True)


def summarize_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate arithmetic means and sample SDs; preserve n=1 as missing SD."""
    required = {"method", "method_name", "group", "cell_count", "seed", "loop_count", "summit_count"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"per-run count table is missing columns: {', '.join(missing)}")
    records: list[dict[str, Any]] = []
    for (method, method_name, group, count), values in frame.groupby(
        ["method", "method_name", "group", "cell_count"], sort=False
    ):
        values = values.sort_values("seed")
        record: dict[str, Any] = {
            "method": method,
            "method_name": method_name,
            "group": group,
            "cell_count": int(count),
            "n_repeats": int(len(values)),
            "seeds": ";".join(str(int(value)) for value in values["seed"]),
        }
        for metric in ("loop_count", "summit_count"):
            metric_values = values[metric].to_numpy(dtype=float)
            record[f"{metric}_mean"] = float(metric_values.mean())
            record[f"{metric}_sd"] = (
                float(metric_values.std(ddof=1)) if len(metric_values) > 1 else np.nan
            )
            record[f"{metric}_min"] = int(metric_values.min())
            record[f"{metric}_max"] = int(metric_values.max())
        records.append(record)
    result = pd.DataFrame(records)
    method_order = {method: index for index, method in enumerate(METHOD_SLUGS)}
    count_order = {count: index for index, count in enumerate(CELL_COUNTS)}
    result["_method_order"] = result["method"].map(method_order)
    result["_count_order"] = result["cell_count"].map(count_order)
    return result.sort_values(["_method_order", "_count_order"]).drop(
        columns=["_method_order", "_count_order"]
    ).reset_index(drop=True)


def _assert_summary_matches_benchmark(summary: pd.DataFrame, reference: pd.DataFrame) -> None:
    reference = reference.loc[
        reference["method"].isin(METHOD_SLUGS)
        & reference["group"].eq(GROUP)
        & reference["cell_count"].astype(int).isin(CELL_COUNTS)
    ].copy()
    if len(reference) != len(METHOD_SLUGS) * len(CELL_COUNTS):
        raise ValueError("benchmark loop_summary.csv lacks the approved 24 summary rows")
    merged = summary.merge(
        reference,
        on=["method", "group", "cell_count"],
        suffixes=("_new", "_reference"),
        validate="one_to_one",
    )
    # Merge suffixing affects columns that exist in both inputs; make the mapping explicit.
    actual_columns = {
        "n_repeats": "repeat_count",
        "loop_count_mean_new": "loop_count_mean_reference",
        "loop_count_sd": "loop_count_std",
        "summit_count_mean_new": "summit_count_mean_reference",
        "summit_count_sd": "summit_count_std",
    }
    for calculated, expected in actual_columns.items():
        if calculated not in merged.columns or expected not in merged.columns:
            raise ValueError(f"cannot cross-check summary columns {calculated}/{expected}")
        if not np.allclose(
            merged[calculated].to_numpy(dtype=float),
            merged[expected].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"calculated {calculated} does not match benchmark {expected}")


def build_count_sources(experiment_root: str | Path, data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = discover_experiment_root(experiment_root)
    benchmark_dir = root / "4_test_corrected_benchmark"
    run_manifest_path = benchmark_dir / "results/manifests/loop_runs.csv"
    loop_summary_path = benchmark_dir / "results/summaries/loop_summary.csv"
    flamingo_run_manifest_path = benchmark_dir / "results_flamingo/manifests/loop_runs.csv"
    flamingo_loop_summary_path = benchmark_dir / "results_flamingo/summaries/loop_summary.csv"
    for required in (
        run_manifest_path,
        loop_summary_path,
        flamingo_run_manifest_path,
        flamingo_loop_summary_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    raw = normalize_run_frame(
        pd.concat(
            [pd.read_csv(run_manifest_path), pd.read_csv(flamingo_run_manifest_path)],
            ignore_index=True,
            sort=False,
        )
    )
    summary = summarize_counts(raw)
    reference = pd.concat(
        [pd.read_csv(loop_summary_path), pd.read_csv(flamingo_loop_summary_path)],
        ignore_index=True,
        sort=False,
    )
    _assert_summary_matches_benchmark(summary, reference)

    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(data_dir / RAW_COUNTS_NAME, index=False)
    summary.to_csv(data_dir / SUMMARY_COUNTS_NAME, index=False)
    pd.DataFrame(
        [
            {
                "role": "loop_run_manifest",
                "path": str(run_manifest_path.resolve()),
                "sha256": sha256_file(run_manifest_path),
            },
            {
                "role": "loop_summary_crosscheck",
                "path": str(loop_summary_path.resolve()),
                "sha256": sha256_file(loop_summary_path),
            },
            {
                "role": "flamingo_loop_run_manifest",
                "path": str(flamingo_run_manifest_path.resolve()),
                "sha256": sha256_file(flamingo_run_manifest_path),
            },
            {
                "role": "flamingo_loop_summary_crosscheck",
                "path": str(flamingo_loop_summary_path.resolve()),
                "sha256": sha256_file(flamingo_loop_summary_path),
            },
        ]
    ).to_csv(data_dir / SOURCE_MANIFEST_NAME, index=False)
    return raw, summary


def load_count_sources(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    raw_path = data_dir / RAW_COUNTS_NAME
    summary_path = data_dir / SUMMARY_COUNTS_NAME
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path)
    expected_rows = len(METHOD_SLUGS) * (3 * 3 + 1)
    if len(raw) != expected_rows:
        raise ValueError(f"frozen raw count table has {len(raw)} rows, expected {expected_rows}")
    if len(summary) != len(METHOD_SLUGS) * len(CELL_COUNTS):
        raise ValueError("frozen summary table must contain 24 rows")
    return raw, summary


def plot_repeat_counts(
    summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    metric: str = "summit_count",
    style_dir: str | Path,
    dpi: int = 600,
) -> dict[str, Path]:
    if metric not in OUTPUT_STEMS:
        raise ValueError(f"unsupported metric: {metric}")
    if dpi < 300:
        raise ValueError("dpi must be at least 300 for publication output")
    style_dir = Path(style_dir).resolve()
    _add_import_path(style_dir)
    gr_stagefig = importlib.import_module("gr_stagefig")
    gr_stagefig.set_gr_style()
    styles = gr_stagefig.resolve_method_styles(METHOD_LABELS, highlight="scHiC-Diff", baseline="Raw")
    # The shared helper has four comparator colors; the seventh condition needs
    # a fifth non-repeating color/shape to keep the legend unambiguous.
    styles["Higashi-5"] = {
        **styles["Higashi-5"],
        "color": gr_stagefig.OKABE_ITO["orange"],
        "marker": "X",
        "mec": gr_stagefig.OKABE_ITO["orange"],
    }

    x = np.arange(len(CELL_COUNTS), dtype=float)
    figure, axis = plt.subplots(figsize=(gr_stagefig.COL2, gr_stagefig.mm(90.0)))
    for method, label in zip(METHOD_SLUGS, METHOD_LABELS):
        selected = summary.loc[summary["method"].eq(method)].copy()
        selected["_count_order"] = selected["cell_count"].map(
            {count: index for index, count in enumerate(CELL_COUNTS)}
        )
        selected = selected.sort_values("_count_order")
        if selected["cell_count"].astype(int).tolist() != CELL_COUNTS:
            raise ValueError(f"summary lacks the approved cell counts for {method}")
        means = selected[f"{metric}_mean"].to_numpy(dtype=float)
        deviations = selected[f"{metric}_sd"].fillna(0).to_numpy(dtype=float)
        style = styles[label]
        axis.errorbar(
            x,
            means,
            yerr=deviations,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            marker=style["marker"],
            markersize=style["ms"] + 1.0,
            markerfacecolor=style["mfc"],
            markeredgecolor=style["mec"],
            markeredgewidth=style["mew"],
            capsize=2.2,
            capthick=0.6,
            elinewidth=0.7,
            zorder=style["zorder"],
            label=label,
        )

    axis.set_xticks(x, [str(count) for count in CELL_COUNTS])
    axis.set_xlabel("Aggregated cells")
    axis.set_ylabel(Y_LABELS[metric])
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#D9D8D3", linewidth=gr_stagefig.LW_HAIR)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", ncol=3, handlelength=2.0)
    axis.set_title(f"{Y_LABELS[metric]} across aggregation depth", loc="left", pad=8)
    figure.text(
        0.11,
        0.025,
        "Mean ± sample SD across seeds 42/43/44 for 10–200 cells; 476 cells is the full early-neuron group (n=1, no SD).",
        ha="left",
        va="bottom",
        fontsize=gr_stagefig.PT_SMALL,
        color=gr_stagefig.TEXT_MUTED,
    )
    figure.subplots_adjust(left=0.11, right=0.985, top=0.86, bottom=0.22)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_STEMS[metric]
    outputs = {extension: output_dir / f"{stem}.{extension}" for extension in ("png", "pdf", "svg")}
    figure.savefig(outputs["png"], dpi=dpi, facecolor="white")
    figure.savefig(outputs["pdf"], facecolor="white")
    figure.savefig(outputs["svg"], facecolor="white")
    plt.close(figure)
    return outputs


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--style-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=script_dir)
    parser.add_argument("--out-dir", type=Path, default=script_dir)
    parser.add_argument("--rebuild-source", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument(
        "--metric",
        choices=["summit_count", "loop_count", "both"],
        default="summit_count",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    root: Path | None = None
    if args.rebuild_source or not (data_dir / SUMMARY_COUNTS_NAME).is_file():
        root = discover_experiment_root(args.experiment_root)
        _, summary = build_count_sources(root, data_dir)
        source_mode = "rebuilt"
    else:
        _, summary = load_count_sources(data_dir)
        source_mode = "frozen"
    print(f"source: {source_mode} ({data_dir / SUMMARY_COUNTS_NAME})")
    if args.source_only:
        return
    if root is None:
        root = discover_experiment_root(args.experiment_root)
    style_dir = discover_style_dir(root, args.style_dir)
    metrics = ["summit_count", "loop_count"] if args.metric == "both" else [args.metric]
    for metric in metrics:
        print(metric)
        outputs = plot_repeat_counts(summary, args.out_dir, metric=metric, style_dir=style_dir, dpi=args.dpi)
        for extension, path in outputs.items():
            print(f"  {extension}: {path}")


if __name__ == "__main__":
    main()
