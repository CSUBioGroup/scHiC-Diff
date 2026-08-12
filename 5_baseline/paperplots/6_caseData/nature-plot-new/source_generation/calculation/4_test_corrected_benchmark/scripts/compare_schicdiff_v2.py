#!/usr/bin/env python3
"""Utilities and report generation for the scHiC-Diff v1-v2 comparison."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adapters import load_csr_npz
from build_group_matrices import index_sha256
from prepare_apa import BEDPE_COLUMNS, filter_and_rank_loops, load_loop_bedpe


def sync_subset_manifest(
    source_manifest: str | Path,
    destination_dir: str | Path,
) -> pd.DataFrame:
    source_manifest = Path(source_manifest)
    destination_dir = Path(destination_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(source_manifest)
    required = {"subset_path", "index_sha256"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"subset manifest is missing columns: {missing}")

    path_map: dict[str, str] = {}
    for source_raw in manifest["subset_path"].unique():
        source = Path(source_raw)
        if not source.exists():
            raise FileNotFoundError(source)
        destination = destination_dir / source.name
        shutil.copy2(source, destination)
        path_map[str(source_raw)] = str(destination.resolve())
    synced = manifest.copy()
    synced["subset_path"] = synced["subset_path"].map(path_map)
    for row in synced.itertuples(index=False):
        actual = index_sha256(np.load(row.subset_path))
        if actual != row.index_sha256:
            raise ValueError(f"subset hash mismatch after copy: {row.subset_path}")
    synced.to_csv(destination_dir / "subset_manifest.csv", index=False)
    return synced


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sparse_input_stats(path: str | Path, label: str) -> dict[str, object]:
    path = Path(path)
    matrix = load_csr_npz(path)
    data = matrix.data
    return {
        "label": label,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "cells": int(matrix.shape[0]),
        "features": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        "min": float(min(0.0, data.min())) if data.size else 0.0,
        "max": float(max(0.0, data.max())) if data.size else 0.0,
        "nonzero_mean": float(data.mean()) if data.size else 0.0,
        "finite": bool(np.isfinite(data).all()),
    }


def _as_loop_frame(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        missing = [column for column in BEDPE_COLUMNS if column not in value.columns]
        if missing:
            raise ValueError(f"loop frame is missing columns: {missing}")
        return value[BEDPE_COLUMNS].copy()
    return load_loop_bedpe(value)


def _coordinate_set(frame: pd.DataFrame) -> set[tuple[object, ...]]:
    columns = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    return set(map(tuple, frame[columns].to_numpy()))


def loop_overlap(
    v1: pd.DataFrame | str | Path,
    v2: pd.DataFrame | str | Path,
    resolution: int = 20000,
    min_distance_bins: int = 0,
    top_n: int | None = None,
) -> dict[str, object]:
    v1_filtered = filter_and_rank_loops(
        _as_loop_frame(v1), resolution, min_distance_bins=min_distance_bins, top_n=top_n
    )
    v2_filtered = filter_and_rank_loops(
        _as_loop_frame(v2), resolution, min_distance_bins=min_distance_bins, top_n=top_n
    )
    v1_set = _coordinate_set(v1_filtered)
    v2_set = _coordinate_set(v2_filtered)
    intersection = len(v1_set & v2_set)
    union = len(v1_set | v2_set)
    return {
        "min_distance_bins": int(min_distance_bins),
        "top_n": top_n,
        "v1_count": len(v1_set),
        "v2_count": len(v2_set),
        "intersection": intersection,
        "union": union,
        "jaccard": float(intersection / union) if union else 1.0,
        "v1_retained_fraction": float(intersection / len(v1_set)) if v1_set else np.nan,
        "v2_novel_fraction": float((len(v2_set) - intersection) / len(v2_set)) if v2_set else np.nan,
    }


def _combine_versions(
    v1: pd.DataFrame,
    v2: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    v1 = v1.copy()
    v2 = v2.copy()
    v1_keys = set(map(tuple, v1[keys].to_numpy()))
    v2_keys = set(map(tuple, v2[keys].to_numpy()))
    if v1_keys != v2_keys:
        missing_v2 = sorted(v1_keys - v2_keys)
        missing_v1 = sorted(v2_keys - v1_keys)
        raise ValueError(
            f"comparison keys differ; missing in v2={missing_v2[:3]}, missing in v1={missing_v1[:3]}"
        )
    v1["version"] = "v1"
    v2["version"] = "v2"
    return pd.concat([v1, v2], ignore_index=True, sort=False)


def combine_loop_summaries(v1: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    v1 = v1[v1["method"] == "schicdiff"].copy() if "schicdiff" in set(v1["method"]) else v1.copy()
    v2 = v2[v2["method"] == "schicdiff_v2"].copy() if "schicdiff_v2" in set(v2["method"]) else v2.copy()
    return _combine_versions(v1, v2, ["group", "cell_count"])


def combine_map2_summaries(v1: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    v1 = v1[v1["method"] == "schicdiff"].copy() if "schicdiff" in set(v1["method"]) else v1.copy()
    v2 = v2[v2["method"] == "schicdiff_v2"].copy() if "schicdiff_v2" in set(v2["method"]) else v2.copy()
    return _combine_versions(
        v1,
        v2,
        ["group", "cell_count", "transform", "background"],
    )


def combine_apa_manifests(v1: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    v1 = v1[v1["method"] == "schicdiff"].copy() if "schicdiff" in set(v1["method"]) else v1.copy()
    v2 = v2[v2["method"] == "schicdiff_v2"].copy() if "schicdiff_v2" in set(v2["method"]) else v2.copy()
    return _combine_versions(v1, v2, ["min_distance_bins", "set_label"])


def _as_frame(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else pd.read_csv(value)


def build_comparison_tables(
    *,
    v1_input: str | Path,
    v2_input: str | Path,
    v1_loop_summary: pd.DataFrame | str | Path,
    v2_loop_summary: pd.DataFrame | str | Path,
    v1_map2_summary: pd.DataFrame | str | Path,
    v2_map2_summary: pd.DataFrame | str | Path,
    v1_apa: pd.DataFrame | str | Path,
    v2_apa: pd.DataFrame | str | Path,
    v1_loops: pd.DataFrame | str | Path,
    v2_loops: pd.DataFrame | str | Path,
    resolution: int = 20000,
) -> dict[str, pd.DataFrame]:
    overlap_definitions = [
        ("all", 0, None),
        ("ge600kb", 30, None),
        ("ge600kb_top50", 30, 50),
        ("ge600kb_top100", 30, 100),
    ]
    overlap_rows = []
    for label, min_distance_bins, top_n in overlap_definitions:
        row = loop_overlap(
            v1_loops,
            v2_loops,
            resolution=resolution,
            min_distance_bins=min_distance_bins,
            top_n=top_n,
        )
        row["comparison"] = label
        overlap_rows.append(row)

    return {
        "input_comparison": pd.DataFrame(
            [sparse_input_stats(v1_input, "v1"), sparse_input_stats(v2_input, "v2")]
        ),
        "loop_summary_v1_v2": combine_loop_summaries(
            _as_frame(v1_loop_summary), _as_frame(v2_loop_summary)
        ),
        "loop_overlap_476": pd.DataFrame(overlap_rows),
        "map2_summary_v1_v2": combine_map2_summaries(
            _as_frame(v1_map2_summary), _as_frame(v2_map2_summary)
        ),
        "apa_summary_v1_v2": combine_apa_manifests(_as_frame(v1_apa), _as_frame(v2_apa)),
    }


def _plot_loop_counts(frame: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    colors = {"v1": "#4C78A8", "v2": "#E45756"}
    markers = {"earlyNeurons": "o", "nonEarlyNeurons": "s"}
    for (version, group), subset in frame.groupby(["version", "group"], sort=True):
        subset = subset.sort_values("cell_count")
        axis.plot(
            subset["cell_count"],
            subset["loop_count_mean"],
            marker=markers.get(group, "o"),
            color=colors.get(version),
            label=f"{version} · {group}",
        )
    axis.set_xlabel("Cells")
    axis.set_ylabel("Mean loop count")
    axis.set_title("scHiC-Diff v1 vs v2 loop depth curves")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_map2(frame: pd.DataFrame, path: Path) -> None:
    focus = frame[(frame["group"] == "earlyNeurons") & (frame["cell_count"] == 476)].copy()
    if focus.empty:
        focus = frame.copy()
    focus["metric_key"] = focus["transform"].astype(str) + "\n" + focus["background"].astype(str)
    pivot = focus.pivot_table(index="metric_key", columns="version", values="ratio_mean", aggfunc="first")
    pivot = pivot.reindex(columns=[column for column in ["v1", "v2"] if column in pivot])
    figure, axis = plt.subplots(figsize=(8, 5.5))
    pivot.plot.bar(ax=axis, color=["#4C78A8", "#E45756"][: len(pivot.columns)], width=0.75)
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("Transform / distance-matched background")
    axis.set_ylabel("Map2 candidate / control ratio")
    axis.set_title("Map2 at earlyNeurons, 476 cells")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Version", frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_apa(frame: pd.DataFrame, path: Path) -> None:
    focus = frame.copy()
    focus["metric_key"] = focus["min_distance_bins"].map({4: "80 kb", 30: "600 kb"}).fillna(
        focus["min_distance_bins"].astype(str) + " bins"
    ) + "\n" + focus["set_label"].astype(str)
    key_order = list(dict.fromkeys(focus["metric_key"]))
    pivot = focus.pivot_table(index="metric_key", columns="version", values="P2LL", aggfunc="first")
    pivot = pivot.reindex(key_order)
    pivot = pivot.reindex(columns=[column for column in ["v1", "v2"] if column in pivot])
    figure, axis = plt.subplots(figsize=(10, 5.8))
    pivot.plot.bar(ax=axis, color=["#4C78A8", "#E45756"][: len(pivot.columns)], width=0.78)
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("Minimum distance / loop set")
    axis.set_ylabel("APA P2LL")
    axis.set_title("Corrected APA with actual effective loop counts")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Version", frameon=False)
    for container, version in zip(axis.containers, pivot.columns):
        counts = (
            focus[focus["version"] == version]
            .drop_duplicates("metric_key")
            .set_index("metric_key")
            .reindex(key_order)["effective_count"]
        )
        labels = [f"N={int(value)}" if pd.notna(value) else "" for value in counts]
        axis.bar_label(container, labels=labels, padding=2, fontsize=7, rotation=90)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _markdown_report(tables: dict[str, pd.DataFrame]) -> str:
    inputs = tables["input_comparison"].set_index("label")
    loops = tables["loop_summary_v1_v2"]
    overlap = tables["loop_overlap_476"]
    map2 = tables["map2_summary_v1_v2"]
    apa = tables["apa_summary_v1_v2"]

    loop_focus = loops[(loops["group"] == "earlyNeurons") & (loops["cell_count"] == 476)]
    map2_focus = map2[
        (map2["group"] == "earlyNeurons")
        & (map2["cell_count"] == 476)
        & (map2["transform"] == "none")
        & (map2["background"] == "donut")
    ]
    input_rows = "\n".join(
        f"| {label} | {int(row.nnz):,} | {row.density:.6f} | {row.nonzero_mean:.6f} | {row['max']:.6f} |"
        for label, row in inputs.iterrows()
    )
    loop_rows = "\n".join(
        f"| {row.version} | {row.loop_count_mean:.3f} | {row.summit_count_mean:.3f} |"
        for row in loop_focus.itertuples(index=False)
    )
    overlap_rows = "\n".join(
        f"| {row.comparison} | {int(row.v1_count)} | {int(row.v2_count)} | {int(row.intersection)} | {row.jaccard:.4f} |"
        for row in overlap.itertuples(index=False)
    )
    map2_rows = "\n".join(
        f"| {row.version} | {row.ratio_mean:.4f} | {row.percentile_mean:.2f}% | {row.empirical_p_upper_mean:.4f} |"
        for row in map2_focus.itertuples(index=False)
    )
    apa_rows = "\n".join(
        f"| {row.version} | {int(row.min_distance_bins)} | {row.set_label} | {int(row.effective_count)} | {row.P2LL:.4f} | {row.P2M:.4f} | {row.ZscoreLL:.4f} |"
        for row in apa.sort_values(["min_distance_bins", "set_label", "version"]).itertuples(index=False)
    )
    return f"""# scHiC-Diff v1 vs v2 corrected benchmark

## Scope and comparability

- v2 was evaluated independently under `results_eval_v2`; baseline outputs under `results` were not rewritten.
- Both versions use the same canonical 7,466-cell row order, synchronized subset hashes, seeds 42/43/44, loop parameters, Map2 controls, and APA branches.
- The v2 NPZ has no cell names, so equality to the canonical row order remains an explicit assumption.
- APA applies distance filtering before Top50/Top100 and reports the actual effective N.

## Input distribution

| Version | NNZ | Density | Nonzero mean | Maximum |
|---|---:|---:|---:|---:|
{input_rows}

## Loop calling at earlyNeurons, 476 cells

| Version | Loops | Summits |
|---|---:|---:|
{loop_rows}

| Coordinate comparison | v1 N | v2 N | Shared | Jaccard |
|---|---:|---:|---:|---:|
{overlap_rows}

## Map2, untransformed donut background

| Version | Ratio | Control percentile | Empirical upper p |
|---|---:|---:|---:|
{map2_rows}

## Corrected APA

| Version | Min distance (bins) | Set | Effective N | P2LL | P2M | ZscoreLL |
|---|---:|---|---:|---:|---:|---:|
{apa_rows}

## Interpretation

v2 is substantially sparser and far more selective than v1. Its 476-cell loop set is not an expanded version of v1: the exact-coordinate overlap is small, so the change is primarily a different and much smaller call set rather than simple dilution by extra loops.

Map2 and APA answer different questions. Map2 measures support for one predefined candidate relative to distance-matched controls; an improved Map2 percentile can coexist with weaker genome-wide APA. APA evaluates the aggregate enrichment of the called loop set against raw Hi-C, and its effective N must be considered because v2 provides fewer than 50 eligible loops in both distance branches.

For v2, All/Top50/Top100 are identical within each APA distance branch because only 10 loops pass 80 kb and 8 pass 600 kb. Those repeated labels are retained for direct protocol matching, but they are not independent loop sets.
"""


def write_comparison_outputs(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
    report_path: str | Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for name, frame in tables.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        outputs.append(path)

    plot_paths = [
        figures_dir / "schicdiff_v1_v2_loop_counts.png",
        figures_dir / "schicdiff_v1_v2_map2_476.png",
        figures_dir / "schicdiff_v1_v2_apa.png",
    ]
    _plot_loop_counts(tables["loop_summary_v1_v2"], plot_paths[0])
    _plot_map2(tables["map2_summary_v1_v2"], plot_paths[1])
    _plot_apa(tables["apa_summary_v1_v2"], plot_paths[2])
    outputs.extend(plot_paths)

    report_path.write_text(_markdown_report(tables))
    outputs.append(report_path)
    return outputs


def main() -> None:
    default_benchmark = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=default_benchmark)
    args = parser.parse_args()
    benchmark = args.benchmark_dir.resolve()
    workspace = benchmark.parent
    baseline = benchmark / "results"
    v2_root = benchmark / "results_eval_v2"
    tables = build_comparison_tables(
        v1_input=workspace / "imputedData/scHiC-Diff/chr1/denoise_recon_inv.npz",
        v2_input=workspace / "imputedData/scHiC-Diff/eval_v2/denoise_recon_inv.npz",
        v1_loop_summary=baseline / "summaries/loop_summary.csv",
        v2_loop_summary=v2_root / "summaries/loop_summary.csv",
        v1_map2_summary=baseline / "summaries/map2_summary.csv",
        v2_map2_summary=v2_root / "summaries/map2_summary.csv",
        v1_apa=baseline / "apa/apa_manifest.csv",
        v2_apa=v2_root / "apa/apa_manifest.csv",
        v1_loops=baseline / "loops/schicdiff/earlyNeurons/476cells_seed42/loops.loop.bedpe",
        v2_loops=v2_root / "loops/schicdiff_v2/earlyNeurons/476cells_seed42/loops.loop.bedpe",
    )
    outputs = write_comparison_outputs(
        tables,
        v2_root / "summaries",
        benchmark / "docs/SCHICDIFF_V1_V2_COMPARISON.md",
    )
    print(f"wrote {len(outputs)} comparison outputs")


if __name__ == "__main__":
    main()
