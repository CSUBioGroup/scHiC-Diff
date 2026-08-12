"""Strict loaders for the copied formal plotting data.

This module validates and reshapes already computed results. It never reruns a
biological metric, dimensional reduction, loop caller, or APA calculation.
"""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from figure_registry import (
    ALL_CELL_COUNTS,
    ALL_METHODS,
    APA_TOP_N_VALUES,
    IMPUTED_METHODS,
    SEEDS,
    STAGES,
    TOP_N_VALUES,
    canonical_method,
)


@dataclass
class FigureData:
    umap_points: pd.DataFrame
    umap_summary: pd.DataFrame
    stage_metadata: dict
    contact_matrices: Dict[Tuple[str, int], np.ndarray]
    contact_summits: Dict[Tuple[str, int], np.ndarray]
    contact_counts: pd.DataFrame
    loop_counts: pd.DataFrame
    contact_resolution_bp: int
    contact_n_bins: int
    apa_split_matrices: Dict[Tuple[str, int, int], np.ndarray]
    apa_matrices: Dict[Tuple[str, int], np.ndarray]
    apa_metrics: pd.DataFrame
    apa_resolution_bp: int
    apa_window_bins: int
    apa_min_distance_bins: int
    support_fraction: pd.DataFrame
    support_counts: pd.DataFrame


def _require_columns(frame, required, label):
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError("{} is missing columns: {}".format(label, ", ".join(missing)))


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_copied_manifest(data_dir):
    """Verify every copied plotting input before any table or array is loaded."""

    data_dir = Path(data_dir).resolve()
    manifest = data_dir / "copied_data_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError("copied-data manifest does not exist: {}".format(manifest))
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("copied-data manifest is empty")
    for row in rows:
        relative = Path(row["destination"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe copied-data destination: {}".format(relative))
        path = data_dir / relative
        if not path.is_file():
            raise FileNotFoundError("copied plotting input is missing: {}".format(path))
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError("copied-data size mismatch for {}".format(path))
        if _sha256_file(path) != row["sha256"]:
            raise ValueError("copied-data SHA-256 mismatch for {}".format(path))
    return len(rows)


def _canonicalize_method_column(frame, source_column="method"):
    result = frame.copy()
    result["method_id"] = result[source_column].map(canonical_method)
    return result


def validate_umap_tables(points, summary):
    """Validate the formal exported UMAP points and stored silhouette summary."""

    _require_columns(
        points,
        {
            "method",
            "method_storage",
            "stage",
            "cell_id",
            "lineage",
            "schUMAP_1",
            "schUMAP_2",
            "silhouette",
        },
        "UMAP source data",
    )
    _require_columns(
        summary,
        {
            "method",
            "method_storage",
            "stage",
            "n_red",
            "n_blue",
            "n_used",
            "mean_silhouette",
        },
        "UMAP summary",
    )
    points = _canonicalize_method_column(points, "method_storage")
    summary = _canonicalize_method_column(summary, "method_storage")

    if points.duplicated(["method_id", "cell_id"]).any():
        raise ValueError("duplicate method/cell pair in UMAP source data")
    if summary.duplicated(["method_id", "stage"]).any():
        raise ValueError("duplicate method/stage pair in UMAP summary")
    if set(points["method_id"]) != set(ALL_METHODS):
        raise ValueError("UMAP method set does not match the seven formal methods")
    if set(summary["method_id"]) != set(ALL_METHODS):
        raise ValueError("UMAP summary method set does not match the seven formal methods")
    if set(points["stage"].astype(str)) != set(STAGES):
        raise ValueError("UMAP stage set is incomplete")
    if set(summary["stage"].astype(str)) != set(STAGES):
        raise ValueError("UMAP summary stage set is incomplete")
    if len(summary) != len(ALL_METHODS) * len(STAGES):
        raise ValueError("UMAP summary must contain exactly 7 x 7 rows")
    if not set(points["lineage"].dropna().unique()).issubset({"Red", "Blue"}):
        raise ValueError("UMAP source contains lineage values other than Red/Blue")

    numeric = points[["schUMAP_1", "schUMAP_2", "silhouette"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("UMAP coordinates or stored silhouette contain non-finite values")

    grouped = points.groupby(["method_id", "stage"], sort=False)
    observed_counts = grouped.size()
    observed_means = grouped["silhouette"].mean()
    for row in summary.itertuples(index=False):
        key = (row.method_id, str(row.stage))
        if key not in observed_counts:
            raise ValueError("UMAP source is missing method/stage {}".format(key))
        if int(observed_counts.loc[key]) != int(row.n_used):
            raise ValueError("UMAP n_used mismatch for {}".format(key))
        sub = points.loc[
            points["method_id"].eq(row.method_id) & points["stage"].eq(row.stage)
        ]
        if int(sub["lineage"].eq("Red").sum()) != int(row.n_red):
            raise ValueError("UMAP n_red mismatch for {}".format(key))
        if int(sub["lineage"].eq("Blue").sum()) != int(row.n_blue):
            raise ValueError("UMAP n_blue mismatch for {}".format(key))
        if not np.isclose(
            float(observed_means.loc[key]),
            float(row.mean_silhouette),
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError("stored silhouette mean mismatch for {}".format(key))

    method_rank = {method: index for index, method in enumerate(ALL_METHODS)}
    stage_rank = {stage: index for index, stage in enumerate(STAGES)}
    for frame in (points, summary):
        frame["_method_rank"] = frame["method_id"].map(method_rank)
        frame["_stage_rank"] = frame["stage"].map(stage_rank)
    points = points.sort_values(
        ["_method_rank", "_stage_rank", "cell_id"], kind="stable"
    ).drop(columns=["_method_rank", "_stage_rank"])
    summary = summary.sort_values(
        ["_method_rank", "_stage_rank"], kind="stable"
    ).drop(columns=["_method_rank", "_stage_rank"])
    return points.reset_index(drop=True), summary.reset_index(drop=True)


def load_umap(data_dir):
    root = Path(data_dir) / "developmental_stage"
    points = pd.read_csv(root / "source_data_panel_a.csv")
    summary = pd.read_csv(root / "source_data_panel_a_summary.csv")
    with (root / "nature_panel_a_run_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    points, summary = validate_umap_tables(points, summary)
    return points, summary, metadata


def validate_contact_arrays(
    matrices,
    summits,
    n_bins,
    expected_methods=ALL_METHODS,
    expected_counts=ALL_CELL_COUNTS,
):
    expected = {
        (method, int(count)) for method in expected_methods for count in expected_counts
    }
    if set(matrices) != expected or set(summits) != expected:
        raise ValueError("contact matrix/summit method-depth set is incomplete")
    for key in sorted(expected):
        matrix = np.asarray(matrices[key])
        summit = np.asarray(summits[key])
        if matrix.shape != (n_bins, n_bins):
            raise ValueError("contact matrix {} must be 100 x 100".format(key))
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("contact matrix {} contains invalid values".format(key))
        if summit.ndim != 2 or summit.shape[1] != 2:
            raise ValueError("summit array {} must have shape (n, 2)".format(key))
        if summit.size and ((summit < 0).any() or (summit >= n_bins).any()):
            raise ValueError("summit array {} contains out-of-range bins".format(key))
    return matrices, summits


def load_contact_loops(data_dir):
    root = Path(data_dir) / "contact_loops"
    archive_path = root / "call_loops_seed42_panel_data_with_flamingo.npz"
    matrices = {}
    summits = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        source_methods = archive["method_slugs"].astype(str).tolist()
        source_counts = tuple(int(value) for value in archive["cell_counts"].tolist())
        n_bins = int(archive["n_bins"])
        resolution_bp = int(archive["resolution_bp"])
        if {canonical_method(method) for method in source_methods} != set(ALL_METHODS):
            raise ValueError("contact archive method set is incomplete")
        if source_counts != ALL_CELL_COUNTS:
            raise ValueError("contact archive cell-count order is unexpected")
        for source_method in source_methods:
            method = canonical_method(source_method)
            for count in source_counts:
                key = (method, count)
                matrices[key] = np.asarray(
                    archive["matrix__{}__{}".format(source_method, count)], dtype=float
                )
                summits[key] = np.asarray(
                    archive["summits__{}__{}".format(source_method, count)], dtype=int
                )
    validate_contact_arrays(matrices, summits, n_bins)

    counts = pd.read_csv(root / "call_loops_seed42_panel_counts_with_flamingo.csv")
    counts = _canonicalize_method_column(counts, "method")
    if len(counts) != len(ALL_METHODS) * len(ALL_CELL_COUNTS):
        raise ValueError("contact count table must contain 28 rows")
    return matrices, summits, counts, resolution_bp, n_bins


def load_loop_counts(data_dir):
    path = (
        Path(data_dir)
        / "contact_loops/call_loop_counts_three_seed_summary_with_flamingo.csv"
    )
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        {
            "method",
            "cell_count",
            "n_repeats",
            "loop_count_mean",
            "loop_count_sd",
            "summit_count_mean",
            "summit_count_sd",
        },
        "repeat loop-count summary",
    )
    frame = _canonicalize_method_column(frame, "method")
    observed = set(zip(frame["method_id"], frame["cell_count"].astype(int)))
    expected = {
        (method, count) for method in ALL_METHODS for count in ALL_CELL_COUNTS
    }
    if observed != expected or len(frame) != len(expected):
        raise ValueError("repeat loop-count summary method-depth set is incomplete")
    return frame.reset_index(drop=True)


def validate_apa_arrays(
    arrays,
    expected_methods=IMPUTED_METHODS,
    expected_top_n=APA_TOP_N_VALUES,
    expected_seeds=SEEDS,
):
    expected = {
        (method, int(top_n), int(seed))
        for method in expected_methods
        for top_n in expected_top_n
        for seed in expected_seeds
    }
    if set(arrays) != expected:
        raise ValueError("APA matrix method/Top-N/seed set is incomplete")
    for key in sorted(expected):
        matrix = np.asarray(arrays[key])
        if matrix.shape != (21, 21):
            raise ValueError("APA matrix {} must be 21 x 21".format(key))
        if not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError("APA matrix {} contains invalid values".format(key))
    return arrays


def load_apa(data_dir):
    root = Path(data_dir) / "heldout_apa"
    archive_path = (
        root / "apa_600kb_top10_top20_top50_normed_matrices_with_flamingo.npz"
    )
    split_matrices = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        source_methods = archive["method_ids"].astype(str).tolist()
        top_n_values = tuple(int(value) for value in archive["top_n_values"].tolist())
        seeds = tuple(int(value) for value in archive["seeds"].tolist())
        resolution_bp = int(archive["resolution_bp"])
        window_bins = int(archive["window_bins"])
        min_distance_bins = int(archive["min_distance_bins"])
        if {canonical_method(method) for method in source_methods} != set(IMPUTED_METHODS):
            raise ValueError("APA archive method set is incomplete")
        if top_n_values != APA_TOP_N_VALUES or seeds != SEEDS:
            raise ValueError("APA archive Top-N or seed order is unexpected")
        for source_method in source_methods:
            method = canonical_method(source_method)
            for top_n in top_n_values:
                for seed in seeds:
                    key = "{}__top{}__seed{}".format(source_method, top_n, seed)
                    split_matrices[(method, top_n, seed)] = np.asarray(
                        archive[key], dtype=float
                    )
    validate_apa_arrays(split_matrices)
    mean_matrices = {
        (method, top_n): np.mean(
            [split_matrices[(method, top_n, seed)] for seed in SEEDS], axis=0
        )
        for method in IMPUTED_METHODS
        for top_n in APA_TOP_N_VALUES
    }

    metrics = pd.read_csv(
        root / "apa_600kb_top10_top20_top50_metrics_with_flamingo.csv"
    )
    _require_columns(
        metrics,
        {"method", "top_n", "p2ll_mean", "p2ll_sd", "n_splits"},
        "APA metrics",
    )
    metrics = _canonicalize_method_column(metrics, "method")
    observed = set(zip(metrics["method_id"], metrics["top_n"].astype(int)))
    expected = {
        (method, top_n) for method in IMPUTED_METHODS for top_n in APA_TOP_N_VALUES
    }
    if observed != expected or len(metrics) != len(expected):
        raise ValueError("APA metrics must contain exactly 6 x 3 rows")
    if not np.isfinite(metrics[["p2ll_mean", "p2ll_sd"]].to_numpy(float)).all():
        raise ValueError("APA metrics contain non-finite P2LL values")
    return (
        split_matrices,
        mean_matrices,
        metrics.reset_index(drop=True),
        resolution_bp,
        window_bins,
        min_distance_bins,
    )


def load_heldout_support(data_dir):
    root = Path(data_dir) / "heldout_support"
    fraction = pd.read_csv(root / "support_fraction_600kb_data_with_flamingo.csv")
    counts = pd.read_csv(root / "panelB_600kb_raw_supported_counts_with_flamingo.csv")
    _require_columns(
        fraction,
        {
            "method",
            "requested_n",
            "actual_n_mean",
            "actual_n_std",
            "supported_fraction_mean",
            "supported_fraction_std",
        },
        "held-out support fraction",
    )
    _require_columns(
        counts,
        {"method", "supported_mean", "supported_sd", "eligible_mean", "eligible_sd"},
        "held-out supported counts",
    )
    fraction = _canonicalize_method_column(fraction, "method")
    counts = _canonicalize_method_column(counts, "method")
    expected_fraction = {
        (method, top_n) for method in IMPUTED_METHODS for top_n in TOP_N_VALUES
    }
    observed_fraction = set(
        zip(fraction["method_id"], fraction["requested_n"].astype(int))
    )
    if observed_fraction != expected_fraction or len(fraction) != len(expected_fraction):
        raise ValueError("held-out support fraction must contain exactly 6 x 5 rows")
    if set(counts["method_id"]) != set(IMPUTED_METHODS) or len(counts) != len(
        IMPUTED_METHODS
    ):
        raise ValueError("held-out supported counts must contain six methods")
    fraction_numeric = fraction[
        [
            "actual_n_mean",
            "actual_n_std",
            "supported_fraction_mean",
            "supported_fraction_std",
        ]
    ].to_numpy(float)
    count_numeric = counts[
        ["supported_mean", "supported_sd", "eligible_mean", "eligible_sd"]
    ].to_numpy(float)
    if not np.isfinite(fraction_numeric).all() or not np.isfinite(count_numeric).all():
        raise ValueError("held-out support tables contain non-finite values")
    return fraction.reset_index(drop=True), counts.reset_index(drop=True)


def load_figure_data(data_dir):
    """Verify and load the complete standalone formal plotting bundle."""

    data_dir = Path(data_dir).resolve()
    verify_copied_manifest(data_dir)
    umap_points, umap_summary, stage_metadata = load_umap(data_dir)
    (
        contact_matrices,
        contact_summits,
        contact_counts,
        contact_resolution_bp,
        contact_n_bins,
    ) = load_contact_loops(data_dir)
    loop_counts = load_loop_counts(data_dir)
    (
        apa_split_matrices,
        apa_matrices,
        apa_metrics,
        apa_resolution_bp,
        apa_window_bins,
        apa_min_distance_bins,
    ) = load_apa(data_dir)
    support_fraction, support_counts = load_heldout_support(data_dir)
    return FigureData(
        umap_points=umap_points,
        umap_summary=umap_summary,
        stage_metadata=stage_metadata,
        contact_matrices=contact_matrices,
        contact_summits=contact_summits,
        contact_counts=contact_counts,
        loop_counts=loop_counts,
        contact_resolution_bp=contact_resolution_bp,
        contact_n_bins=contact_n_bins,
        apa_split_matrices=apa_split_matrices,
        apa_matrices=apa_matrices,
        apa_metrics=apa_metrics,
        apa_resolution_bp=apa_resolution_bp,
        apa_window_bins=apa_window_bins,
        apa_min_distance_bins=apa_min_distance_bins,
        support_fraction=support_fraction,
        support_counts=support_counts,
    )
