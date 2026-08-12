#!/usr/bin/env python3
"""Evaluate Higashi imputation quality on HiCImputeData outputs."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_MANIFEST = BASE_DIR / "manifest.tsv"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output"
DEFAULT_METRICS_DIR = DEFAULT_OUTPUT_ROOT / "metrics"


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    sim_h5ad: str
    gt_npz: str
    n_cells: int
    n_features: int
    n_beads: int


def read_manifest(path: Path) -> list[DatasetRecord]:
    with path.open(newline="") as handle:
        return [
            DatasetRecord(
                dataset_id=row["dataset_id"],
                sim_h5ad=row["sim_h5ad"],
                gt_npz=row["gt_npz"],
                n_cells=int(row["n_cells"]),
                n_features=int(row["n_features"]),
                n_beads=int(row["n_beads"]),
            )
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def dataset_sort_key(dataset_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"K562_T([123])_(\d+)k", dataset_id)
    if not match:
        return (999999, 999999)
    timepoint, depth = match.groups()
    return (int(timepoint), int(depth))


def parse_dataset_id(dataset_id: str) -> dict[str, object]:
    match = re.fullmatch(r"K562_(T[123])_(\d+k)", dataset_id)
    if not match:
        return {"timepoint": "", "depth": ""}
    timepoint, depth = match.groups()
    return {"timepoint": timepoint, "depth": depth}


def load_h5ad_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if "X/data" in handle and "X/indices" in handle and "X/indptr" in handle:
            data = np.asarray(handle["X/data"], dtype=np.float64)
            indices = np.asarray(handle["X/indices"], dtype=np.int32)
            indptr = np.asarray(handle["X/indptr"], dtype=np.int32)
            n_cells = len(indptr) - 1
            n_features = len(handle["var/_index"])
            matrix = sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_features)).toarray()
        else:
            matrix = np.asarray(handle["X"], dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix[matrix < 0] = 0.0
    return matrix


def load_sparse_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    matrix = sparse.load_npz(path).toarray().astype(np.float64, copy=False)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix[matrix < 0] = 0.0
    return matrix


def prediction_path(output_root: Path, dataset_id: str, neighbor: int) -> Path:
    return output_root / "npz_lower_tri" / f"{dataset_id}_higashi_nbr_{neighbor}_lower_tri.npz"


def make_masks(truth: np.ndarray, observed: np.ndarray) -> dict[str, np.ndarray]:
    all_mask = np.isfinite(truth) & (truth > 0)
    observed_mask = all_mask & np.isfinite(observed) & (observed > 0)
    heldout_mask = all_mask & ~observed_mask
    return {
        "observed": observed_mask,
        "heldout": heldout_mask,
        "all": all_mask,
    }


def safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    result = fn(x, y)
    if hasattr(result, "statistic"):
        return float(result.statistic)
    if isinstance(result, tuple):
        return float(result[0])
    return float(result)


def metric_block(prefix: str, pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(pred) & np.isfinite(truth)
    x = pred[valid]
    y = truth[valid]
    if x.size == 0:
        return {
            f"n_{prefix}": 0,
            f"pcc_{prefix}": float("nan"),
            f"spearman_{prefix}": float("nan"),
            f"mae_{prefix}": float("nan"),
        }
    return {
        f"n_{prefix}": int(x.size),
        f"pcc_{prefix}": safe_corr(pearsonr, x, y),
        f"spearman_{prefix}": safe_corr(spearmanr, x, y),
        f"mae_{prefix}": float(np.mean(np.abs(x - y))),
    }


def evaluate_cell(pred: np.ndarray, truth: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    masks = make_masks(truth, observed)
    out: dict[str, float] = {
        "observed_fraction": float(masks["observed"].sum() / masks["all"].sum())
        if masks["all"].sum()
        else float("nan")
    }
    for prefix in ("observed", "heldout", "all"):
        out.update(metric_block(prefix, pred, truth, masks[prefix]))
    return out


def summarize_cells(cell_df: pd.DataFrame) -> dict[str, float]:
    summary: dict[str, float] = {"n_cells": int(len(cell_df))}
    for col in cell_df.columns:
        if col in {"dataset_id", "config", "neighbor", "cell_idx", "timepoint", "depth"}:
            continue
        if not pd.api.types.is_numeric_dtype(cell_df[col]):
            continue
        if col.startswith("n_"):
            summary[f"{col}_total"] = int(cell_df[col].sum(skipna=True))
        else:
            summary[f"{col}_mean"] = float(cell_df[col].mean(skipna=True))
            summary[f"{col}_std"] = float(cell_df[col].std(skipna=True))
    return summary


def evaluate_dataset(record: DatasetRecord, output_root: Path, neighbor: int) -> tuple[pd.DataFrame, dict[str, object]]:
    config = f"Higashi {neighbor} nbr"
    pred = load_sparse_matrix(prediction_path(output_root, record.dataset_id, neighbor))
    observed = load_h5ad_matrix(Path(record.sim_h5ad))
    truth = load_sparse_matrix(Path(record.gt_npz))
    if truth.shape[0] == 1:
        truth = np.repeat(truth, record.n_cells, axis=0)
    if pred.shape != observed.shape or pred.shape != truth.shape:
        raise ValueError(
            f"{record.dataset_id} shape mismatch for nbr{neighbor}: "
            f"pred={pred.shape}, observed={observed.shape}, truth={truth.shape}"
        )
    if pred.shape != (record.n_cells, record.n_features):
        raise ValueError(f"{record.dataset_id} unexpected prediction shape for nbr{neighbor}: {pred.shape}")

    rows = []
    meta = parse_dataset_id(record.dataset_id)
    for cell_idx in range(record.n_cells):
        rows.append(
            {
                "dataset_id": record.dataset_id,
                "config": config,
                "neighbor": neighbor,
                "cell_idx": cell_idx,
                **meta,
                **evaluate_cell(pred[cell_idx], truth[cell_idx], observed[cell_idx]),
            }
        )
    cell_df = pd.DataFrame(rows)
    summary = {
        "dataset_id": record.dataset_id,
        "config": config,
        "neighbor": neighbor,
        **meta,
        "n_beads": record.n_beads,
        "n_features": record.n_features,
        **summarize_cells(cell_df),
    }
    return cell_df, summary


def make_requested_metric_rows(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    metric_labels = [
        ("observed pcc", "pcc_observed_mean"),
        ("observed spearman", "spearman_observed_mean"),
        ("heldout pcc", "pcc_heldout_mean"),
        ("heldout spearman", "spearman_heldout_mean"),
        ("all pcc", "pcc_all_mean"),
        ("all spearman", "spearman_all_mean"),
    ]
    rows = []
    for dataset_id, group in dataset_summary.groupby("dataset_id", sort=False):
        for label, col in metric_labels:
            row: dict[str, object] = {"dataset_id": dataset_id, "metric": label}
            for _, item in group.sort_values("neighbor").iterrows():
                row[item["config"]] = item[col]
            rows.append(row)
    return pd.DataFrame(rows)


def make_requested_wide(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    metric_labels = [
        ("observed pcc", "pcc_observed_mean"),
        ("observed spearman", "spearman_observed_mean"),
        ("heldout pcc", "pcc_heldout_mean"),
        ("heldout spearman", "spearman_heldout_mean"),
        ("all pcc", "pcc_all_mean"),
        ("all spearman", "spearman_all_mean"),
    ]
    rows = []
    for dataset_id, group in dataset_summary.groupby("dataset_id", sort=False):
        row: dict[str, object] = {"dataset_id": dataset_id}
        for label, col in metric_labels:
            for _, item in group.sort_values("neighbor").iterrows():
                row[f"{item['config']} {label}"] = item[col]
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--neighbors", type=int, nargs="+", default=[0, 5])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = read_manifest(args.manifest)
    records = sorted(records, key=lambda record: dataset_sort_key(record.dataset_id))
    if args.datasets:
        keep = set(args.datasets)
        records = [record for record in records if record.dataset_id in keep]
    if not records:
        raise ValueError("No datasets selected")

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    cell_frames = []
    summary_rows = []
    for record in records:
        for neighbor in args.neighbors:
            print(f"Evaluating {record.dataset_id} Higashi nbr{neighbor}", flush=True)
            cell_df, summary = evaluate_dataset(record, args.output_root, neighbor)
            cell_frames.append(cell_df)
            summary_rows.append(summary)

    cell_metrics = pd.concat(cell_frames, ignore_index=True)
    dataset_summary = pd.DataFrame(summary_rows)
    metric_rows = make_requested_metric_rows(dataset_summary)
    wide = make_requested_wide(dataset_summary)

    cell_path = args.metrics_dir / "higashi_hicimpute_cell_metrics.csv"
    dataset_path = args.metrics_dir / "higashi_hicimpute_dataset_summary.csv"
    metric_rows_path = args.metrics_dir / "higashi_hicimpute_requested_metric_rows.csv"
    wide_path = args.metrics_dir / "higashi_hicimpute_requested_wide.csv"

    cell_metrics.to_csv(cell_path, index=False)
    dataset_summary.to_csv(dataset_path, index=False)
    metric_rows.to_csv(metric_rows_path, index=False)
    wide.to_csv(wide_path, index=False)

    print(f"Saved cell metrics: {cell_path.resolve()}", flush=True)
    print(f"Saved dataset summary: {dataset_path.resolve()}", flush=True)
    print(f"Saved requested metric-row table: {metric_rows_path.resolve()}", flush=True)
    print(f"Saved requested wide table: {wide_path.resolve()}", flush=True)
    print(metric_rows.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
