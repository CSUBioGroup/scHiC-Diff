#!/usr/bin/env python3
"""Evaluate ScUnicorn FLAMINGO 10-fold imputation quality."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

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
    sim_npz: str
    gt_npz: str
    n_cells: int
    n_features: int
    n_beads: int


def read_manifest(path: Path) -> list[DatasetRecord]:
    with path.open(newline="") as handle:
        return [
            DatasetRecord(
                dataset_id=row["dataset_id"],
                sim_npz=row["sim_npz"],
                gt_npz=row["gt_npz"],
                n_cells=int(row["n_cells"]),
                n_features=int(row["n_features"]),
                n_beads=int(row["n_beads"]),
            )
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def parse_dataset_id(dataset_id: str) -> dict[str, object]:
    match = re.fullmatch(r"beads_(\d+)_W_([0-9.]+)_level_(\d+)_T(\d+)", dataset_id)
    if not match:
        return {"beads": np.nan, "w": np.nan, "level": np.nan, "timepoint": ""}
    beads, width, level, timepoint = match.groups()
    return {
        "beads": int(beads),
        "w": float(width),
        "level": int(level),
        "timepoint": f"T{timepoint}",
    }


def load_source_npz(path: Path) -> np.ndarray:
    archive = np.load(path, allow_pickle=True)
    data = np.asarray(archive["data"], dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"{path} data must be 2D, got {data.shape}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data[data < 0] = 0.0
    return data


def load_prediction_npz(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    data = sparse.load_npz(path).toarray().astype(np.float64, copy=False)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data[data < 0] = 0.0
    return data


def prediction_path(output_root: Path, dataset_id: str) -> Path:
    return (
        output_root
        / "combined_10fold"
        / "npz_lower_tri"
        / f"{dataset_id}_scunicorn_10fold_lower_tri.npz"
    )


def make_masks(truth: np.ndarray, observed: np.ndarray) -> dict[str, np.ndarray]:
    all_mask = np.isfinite(truth) & (truth > 0)
    observed_mask = all_mask & np.isfinite(observed) & (observed > 0)
    heldout_mask = all_mask & ~observed_mask
    return {
        "all": all_mask,
        "observed": observed_mask,
        "heldout": heldout_mask,
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
            f"rmse_{prefix}": float("nan"),
            f"log1p_pcc_{prefix}": float("nan"),
            f"log1p_spearman_{prefix}": float("nan"),
        }
    diff = x - y
    log_x = np.log1p(np.maximum(x, 0.0))
    log_y = np.log1p(np.maximum(y, 0.0))
    return {
        f"n_{prefix}": int(x.size),
        f"pcc_{prefix}": safe_corr(pearsonr, x, y),
        f"spearman_{prefix}": safe_corr(spearmanr, x, y),
        f"mae_{prefix}": float(np.mean(np.abs(diff))),
        f"rmse_{prefix}": float(np.sqrt(np.mean(diff**2))),
        f"log1p_pcc_{prefix}": safe_corr(pearsonr, log_x, log_y),
        f"log1p_spearman_{prefix}": safe_corr(spearmanr, log_x, log_y),
    }


def evaluate_cell(pred: np.ndarray, truth: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    masks = make_masks(truth, observed)
    truth_count = int(masks["all"].sum())
    out: dict[str, float] = {
        "observed_fraction": float(masks["observed"].sum() / truth_count)
        if truth_count
        else float("nan")
    }
    for prefix in ("heldout", "observed", "all"):
        out.update(metric_block(prefix, pred, truth, masks[prefix]))
    return out


def summarize_cells(cell_df: pd.DataFrame) -> dict[str, float]:
    summary: dict[str, float] = {"n_cells": int(len(cell_df))}
    for col in cell_df.columns:
        if col in {"dataset_id", "cell_idx", "timepoint"}:
            continue
        if not pd.api.types.is_numeric_dtype(cell_df[col]):
            continue
        if col.startswith("n_"):
            summary[f"{col}_total"] = int(cell_df[col].sum(skipna=True))
        else:
            summary[f"{col}_mean"] = float(cell_df[col].mean(skipna=True))
            summary[f"{col}_std"] = float(cell_df[col].std(skipna=True))
    return summary


def evaluate_dataset(record: DatasetRecord, output_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    pred = load_prediction_npz(prediction_path(output_root, record.dataset_id))
    observed = load_source_npz(Path(record.sim_npz))
    gt = load_source_npz(Path(record.gt_npz))
    if gt.shape[0] == 1:
        truth = np.repeat(gt, record.n_cells, axis=0)
    elif gt.shape[0] == record.n_cells:
        truth = gt
    else:
        raise ValueError(f"Unexpected gt rows for {record.dataset_id}: {gt.shape}")
    if pred.shape != observed.shape or pred.shape != truth.shape:
        raise ValueError(
            f"{record.dataset_id} shape mismatch: pred={pred.shape}, observed={observed.shape}, truth={truth.shape}"
        )
    if pred.shape != (record.n_cells, record.n_features):
        raise ValueError(f"{record.dataset_id} unexpected pred shape: {pred.shape}")

    rows = []
    meta = parse_dataset_id(record.dataset_id)
    for cell_idx in range(record.n_cells):
        rows.append(
            {
                "dataset_id": record.dataset_id,
                "cell_idx": cell_idx,
                **meta,
                **evaluate_cell(pred[cell_idx], truth[cell_idx], observed[cell_idx]),
            }
        )
    cell_df = pd.DataFrame(rows)
    summary = {
        "dataset_id": record.dataset_id,
        **meta,
        "n_beads": record.n_beads,
        "n_features": record.n_features,
        **summarize_cells(cell_df),
    }
    return cell_df, summary


def summarize_overall(summary_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in summary_df.columns
        if col not in {"dataset_id", "timepoint"}
        and pd.api.types.is_numeric_dtype(summary_df[col])
    ]
    row: dict[str, object] = {"method": "ScUnicorn", "n_datasets": int(len(summary_df))}
    for col in numeric_cols:
        row[f"{col}_mean_across_datasets"] = float(summary_df[col].mean(skipna=True))
        row[f"{col}_std_across_datasets"] = float(summary_df[col].std(skipna=True))
    return pd.DataFrame([row])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--datasets", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = read_manifest(args.manifest)
    if args.datasets:
        keep = set(args.datasets)
        records = [record for record in records if record.dataset_id in keep]
    if not records:
        raise ValueError("No datasets selected")

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    cell_frames = []
    summary_rows = []
    for record in records:
        print(f"Evaluating {record.dataset_id}", flush=True)
        cell_df, summary = evaluate_dataset(record, args.output_root)
        cell_frames.append(cell_df)
        summary_rows.append(summary)

    cell_metrics = pd.concat(cell_frames, ignore_index=True)
    dataset_summary = pd.DataFrame(summary_rows)
    overall_summary = summarize_overall(dataset_summary)

    cell_path = args.metrics_dir / "scunicorn_flamingo_cell_metrics.csv"
    dataset_path = args.metrics_dir / "scunicorn_flamingo_dataset_summary.csv"
    overall_path = args.metrics_dir / "scunicorn_flamingo_overall_summary.csv"
    requested_path = args.metrics_dir / "scunicorn_flamingo_requested_pcc_spearman.csv"

    cell_metrics.to_csv(cell_path, index=False)
    dataset_summary.to_csv(dataset_path, index=False)
    overall_summary.to_csv(overall_path, index=False)
    requested_cols = [
        "dataset_id",
        "beads",
        "w",
        "level",
        "timepoint",
        "pcc_heldout_mean",
        "spearman_heldout_mean",
        "pcc_observed_mean",
        "spearman_observed_mean",
        "pcc_all_mean",
        "spearman_all_mean",
    ]
    dataset_summary[requested_cols].to_csv(requested_path, index=False)

    print(f"Saved cell metrics: {cell_path.resolve()}", flush=True)
    print(f"Saved dataset summary: {dataset_path.resolve()}", flush=True)
    print(f"Saved overall summary: {overall_path.resolve()}", flush=True)
    print(f"Saved requested PCC/Spearman table: {requested_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
