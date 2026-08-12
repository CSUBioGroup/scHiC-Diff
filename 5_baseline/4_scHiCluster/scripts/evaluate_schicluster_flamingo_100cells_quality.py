#!/usr/bin/env python3
"""Evaluate scHiCluster imputation quality for FLAMINGO 100-cell data."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import pearsonr, spearmanr


DEFAULT_DATA_DIR = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/3_DiffusionModel/"
    "scHiC-Diff-master/data/SimuData/2_FLAMINGOData/100cells"
)
DEFAULT_WORK_DIR = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/0_scHiCDiff"
)
DEFAULT_OUTPUT_CSV = DEFAULT_WORK_DIR / "metrics" / "scHiCluster_FLAMINGO_100cells_quality_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate all/observed/heldout PCC and Spearman metrics for "
            "100-cell FLAMINGO scHiCluster imputation outputs."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--method", default="scHiCluster")
    parser.add_argument("--datasets", nargs="*", default=None)
    return parser.parse_args()


def load_npz_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return load_npz(path).toarray().astype(np.float64, copy=False)
    except ValueError:
        with np.load(path, allow_pickle=False) as data:
            if "data" not in data.files:
                raise ValueError(f"{path} is not scipy sparse npz and has no 'data' array")
            return np.asarray(data["data"], dtype=np.float64)


def safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    value = fn(x, y)
    if hasattr(value, "statistic"):
        return float(value.statistic)
    if isinstance(value, tuple):
        return float(value[0])
    return float(value)


def metric_block(prefix: str, pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(pred) & np.isfinite(truth)
    x = pred[valid]
    y = truth[valid]
    return {
        f"n_{prefix}": int(x.size),
        f"pcc_{prefix}": safe_corr(pearsonr, x, y),
        f"spearman_{prefix}": safe_corr(spearmanr, x, y),
    }


def evaluate_cell(pred: np.ndarray, truth: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    truth_mask = np.isfinite(truth) & (truth > 0)
    observed_mask = truth_mask & np.isfinite(observed) & (observed > 0)
    heldout_mask = truth_mask & ~observed_mask

    metrics: dict[str, float] = {
        "observed_frac": float(observed_mask.sum() / truth_mask.sum()) if truth_mask.sum() else float("nan")
    }
    metrics.update(metric_block("all", pred, truth, truth_mask))
    metrics.update(metric_block("observed", pred, truth, observed_mask))
    metrics.update(metric_block("heldout", pred, truth, heldout_mask))
    return metrics


def broadcast_truth(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if truth.shape == pred.shape:
        return truth
    if truth.shape[0] == 1 and truth.shape[1] == pred.shape[1]:
        return np.repeat(truth, pred.shape[0], axis=0)
    raise ValueError(f"Cannot broadcast truth shape {truth.shape} to prediction shape {pred.shape}")


def summarize(cell_rows: list[dict[str, float]]) -> dict[str, float]:
    df = pd.DataFrame(cell_rows)
    summary: dict[str, float] = {
        "n_cells": int(len(df)),
        "observed_frac_mean": float(df["observed_frac"].mean(skipna=True)),
        "observed_frac_std": float(df["observed_frac"].std(skipna=True)),
    }
    for col in df.columns:
        if col == "observed_frac":
            continue
        if col.startswith("n_"):
            summary[f"{col}_total"] = int(df[col].sum(skipna=True))
        else:
            summary[f"{col}_mean"] = float(df[col].mean(skipna=True))
            summary[f"{col}_std"] = float(df[col].std(skipna=True))
    return summary


def parse_dataset_name(dataset: str) -> dict[str, object]:
    match = re.fullmatch(r"beads_(?P<beads>\d+)_W_(?P<w>[0-9.]+)_level_(?P<level>\d+)_(?P<ctype>T[123])", dataset)
    if not match:
        return {"beads": np.nan, "w": "", "level": np.nan, "ctype": ""}
    return {
        "beads": int(match.group("beads")),
        "w": match.group("w"),
        "level": int(match.group("level")),
        "ctype": match.group("ctype"),
    }


def discover_datasets(gt_root: Path) -> list[str]:
    return sorted(path.stem for path in gt_root.glob("*.npz"))


def evaluate_dataset(dataset: str, data_dir: Path, work_dir: Path) -> dict[str, object]:
    gt_path = data_dir / "gt" / "1_lower_tri_feature" / "npz" / f"{dataset}.npz"
    sim_path = data_dir / "sim" / "1_lower_tri_feature" / "npz" / f"{dataset}.npz"
    pred_path = work_dir / "output" / "2_lower_tri_npz" / f"{dataset}_scHiCluster_imputed.npz"

    truth = load_npz_array(gt_path)
    observed = load_npz_array(sim_path)
    pred = load_npz_array(pred_path)
    truth = broadcast_truth(truth, pred)

    if truth.shape != observed.shape or truth.shape != pred.shape:
        raise ValueError(
            f"{dataset} shape mismatch: truth={truth.shape}, observed={observed.shape}, pred={pred.shape}"
        )

    cell_metrics = [
        evaluate_cell(pred[cell_id], truth[cell_id], observed[cell_id])
        for cell_id in range(pred.shape[0])
    ]
    return {
        "method": "scHiCluster",
        "data_name": dataset,
        **parse_dataset_name(dataset),
        "n_features": int(pred.shape[1]),
        **summarize(cell_metrics),
    }


def main() -> int:
    args = parse_args()
    gt_root = args.data_dir / "gt" / "1_lower_tri_feature" / "npz"
    datasets = args.datasets or discover_datasets(gt_root)
    if not datasets:
        raise ValueError(f"No gt npz datasets found under {gt_root}")

    rows = []
    for dataset in datasets:
        print(f"Evaluating {dataset}", flush=True)
        row = evaluate_dataset(dataset, args.data_dir, args.work_dir)
        row["method"] = args.method
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Saved results to: {args.output_csv.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
