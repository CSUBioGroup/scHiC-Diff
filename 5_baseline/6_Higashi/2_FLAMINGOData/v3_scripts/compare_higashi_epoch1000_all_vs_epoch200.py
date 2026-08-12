#!/usr/bin/env python3
"""Compare all FLAMINGO v3 Higashi updates=1000 outputs against updates=200 baseline."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import load_npz


ROOT = Path("/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi")
DATA_DIR = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/"
    "5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/"
    "3_fixed_flamingoGen_datasets/5_paramsweep_datasets"
)
BASELINE_OUTPUT_ROOT = ROOT / "2_FLAMINGOData/v3_outputData"
EXPERIMENT_OUTPUT_ROOT = ROOT / "2_FLAMINGOData/v3_epoch1000_outputData"
DATASETS = (
    "v3_hybrid_W0p5_500cells_level0",
    "v3_hybrid_W0p6_500cells_level0",
    "v3_hybrid_W0p7_500cells_level0",
    "v3_hybrid_W0p7_500cells_level0_r0p01",
    "v3_hybrid_W0p7_500cells_level0_r0p05",
    "v3_hybrid_W0p8_500cells_level0",
    "v3_hybrid_W0p9_500cells_level0",
)
TASKS = tuple((stem, neighbor) for stem in DATASETS for neighbor in (0, 5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--baseline-output-root", type=Path, default=BASELINE_OUTPUT_ROOT)
    parser.add_argument("--experiment-output-root", type=Path, default=EXPERIMENT_OUTPUT_ROOT)
    parser.add_argument("--skip-pred-delta", action="store_true")
    return parser.parse_args()


def pred_path(output_root: Path, stem: str, neighbor: int) -> Path:
    return output_root / "npz_lower_tri" / f"{stem}_higashi_nbr_{neighbor}_lower_tri.npz"


def safe_pcc(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def csr_group_row(group: h5py.Group, row_idx: int, n_features: int) -> np.ndarray:
    start = int(group["indptr"][row_idx])
    end = int(group["indptr"][row_idx + 1])
    row = np.zeros(n_features, dtype=np.float64)
    row[group["indices"][start:end]] = group["data"][start:end]
    return row


def compute_raw_metrics(pred_npz: Path, gt_h5ad: Path) -> dict[str, float | int]:
    pred = load_npz(pred_npz).tocsr()
    n_cells, n_features = pred.shape
    pccs: list[float] = []
    maes: list[float] = []
    with h5py.File(gt_h5ad, "r") as handle:
        gt_group = handle["layers"]["gt"]
        gt_cells = len(gt_group["indptr"]) - 1
        if gt_cells != n_cells:
            raise ValueError(f"{pred_npz.name}: pred rows {n_cells}, gt rows {gt_cells}")
        for row_idx in range(n_cells):
            true = csr_group_row(gt_group, row_idx, n_features)
            result = pred.getrow(row_idx).toarray().ravel().astype(np.float64, copy=False)
            pccs.append(safe_pcc(true, result))
            maes.append(float(np.mean(np.abs(true - result))))
    pccs_arr = np.asarray(pccs, dtype=np.float64)
    maes_arr = np.asarray(maes, dtype=np.float64)
    return {
        "n_cells": int(n_cells),
        "n_features": int(n_features),
        "raw_pcc_mean": float(np.nanmean(pccs_arr)),
        "raw_pcc_std": float(np.nanstd(pccs_arr)),
        "raw_mae_mean": float(np.nanmean(maes_arr)),
        "raw_mae_std": float(np.nanstd(maes_arr)),
    }


def compute_prediction_delta(baseline_npz: Path, experiment_npz: Path) -> dict[str, float | int]:
    base = load_npz(baseline_npz).tocsr()
    exp = load_npz(experiment_npz).tocsr()
    if base.shape != exp.shape:
        raise ValueError(f"shape mismatch {baseline_npz.name}: {base.shape} vs {exp.shape}")
    n_cells, n_features = base.shape
    total_entries = n_cells * n_features
    abs_sum = 0.0
    sq_sum = 0.0
    base_sum = 0.0
    exp_sum = 0.0
    pccs: list[float] = []
    for row_idx in range(n_cells):
        b = base.getrow(row_idx).toarray().ravel().astype(np.float64, copy=False)
        e = exp.getrow(row_idx).toarray().ravel().astype(np.float64, copy=False)
        diff = e - b
        abs_sum += float(np.abs(diff).sum())
        sq_sum += float(np.dot(diff, diff))
        base_sum += float(b.sum())
        exp_sum += float(e.sum())
        pccs.append(safe_pcc(b, e))
    return {
        "pred_mean_200": base_sum / total_entries,
        "pred_mean_1000": exp_sum / total_entries,
        "pred_mean_delta": (exp_sum - base_sum) / total_entries,
        "pred_mae_1000_vs_200": abs_sum / total_entries,
        "pred_rmse_1000_vs_200": math.sqrt(sq_sum / total_entries),
        "pred_pcc_1000_vs_200_mean": float(np.nanmean(np.asarray(pccs, dtype=np.float64))),
    }


def load_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def pick_one(df: pd.DataFrame, mask, label: str) -> pd.Series:
    sub = df.loc[mask]
    if len(sub) != 1:
        raise ValueError(f"{label}: expected 1 row, found {len(sub)}")
    return sub.iloc[0]


def aggregate_log1p_task_metrics(experiment_root: Path, metrics_dir: Path) -> pd.DataFrame:
    task_dir = experiment_root / "metrics/log1p_tasks"
    files = sorted(task_dir.glob("*_log1p_metrics.csv"))
    if len(files) != len(TASKS):
        raise RuntimeError(f"expected {len(TASKS)} log1p task CSVs in {task_dir}, found {len(files)}")
    df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True, sort=False)
    out = metrics_dir / "higashi_epoch1000_all_log1p_metrics.csv"
    df.to_csv(out, index=False)
    print(f"[compare-all] saved {out} ({len(df)} rows)", flush=True)
    return df


def main() -> int:
    args = parse_args()
    metrics_dir = args.experiment_output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    baseline_raw = load_required_csv(
        args.baseline_output_root / "metrics/higashi_FLAMINGO_v3_paramsweep_raw_gt_PCC_MAE.csv"
    )
    baseline_log1p = load_required_csv(
        args.baseline_output_root / "metrics/scHiCluster_FLAMINGO_v3_paramsweep_quality_metrics.csv"
    )
    experiment_log1p = aggregate_log1p_task_metrics(args.experiment_output_root, metrics_dir)

    raw_rows = []
    summary_rows = []
    for stem, neighbor in TASKS:
        print(f"[compare-all] {stem} nbr{neighbor}", flush=True)
        exp_npz = pred_path(args.experiment_output_root, stem, neighbor)
        base_npz = pred_path(args.baseline_output_root, stem, neighbor)
        gt_h5ad = args.data_dir / f"{stem}_scdiff2.h5ad"
        for required in (exp_npz, base_npz, gt_h5ad):
            if not required.exists():
                raise FileNotFoundError(required)

        exp_raw = compute_raw_metrics(exp_npz, gt_h5ad)
        raw_rows.append({
            "method": "Higashi",
            "config_tag": f"updates1000_nbr_{neighbor}",
            "dataset": stem,
            "h5ad": gt_h5ad.name,
            "imputed_file": exp_npz.name,
            "transform": "raw_gt_and_raw_prediction",
            "neighbor": neighbor,
            **exp_raw,
        })

        base_raw = pick_one(
            baseline_raw,
            (baseline_raw["dataset"] == stem) & (baseline_raw["neighbor"].astype(int) == neighbor),
            f"baseline raw {stem} nbr{neighbor}",
        )
        base_log = pick_one(
            baseline_log1p,
            (baseline_log1p["data_name"] == stem) & (baseline_log1p["config_tag"] == f"{neighbor}nbr"),
            f"baseline log1p {stem} nbr{neighbor}",
        )
        exp_log = pick_one(
            experiment_log1p,
            (experiment_log1p["data_name"] == stem)
            & (experiment_log1p["config_tag"] == f"updates1000_{neighbor}nbr"),
            f"experiment log1p {stem} nbr{neighbor}",
        )

        row = {
            "dataset": stem,
            "neighbor": neighbor,
            "raw_pcc_200": float(base_raw["pcc_mean"]),
            "raw_pcc_1000": float(exp_raw["raw_pcc_mean"]),
            "raw_pcc_delta_1000_minus_200": float(exp_raw["raw_pcc_mean"] - base_raw["pcc_mean"]),
            "raw_mae_200": float(base_raw["mae_mean"]),
            "raw_mae_1000": float(exp_raw["raw_mae_mean"]),
            "raw_mae_delta_1000_minus_200": float(exp_raw["raw_mae_mean"] - base_raw["mae_mean"]),
            "log1p_pcc_all_200": float(base_log["pcc_all_mean"]),
            "log1p_pcc_all_1000": float(exp_log["pcc_all_mean"]),
            "log1p_pcc_all_delta_1000_minus_200": float(exp_log["pcc_all_mean"] - base_log["pcc_all_mean"]),
            "log1p_mae_all_200": float(base_log["mae_all_mean"]),
            "log1p_mae_all_1000": float(exp_log["mae_all_mean"]),
            "log1p_mae_all_delta_1000_minus_200": float(exp_log["mae_all_mean"] - base_log["mae_all_mean"]),
            "log1p_pcc_heldout_200": float(base_log["pcc_heldout_mean"]),
            "log1p_pcc_heldout_1000": float(exp_log["pcc_heldout_mean"]),
            "log1p_pcc_heldout_delta_1000_minus_200": float(exp_log["pcc_heldout_mean"] - base_log["pcc_heldout_mean"]),
            "log1p_mae_heldout_200": float(base_log["mae_heldout_mean"]),
            "log1p_mae_heldout_1000": float(exp_log["mae_heldout_mean"]),
            "log1p_mae_heldout_delta_1000_minus_200": float(exp_log["mae_heldout_mean"] - base_log["mae_heldout_mean"]),
        }
        if not args.skip_pred_delta:
            row.update(compute_prediction_delta(base_npz, exp_npz))
        summary_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)
    raw_out = metrics_dir / "higashi_epoch1000_all_raw_gt_PCC_MAE.csv"
    raw_df.to_csv(raw_out, index=False)
    print(f"[compare-all] saved {raw_out} ({len(raw_df)} rows)", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary_out = metrics_dir / "higashi_epoch1000_all_vs_epoch200_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"[compare-all] saved {summary_out} ({len(summary)} rows)", flush=True)
    show_cols = [
        "dataset",
        "neighbor",
        "raw_pcc_delta_1000_minus_200",
        "raw_mae_delta_1000_minus_200",
        "log1p_pcc_all_delta_1000_minus_200",
        "log1p_mae_all_delta_1000_minus_200",
        "log1p_pcc_heldout_delta_1000_minus_200",
        "log1p_mae_heldout_delta_1000_minus_200",
    ]
    print(summary[show_cols].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
