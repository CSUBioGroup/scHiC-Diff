#!/usr/bin/env python3
"""Evaluate scHiCluster imputation quality for Simu_Data (HiCImpute benchmark).

For each of the 12 K562_T{1,2,3}_{1k,2k,4k,7k} datasets we compare the
scHiCluster imputed contacts against the ground truth stored in the dxy
processed NPZ files.  Metrics are computed cell-wise on
``log1p(max(x, 0))`` transformed contacts, following the v3 FLAMINGO
convention:

  * log1p Pearson (PCC) mean / std
  * log1p Spearman (SCC) mean / std
  * log1p MAE mean / std

Three evaluation masks are reported:
  * all      - every feature where GT > 0
  * observed - features present in the input (sim > 0) and GT > 0
  * heldout  - features absent from the input (sim == 0) but GT > 0

Ground truth NPZ files (dxy version, verified identical to
``0_gtData/1_Gt_HiCImputeData``):
  ``{sim_dir}/{dataset}_true.npz``  — (100, 1830) CSR float64
  ``{sim_dir}/{dataset}_sim.npz``   — (100, 1830) CSR int64

Predictions are loaded from the per-dataset feature NPZ written by
``05_collect_simu_hdf5.py`` (shape ``(100, 1830)``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import pearsonr, spearmanr


DEFAULT_GT_DIR = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "0_gtData/1_Gt_HiCImputeData"
)
DEFAULT_SIM_DIR = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/"
    "1-HiCImpute_Simulation_Data/2_processed_data_dxy/"
    "1_lower_tri_feature/npz"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "4_scHiCluster/1_HiCImputeDate/output"
)
DEFAULT_OUTPUT_CSV = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "4_scHiCluster/1_HiCImputeDate/output/metrics/"
    "scHiCluster_Simu_Data_quality_metrics.csv"
)

DATASETS = [f"K562_T{t}_{d}" for t in (1, 2, 3) for d in ("1k", "2k", "4k", "7k")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate cell-wise PCC/Spearman/MAE for Simu_Data scHiCluster imputation."
    )
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--method", default="scHiCluster")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--append", action="store_true",
                        help="Append results to the output CSV instead of overwriting")
    return parser.parse_args()


def load_npz_dense(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return load_npz(path).toarray().astype(np.float64, copy=False)


def safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    value = fn(x, y)
    if hasattr(value, "statistic"):
        return float(value.statistic)
    if isinstance(value, tuple):
        return float(value[0])
    return float(value)


def cell_metrics(pred_row: np.ndarray, gt_row: np.ndarray,
                 sim_row: np.ndarray) -> dict:
    gt_mask = np.isfinite(gt_row) & (gt_row > 0)
    obs_mask = gt_mask & (sim_row > 0)
    held_mask = gt_mask & (sim_row == 0)
    out: dict = {
        "n_all": int(gt_mask.sum()),
        "n_observed": int(obs_mask.sum()),
        "n_heldout": int(held_mask.sum()),
        "observed_frac": float(obs_mask.sum() / gt_mask.sum()) if gt_mask.sum() else float("nan"),
    }
    for tag, mask in (("all", gt_mask), ("observed", obs_mask), ("heldout", held_mask)):
        p = np.log1p(np.maximum(pred_row[mask], 0.0))
        g = np.log1p(np.maximum(gt_row[mask], 0.0))
        out[f"pcc_{tag}"] = safe_corr(pearsonr, p, g)
        out[f"spearman_{tag}"] = safe_corr(spearmanr, p, g)
        out[f"mae_{tag}"] = float(np.mean(np.abs(p - g))) if p.size else float("nan")
    return out


def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    summary: dict = {"n_cells": int(len(df))}
    for col in df.columns:
        if col.startswith("n_"):
            summary[f"{col}_total"] = int(df[col].sum(skipna=True))
        elif col == "observed_frac":
            summary[f"{col}_mean"] = float(df[col].mean(skipna=True))
            summary[f"{col}_std"] = float(df[col].std(skipna=True))
        else:
            summary[f"{col}_mean"] = float(df[col].mean(skipna=True))
            summary[f"{col}_std"] = float(df[col].std(skipna=True))
    return summary


def evaluate_dataset(dataset: str, args: argparse.Namespace) -> dict:
    gt_path = args.gt_dir / f"{dataset}_true.npz"
    sim_path = args.sim_dir / f"{dataset}_sim.npz"
    pred_path = args.output_root / "2_lower_tri_npz" / f"{dataset}_scHiCluster_imputed.npz"

    print(f"[eval] {dataset}: loading GT {gt_path.name}", flush=True)
    gt = load_npz_dense(gt_path)
    print(f"[eval] {dataset}: loading sim {sim_path.name}", flush=True)
    sim = load_npz_dense(sim_path)
    print(f"[eval] {dataset}: loading pred {pred_path.name}", flush=True)
    pred = load_npz_dense(pred_path)

    if pred.shape != gt.shape:
        raise ValueError(f"{dataset} pred {pred.shape} != gt {gt.shape}")
    if sim.shape != gt.shape:
        raise ValueError(f"{dataset} sim {sim.shape} != gt {gt.shape}")

    rows = []
    for c in range(pred.shape[0]):
        rows.append(cell_metrics(pred[c], gt[c], sim[c]))
    summ = summarize(rows)
    return {
        "method": args.method,
        "data_name": dataset,
        "n_cells": pred.shape[0],
        "n_features": int(pred.shape[1]),
        "gt_file": str(gt_path),
        "sim_file": str(sim_path),
        "pred_file": str(pred_path),
        **summ,
    }


def main() -> int:
    args = parse_args()
    datasets = args.datasets or list(DATASETS)
    if not datasets:
        raise ValueError("No datasets specified")

    rows = []
    for ds in datasets:
        row = evaluate_dataset(ds, args)
        rows.append(row)
        print(f"[eval] {ds}: PCC all={row['pcc_all_mean']:.4f} "
              f"obs={row['pcc_observed_mean']:.4f} "
              f"held={row['pcc_heldout_mean']:.4f}",
              flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if args.append and args.output_csv.exists():
        old_df = pd.read_csv(args.output_csv)
        combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)
        cols = list(old_df.columns) + [c for c in new_df.columns if c not in old_df.columns]
        combined = combined.reindex(columns=cols)
        combined.to_csv(args.output_csv, index=False)
        print(f"[eval] appended {len(new_df)} rows -> {args.output_csv} "
              f"(now {len(combined)} rows)", flush=True)
    else:
        new_df.to_csv(args.output_csv, index=False)
        print(f"[eval] saved {args.output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
