#!/usr/bin/env python3
"""Diagnose HiCImpute v3 output vs GT and (re)compute PCC/MAE metrics.

Investigates why previously computed PCC is near 0:
  1. npz prediction value distribution (constant? range? sparsity?)
  2. h5ad GT layer value distribution
  3. Per-cell variance of prediction (zero variance -> PCC undefined/near 0)
  4. Direct sample-cell comparison
  5. Full cell-wise PCC/MAE for raw and log1p transforms

Outputs:
  <metrics_dir>/HiCImpute_FLAMINGO_v3_raw.csv
  <metrics_dir>/HiCImpute_FLAMINGO_v3_log1p.csv
  <metrics_dir>/diagnostics_report.txt
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import load_npz


def csr_group_row(group: h5py.Group, row_idx: int, n_features: int) -> np.ndarray:
    indptr = group["indptr"]
    start = int(indptr[row_idx])
    end = int(indptr[row_idx + 1])
    row = np.zeros(n_features, dtype=np.float64)
    row[group["indices"][start:end]] = group["data"][start:end]
    return row


def safe_pcc(a: np.ndarray, b: np.ndarray) -> float:
    sa = a.std()
    sb = b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def diagnose_one(pred_path: Path, gt_path: Path, log_handle) -> Dict:
    name = pred_path.name
    log = lambda m: (print(m, flush=True), log_handle.write(m + "\n"))
    log(f"\n{'='*70}")
    log(f"DIAGNOSE: {name}")
    log(f"GT:       {gt_path.name}")
    log(f"{'='*70}")

    pred_csr = load_npz(pred_path).tocsr()
    n_cells, n_features = pred_csr.shape
    log(f"pred shape: ({n_cells}, {n_features})  nnz={pred_csr.nnz}  "
        f"density={pred_csr.nnz/(n_cells*n_features):.4f}")

    with h5py.File(gt_path, "r") as f:
        gt_group = f["layers"]["gt"]
        gt_n_cells = len(gt_group["indptr"]) - 1
        gt_nnz = int(gt_group["indptr"][gt_n_cells])
        log(f"gt shape:   ({gt_n_cells}, {n_features})  nnz={gt_nnz}  "
            f"density={gt_nnz/(gt_n_cells*n_features):.4f}")
        if gt_n_cells != n_cells:
            log(f"ERROR: cell count mismatch {n_cells} vs {gt_n_cells}")
            return {}

        # sample prediction rows
        log("\n-- prediction row stats (first 5 + 3 random cells) --")
        sample_idx = list(range(5)) + list(np.random.choice(n_cells, 3, replace=False).tolist())
        pred_global_vals = []
        gt_global_vals = []
        for ci in sample_idx:
            prow = pred_csr.getrow(ci).toarray().ravel().astype(np.float64)
            grow = csr_group_row(gt_group, ci, n_features)
            pred_global_vals.append(prow)
            gt_global_vals.append(grow)
            log(f"  cell {ci:4d}: pred min={prow.min():.4g} max={prow.max():.4g} "
                f"mean={prow.mean():.4g} std={prow.std():.4g} nnz={int((prow!=0).sum()):6d} | "
                f"gt min={grow.min():.4g} max={grow.max():.4g} mean={grow.mean():.4g} "
                f"std={grow.std():.4g} nnz={int((grow!=0).sum()):6d} | "
                f"pcc={safe_pcc(prow, grow):.6f}")

        # Check: is prediction constant across features per cell?
        log("\n-- prediction per-cell std distribution (sample 200 cells) --")
        sample200 = np.random.choice(n_cells, min(200, n_cells), replace=False)
        stds = []
        for ci in sample200:
            prow = pred_csr.getrow(ci).toarray().ravel().astype(np.float64)
            stds.append(prow.std())
        stds = np.asarray(stds)
        log(f"  pred per-cell std: min={stds.min():.6g} q25={np.percentile(stds,25):.6g} "
            f"median={np.median(stds):.6g} q75={np.percentile(stds,75):.6g} max={stds.max():.6g}")
        n_const = int((stds < 1e-9).sum())
        log(f"  cells with near-zero pred std (<1e-9): {n_const}/{len(stds)}")

        # Global value stats over a sample
        pred_arr = np.concatenate(pred_global_vals)
        gt_arr = np.concatenate(gt_global_vals)
        log(f"\n-- pooled sample value ranges (8 cells) --")
        log(f"  pred: min={pred_arr.min():.4g} max={pred_arr.max():.4g} "
            f"mean={pred_arr.mean():.4g} std={pred_arr.std():.4g}")
        log(f"  gt:   min={gt_arr.min():.4g} max={gt_arr.max():.4g} "
            f"mean={gt_arr.mean():.4g} std={gt_arr.std():.4g}")

        # Check unique values in prediction (is it quantized/constant?)
        uniq = np.unique(pred_arr)
        log(f"  pred unique values in sample: {len(uniq)}  (first 10: {uniq[:10]})")

        # Full cell-wise PCC/MAE for raw and log1p
        log("\n-- computing full cell-wise metrics (raw + log1p) --")
        t0 = time.time()
        pcc_raw, pcc_log, mae_raw, mae_log = [], [], [], []
        for ci in range(n_cells):
            prow = pred_csr.getrow(ci).toarray().ravel().astype(np.float64)
            grow = csr_group_row(gt_group, ci, n_features)
            pcc_raw.append(safe_pcc(prow, grow))
            mae_raw.append(mae(prow, grow))
            if grow.min() >= -1 and prow.min() >= -1:
                pcc_log.append(safe_pcc(np.log1p(prow), np.log1p(grow)))
                mae_log.append(mae(np.log1p(prow), np.log1p(grow)))
            else:
                pcc_log.append(float("nan"))
                mae_log.append(float("nan"))
            if (ci + 1) % 300 == 0:
                log(f"  ... {ci+1}/{n_cells} cells "
                    f"({(time.time()-t0)/60:.1f} min)")

    pcc_raw = np.asarray(pcc_raw, dtype=np.float64)
    pcc_log = np.asarray(pcc_log, dtype=np.float64)
    mae_raw = np.asarray(mae_raw, dtype=np.float64)
    mae_log = np.asarray(mae_log, dtype=np.float64)
    log(f"\n  RAW  PCC: mean={np.nanmean(pcc_raw):.6f} std={np.nanstd(pcc_raw):.6f}  "
        f"MAE: mean={np.nanmean(mae_raw):.6f} std={np.nanstd(mae_raw):.6f}")
    log(f"  LOG1P PCC: mean={np.nanmean(pcc_log):.6f} std={np.nanstd(pcc_log):.6f}  "
        f"MAE: mean={np.nanmean(mae_log):.6f} std={np.nanstd(mae_log):.6f}")
    log(f"  PCC raw nan count: {int(np.isnan(pcc_raw).sum())}/{n_cells}")
    log(f"  elapsed: {(time.time()-t0)/60:.1f} min")

    return {
        "n_cells": n_cells,
        "n_features": n_features,
        "pcc_raw_mean": float(np.nanmean(pcc_raw)),
        "pcc_raw_std": float(np.nanstd(pcc_raw)),
        "mae_raw_mean": float(np.nanmean(mae_raw)),
        "mae_raw_std": float(np.nanstd(mae_raw)),
        "pcc_log_mean": float(np.nanmean(pcc_log)),
        "pcc_log_std": float(np.nanstd(pcc_log)),
        "mae_log_mean": float(np.nanmean(mae_log)),
        "mae_log_std": float(np.nanstd(mae_log)),
        "n_const_pred_cells_sample": n_const,
        "pred_std_median": float(np.median(stds)),
    }


DATASETS = [
    "v3_hybrid_W0p5_500cells_level0",
    "v3_hybrid_W0p6_500cells_level0",
    "v3_hybrid_W0p7_500cells_level0",
    "v3_hybrid_W0p7_500cells_level0_r0p01",
    "v3_hybrid_W0p7_500cells_level0_r0p05",
    "v3_hybrid_W0p8_500cells_level0",
    "v3_hybrid_W0p9_500cells_level0",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--metrics-dir", type=Path, required=True)
    args = ap.parse_args()
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.metrics_dir / "diagnostics_report.txt"
    log_handle = open(report_path, "w")

    raw_rows = []
    log_rows = []
    for stem in DATASETS:
        pred_path = args.pred_dir / f"{stem}_hicimpute_Impute_All_lower_tri.npz"
        gt_path = args.gt_dir / f"{stem}_scdiff2.h5ad"
        if not pred_path.exists():
            print(f"[skip] missing pred: {pred_path}", flush=True)
            continue
        if not gt_path.exists():
            print(f"[skip] missing gt: {gt_path}", flush=True)
            continue
        info = diagnose_one(pred_path, gt_path, log_handle)
        if not info:
            continue

        raw_rows.append({
            "method": "HiCImpute", "config_tag": "Impute_All", "dataset": stem,
            "h5ad": gt_path.name, "imputed_file": pred_path.name,
            "transform": "raw_gt_and_raw_prediction",
            "n_cells": info["n_cells"], "n_features": info["n_features"],
            "pcc_mean": info["pcc_raw_mean"], "pcc_std": info["pcc_raw_std"],
            "mae_mean": info["mae_raw_mean"], "mae_std": info["mae_raw_std"],
            "impute_variant": "Impute_All",
        })
        log_rows.append({
            "method": "HiCImpute", "config_tag": "Impute_All", "dataset": stem,
            "h5ad": gt_path.name, "imputed_file": pred_path.name,
            "transform": "log1p_gt_and_log1p_prediction",
            "n_cells": info["n_cells"], "n_features": info["n_features"],
            "pcc_mean": info["pcc_log_mean"], "pcc_std": info["pcc_log_std"],
            "mae_mean": info["mae_log_mean"], "mae_std": info["mae_log_std"],
            "impute_variant": "Impute_All",
        })

    log_handle.close()

    raw_csv = args.metrics_dir / "HiCImpute_FLAMINGO_v3_raw.csv"
    log_csv = args.metrics_dir / "HiCImpute_FLAMINGO_v3_log1p.csv"
    if raw_rows:
        pd.DataFrame(raw_rows).to_csv(raw_csv, index=False)
        print(f"\nSaved: {raw_csv}", flush=True)
    if log_rows:
        pd.DataFrame(log_rows).to_csv(log_csv, index=False)
        print(f"Saved: {log_csv}", flush=True)
    print(f"Saved diagnostics: {report_path}", flush=True)


if __name__ == "__main__":
    main()
