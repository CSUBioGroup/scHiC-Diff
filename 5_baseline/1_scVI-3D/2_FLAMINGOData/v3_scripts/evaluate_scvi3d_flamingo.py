#!/usr/bin/env python3
"""Evaluate scVI-3D FLAMINGO imputation: cell-wise PCC and MAE.

Reads the imputed npz and the h5ad ground truth, computes cell-wise
Pearson correlation coefficient and mean absolute error on log1p scale.

Uses multiprocessing to parallelize across cells.

Usage::

    python evaluate_scvi3d_flamingo.py \\
        --pred-dir  <dir with *_scVI3D_imputed.npz> \\
        --gt-dir    <dir with *_scdiff2.h5ad> \\
        --output-csv <path> \\
        --workers 20 --log1p
"""

from __future__ import annotations

import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import load_npz


# ---------------------------------------------------------------------------
# h5ad row reader (shared via initializer for Pool)
# ---------------------------------------------------------------------------
_GT_H5PATH: str = ""
_GT_H5: h5py.File | None = None
_OBS_H5: h5py.File | None = None
_N_FEATURES: int = 0
_LOG1P: bool = True


def _init_worker(gt_h5path: str, n_features: int, log1p: bool):
    global _GT_H5PATH, _GT_H5, _OBS_H5, _N_FEATURES, _LOG1P
    _GT_H5PATH = gt_h5path
    _N_FEATURES = n_features
    _LOG1P = log1p
    _GT_H5 = h5py.File(gt_h5path, "r")
    _OBS_H5 = h5py.File(gt_h5path, "r")


def _csr_row(group: h5py.Group, row_idx: int) -> np.ndarray:
    """Extract one row from a CSR group in h5ad."""
    indptr = group["indptr"]
    start = int(indptr[row_idx])
    end = int(indptr[row_idx + 1])
    row = np.zeros(_N_FEATURES, dtype=np.float64)
    row[group["indices"][start:end]] = group["data"][start:end]
    return row


def _compute_cell(args: tuple[int, np.ndarray]) -> tuple[int, float, float, float, float, float]:
    """Compute PCC and MAE for one cell (all / observed / heldout)."""
    cell_idx, pred_row = args
    gt_row = _csr_row(_GT_H5["layers/gt"], cell_idx)
    obs_row = _csr_row(_OBS_H5["layers/counts"], cell_idx)

    if _LOG1P:
        pred_use = np.log1p(np.maximum(pred_row, 0.0))
        gt_use = np.log1p(np.maximum(gt_row, 0.0))
    else:
        pred_use = pred_row.copy()
        gt_use = gt_row.copy()

    gt_mask = gt_use > 0
    obs_mask = obs_row > 0
    held_mask = gt_mask & (~obs_mask)

    def safe_pcc_mae(p, g):
        if p.size < 2 or np.std(p) == 0 or np.std(g) == 0:
            return float("nan"), float("nan")
        corr = np.corrcoef(p, g)
        pcc = float(corr[0, 1])
        mae = float(np.mean(np.abs(p - g)))
        return pcc, mae

    pcc_all, mae_all = safe_pcc_mae(pred_use[gt_mask], gt_use[gt_mask])
    pcc_obs, mae_obs = safe_pcc_mae(pred_use[obs_mask], gt_use[obs_mask])
    pcc_held, mae_held = safe_pcc_mae(pred_use[held_mask], gt_use[held_mask])

    n_all = int(gt_mask.sum())
    n_obs = int(obs_mask.sum())
    n_held = int(held_mask.sum())

    return cell_idx, pcc_all, mae_all, pcc_obs, mae_obs, pcc_held, mae_held, n_all, n_obs, n_held


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True,
                        help="Directory with *_scVI3D_imputed.npz files")
    parser.add_argument("--gt-dir", type=Path, required=True,
                        help="Directory with *_scdiff2.h5ad files")
    parser.add_argument("--output-csv", type=Path, required=True,
                        help="Output CSV path for metrics")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = SLURM_CPUS_PER_TASK or 1)")
    parser.add_argument("--log1p", action="store_true", default=True,
                        help="Apply log1p transform before PCC/MAE (default: True)")
    parser.add_argument("--no-log1p", dest="log1p", action="store_false",
                        help="Use raw values (no log1p)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Dataset stems to evaluate (default: all npz in pred-dir)")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing CSV")
    return parser.parse_args()


def weight_tag(dataset: str) -> str:
    m = re.search(r"_W([0-9]p[0-9]+)_", dataset)
    return f"W{m.group(1)}" if m else ""


def retention_tag(dataset: str) -> str:
    m = re.search(r"_(r0p[0-9]+)(?:_|$)", dataset)
    return m.group(1) if m else ""


def evaluate_one(pred_path: Path, gt_dir: Path, workers: int, log1p: bool) -> dict:
    """Evaluate one prediction npz against its h5ad ground truth."""
    # Parse dataset stem from filename
    m = re.match(r"(.+)_scVI3D_imputed\.npz$", pred_path.name)
    if not m:
        raise ValueError(f"Unexpected npz name: {pred_path.name}")
    stem = m.group(1)
    gt_h5path = gt_dir / f"{stem}_scdiff2.h5ad"
    if not gt_h5path.exists():
        raise FileNotFoundError(f"Missing GT h5ad: {gt_h5path}")

    print(f"[eval] {stem}: loading prediction", flush=True)
    pred = load_npz(pred_path).toarray().astype(np.float64, copy=False)
    n_cells, n_features = pred.shape
    print(f"[eval] {stem}: pred shape={pred.shape}", flush=True)

    # Build args for each cell
    items = [(c, pred[c]) for c in range(n_cells)]

    if workers <= 1:
        # Single process: open h5ad directly
        global _GT_H5, _OBS_H5, _N_FEATURES, _LOG1P
        _GT_H5 = h5py.File(gt_h5path, "r")
        _OBS_H5 = h5py.File(gt_h5path, "r")
        _N_FEATURES = n_features
        _LOG1P = log1p
        results = [_compute_cell(it) for it in items]
        _GT_H5.close()
        _OBS_H5.close()
    else:
        with Pool(processes=workers, initializer=_init_worker,
                  initargs=(str(gt_h5path), n_features, log1p)) as p:
            results = p.map(_compute_cell, items,
                            chunksize=max(1, n_cells // (workers * 4)))

    results.sort(key=lambda r: r[0])

    pcc_all = [r[1] for r in results]
    mae_all = [r[2] for r in results]
    pcc_obs = [r[3] for r in results]
    mae_obs = [r[4] for r in results]
    pcc_held = [r[5] for r in results]
    mae_held = [r[6] for r in results]

    row = {
        "method": "scVI-3D",
        "config_tag": "scVI3D_whole_ceil",
        "dataset": stem,
        "h5ad": gt_h5path.name,
        "imputed_file": pred_path.name,
        "transform": "log1p_gt_and_log1p_prediction" if log1p else "raw_gt_and_raw_prediction",
        "n_cells": n_cells,
        "n_features": n_features,
        "pcc_all_mean": float(np.nanmean(pcc_all)),
        "pcc_all_std": float(np.nanstd(pcc_all)),
        "pcc_obs_mean": float(np.nanmean(pcc_obs)),
        "pcc_obs_std": float(np.nanstd(pcc_obs)),
        "pcc_held_mean": float(np.nanmean(pcc_held)),
        "pcc_held_std": float(np.nanstd(pcc_held)),
        "mae_all_mean": float(np.nanmean(mae_all)),
        "mae_all_std": float(np.nanstd(mae_all)),
        "mae_obs_mean": float(np.nanmean(mae_obs)),
        "mae_obs_std": float(np.nanstd(mae_obs)),
        "mae_held_mean": float(np.nanmean(mae_held)),
        "mae_held_std": float(np.nanstd(mae_held)),
        "weight": weight_tag(stem),
        "retention": retention_tag(stem),
        "band_max": "whole",
        "count_transform": "ceil_to_int",
    }
    print(
        f"[eval] {stem}: PCC all={row['pcc_all_mean']:.4f} "
        f"obs={row['pcc_obs_mean']:.4f} held={row['pcc_held_mean']:.4f}",
        flush=True,
    )
    return row


def main() -> int:
    args = parse_args()
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    # Find prediction npz files
    if args.datasets:
        npz_files = [args.pred_dir / f"{s}_scVI3D_imputed.npz" for s in args.datasets]
        npz_files = [f for f in npz_files if f.exists()]
    else:
        npz_files = sorted(args.pred_dir.glob("*_scVI3D_imputed.npz"))

    if not npz_files:
        print(f"[eval] No npz files found in {args.pred_dir}", flush=True)
        return 1

    print(f"[eval] {len(npz_files)} datasets, {workers} workers, log1p={args.log1p}", flush=True)

    rows = []
    for pred_path in npz_files:
        row = evaluate_one(pred_path, args.gt_dir, workers, args.log1p)
        rows.append(row)

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.append and args.output_csv.exists():
        old = pd.read_csv(args.output_csv)
        cols = list(old.columns) + [c for c in df.columns if c not in old.columns]
        comb = pd.concat([old, df], ignore_index=True).reindex(columns=cols)
        comb.to_csv(args.output_csv, index=False)
        print(f"[eval] appended {len(df)} -> {args.output_csv} ({len(comb)} rows)", flush=True)
    else:
        df.to_csv(args.output_csv, index=False)
        print(f"[eval] saved {args.output_csv}", flush=True)

    print(f"\n{df.to_string(index=False)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
