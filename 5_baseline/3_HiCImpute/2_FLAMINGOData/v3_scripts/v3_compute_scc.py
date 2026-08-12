#!/usr/bin/env python3
"""Compute SCC (Stratified Correlation Coefficient) between HiCImpute v3
predictions and GT for all 7 FLAMINGO datasets.

SCC definition (hicrep, Yang et al. 2017):
  For each genomic distance d = |i - j|:
    r_d = Pearson(pred[d], gt[d])     within distance stratum d
    w_d = n_d                          number of bead-pairs at distance d
  SCC = sum(w_d * r_d) / sum(w_d)     over all valid strata

Also computes SCC with optional smoothing (moving-average over distance
window [d-h, d+h]) as in hicrep's h parameter.

Per-cell SCC is computed, then mean/std across 1500 cells.

Outputs:
  <metrics_dir>/HiCImpute_FLAMINGO_v3_SCC.csv
  <metrics_dir>/scc_diagnostics_report.txt
"""
import argparse
import time
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import load_npz


N_BEADS = 500


def precompute_distance_indices(n_beads: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (distances, sorted_index) for upper-triangle features.

    distances[k] = genomic distance (j - i) for feature k where
    feature k = (iu[k], ju[k]) from np.triu_indices(n, k=1).
    sorted_index sorts features by distance for fast stratum slicing.
    """
    iu, ju = np.triu_indices(n_beads, k=1)
    distances = (ju - iu).astype(np.int32)
    order = np.argsort(distances, kind="stable")
    distances_sorted = distances[order]
    return distances_sorted, order


def stratum_boundaries(distances_sorted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (unique_distances, start_indices) so that
    features at distance unique_distances[k] are
    distances_sorted[start[k]:start[k+1]] (and order[...])."""
    uniq, starts = np.unique(distances_sorted, return_index=True)
    starts = np.append(starts, len(distances_sorted))
    return uniq, starts


def smooth_by_distance(values_by_dist: np.ndarray, h: int) -> np.ndarray:
    """Moving-average smoothing over distance axis.
    values_by_dist[d] = mean contact at distance d.
    Smoothed[d] = mean(values_by_dist[d-h : d+h+1])."""
    n = len(values_by_dist)
    if h <= 0:
        return values_by_dist
    out = np.empty(n, dtype=np.float64)
    for d in range(n):
        lo = max(0, d - h)
        hi = min(n, d + h + 1)
        out[d] = values_by_dist[lo:hi].mean()
    return out


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    sa = a.std()
    sb = b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_scc_cell(
    pred_vec: np.ndarray,
    gt_vec: np.ndarray,
    order: np.ndarray,
    uniq_d: np.ndarray,
    starts: np.ndarray,
    h: int = 0,
) -> Tuple[float, int]:
    """Compute SCC for one cell.

    If h > 0, applies hicrep-style smoothing: for each distance d, the
    predicted and GT values are smoothed by averaging over the distance
    window [d-h, d+h] before computing per-stratum correlation.

    Returns (scc, n_valid_strata).
    """
    pred_sorted = pred_vec[order]
    gt_sorted = gt_vec[order]

    if h <= 0:
        # No smoothing: per-distance Pearson, weighted by n_pairs
        weighted_sum = 0.0
        weight_total = 0.0
        n_valid = 0
        for k in range(len(uniq_d)):
            s, e = starts[k], starts[k + 1]
            p = pred_sorted[s:e]
            g = gt_sorted[s:e]
            n = e - s
            r = safe_pearson(p, g)
            if not np.isnan(r):
                weighted_sum += n * r
                weight_total += n
                n_valid += 1
        if weight_total == 0:
            return float("nan"), 0
        return weighted_sum / weight_total, n_valid

    # With smoothing (hicrep h parameter):
    # For each distance d, compute smoothed pred and gt by averaging
    # over the distance window [d-h, d+h], then compute Pearson on
    # the smoothed values within stratum d.
    n_d = len(uniq_d)
    pred_means = np.empty(n_d, dtype=np.float64)
    gt_means = np.empty(n_d, dtype=np.float64)
    for k in range(n_d):
        s, e = starts[k], starts[k + 1]
        pred_means[k] = pred_sorted[s:e].mean()
        gt_means[k] = gt_sorted[s:e].mean()

    pred_smooth = smooth_by_distance(pred_means, h)
    gt_smooth = smooth_by_distance(gt_means, h)

    # Correlation of smoothed distance-profiles
    r = safe_pearson(pred_smooth, gt_smooth)
    return r, n_d


def csr_group_row(group: h5py.Group, row_idx: int, n_features: int) -> np.ndarray:
    indptr = group["indptr"]
    start = int(indptr[row_idx])
    end = int(indptr[row_idx + 1])
    row = np.zeros(n_features, dtype=np.float64)
    row[group["indices"][start:end]] = group["data"][start:end]
    return row


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
    ap.add_argument("--smoothing-h", type=int, nargs="+", default=[0, 1, 2, 5],
                    help="hicrep smoothing window h values (0 = no smoothing)")
    args = ap.parse_args()
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.metrics_dir / "scc_diagnostics_report.txt"
    log_handle = open(report_path, "w")
    log = lambda m: (print(m, flush=True), log_handle.write(m + "\n"))

    distances_sorted, order = precompute_distance_indices(N_BEADS)
    uniq_d, starts = stratum_boundaries(distances_sorted)
    log(f"Beads: {N_BEADS}, features: {len(order)}, "
        f"distances: {uniq_d.min()}..{uniq_d.max()} ({len(uniq_d)} strata)")
    log(f"Smoothing h values: {args.smoothing_h}")

    all_rows = []
    for stem in DATASETS:
        pred_path = args.pred_dir / f"{stem}_hicimpute_Impute_All_lower_tri.npz"
        gt_path = args.gt_dir / f"{stem}_scdiff2.h5ad"
        if not pred_path.exists() or not gt_path.exists():
            log(f"[skip] {stem}: missing files")
            continue

        log(f"\n{'='*70}")
        log(f"SCC: {stem}")
        log(f"{'='*70}")

        pred_csr = load_npz(pred_path).tocsr()
        n_cells, n_features = pred_csr.shape

        t0 = time.time()
        # Collect per-cell SCC for each h value
        scc_by_h = {h: [] for h in args.smoothing_h}
        n_valid_by_h = {h: [] for h in args.smoothing_h}

        with h5py.File(gt_path, "r") as f:
            gt_group = f["layers"]["gt"]
            for ci in range(n_cells):
                prow = pred_csr.getrow(ci).toarray().ravel().astype(np.float64)
                grow = csr_group_row(gt_group, ci, n_features)
                for h in args.smoothing_h:
                    scc, nv = compute_scc_cell(
                        prow, grow, order, uniq_d, starts, h=h
                    )
                    scc_by_h[h].append(scc)
                    n_valid_by_h[h].append(nv)
                if (ci + 1) % 300 == 0:
                    log(f"  ... {ci+1}/{n_cells} cells "
                        f"({(time.time()-t0)/60:.1f} min)")

        for h in args.smoothing_h:
            arr = np.asarray(scc_by_h[h], dtype=np.float64)
            nv = np.asarray(n_valid_by_h[h], dtype=np.float64)
            log(f"  h={h}: SCC mean={np.nanmean(arr):.6f} "
                f"std={np.nanstd(arr):.6f} "
                f"median={np.nanmedian(arr):.6f} "
                f"nan={int(np.isnan(arr).sum())}/{n_cells} "
                f"valid_strata/cell={nv.mean():.1f}")
            all_rows.append({
                "method": "HiCImpute",
                "config_tag": "Impute_All",
                "dataset": stem,
                "h5ad": gt_path.name,
                "imputed_file": pred_path.name,
                "metric": "SCC",
                "smoothing_h": h,
                "n_cells": n_cells,
                "n_features": n_features,
                "n_strata": len(uniq_d),
                "scc_mean": float(np.nanmean(arr)),
                "scc_std": float(np.nanstd(arr)),
                "scc_median": float(np.nanmedian(arr)),
                "scc_min": float(np.nanmin(arr)),
                "scc_max": float(np.nanmax(arr)),
                "n_nan": int(np.isnan(arr).sum()),
                "valid_strata_mean": float(nv.mean()),
            })

        log(f"  elapsed: {(time.time()-t0)/60:.1f} min")

    log_handle.close()

    csv_path = args.metrics_dir / "HiCImpute_FLAMINGO_v3_SCC.csv"
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(csv_path, index=False)
        log(f"\nSaved: {csv_path}")
        log(f"\n{df.to_string(index=False)}")
    print(f"\nSaved: {csv_path}")
    print(f"Saved diagnostics: {report_path}")


if __name__ == "__main__":
    main()
