#!/usr/bin/env python3
"""Evaluate imputation quality for the FLAMINGO v3 paramsweep baselines.

Computes cell-wise log1p(max(x,0)) PCC / Spearman / MAE for ``all`` /
``observed`` / ``heldout`` masks, following the convention of
``12_hpc_cal_FLAMINGOData_ALL_Pearson_MAE.py``.  The per-cell GT comes from
``layers['gt']`` of ``*_scdiff2.h5ad`` (1500 rows).  Observed mask = GT counts
layer (what was fed to the imputer).  Held-out indices come from
``fixed_heldout_indices.npz`` (array ``heldout`` shape ``(n_heldout, 2)`` with
columns ``[cell_index, feature_index]``).

Usage:
  python v3_evaluate.py --method scHiCluster \\
      --pred-dir <dir with *_scHiCluster_imputed.npz> \\
      --pred-pattern "{stem}_scHiCluster_imputed.npz" \\
      --output-csv <path> [--append]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_common import (DATASETS, DEFAULT_DATA_DIR, N_BINS, N_FEATURES,
                       load_layer_dense, parse_tag)

HELDOUT_NPZ = "fixed_heldout_indices.npz"


def parse_args() -> argparse.Namespace:
    d = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--pred-pattern", default="{stem}_imputed.npz",
                        help="Filename template; {stem} replaced by dataset stem")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--config-tag", default="",
                        help="Extra label appended to method (e.g. 0nbr / 5nbr)")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def load_pred(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return load_npz(path).toarray().astype(np.float64, copy=False)


def safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    v = fn(x, y)
    if hasattr(v, "statistic"):
        return float(v.statistic)
    if isinstance(v, tuple):
        return float(v[0])
    return float(v)


def cell_metrics(pred_row, gt_row, obs_mask, held_mask) -> dict:
    gt_mask = np.isfinite(gt_row) & (gt_row > 0)
    obs_m = gt_mask & obs_mask
    held_m = gt_mask & held_mask
    out = {"n_all": int(gt_mask.sum()), "n_observed": int(obs_m.sum()),
           "n_heldout": int(held_m.sum()),
           "observed_frac": float(obs_m.sum() / gt_mask.sum()) if gt_mask.sum() else float("nan")}
    for tag, mask in (("all", gt_mask), ("observed", obs_m), ("heldout", held_m)):
        p = np.log1p(np.maximum(pred_row[mask], 0.0))
        g = np.log1p(np.maximum(gt_row[mask], 0.0))
        out[f"pcc_{tag}"] = safe_corr(pearsonr, p, g)
        out[f"spearman_{tag}"] = safe_corr(spearmanr, p, g)
        out[f"mae_{tag}"] = float(np.mean(np.abs(p - g))) if p.size else float("nan")
    return out


def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    s = {"n_cells": int(len(df))}
    for c in df.columns:
        if c.startswith("n_"):
            s[f"{c}_total"] = int(df[c].sum(skipna=True))
        elif c == "observed_frac":
            s[f"{c}_mean"] = float(df[c].mean(skipna=True))
            s[f"{c}_std"] = float(df[c].std(skipna=True))
        else:
            s[f"{c}_mean"] = float(df[c].mean(skipna=True))
            s[f"{c}_std"] = float(df[c].std(skipna=True))
    return s


def evaluate(ds, args, heldout_cells):
    gt = load_layer_dense(args.data_dir / f"{ds.stem}_scdiff2.h5ad", "gt")
    pred_path = args.pred_dir / args.pred_pattern.format(stem=ds.stem)
    print(f"[eval] {ds.stem}: loading pred {pred_path.name}", flush=True)
    pred = load_pred(pred_path)
    if pred.shape != gt.shape:
        raise ValueError(f"{ds.stem} shape mismatch pred {pred.shape} vs gt {gt.shape}")
    observed = load_layer_dense(args.data_dir / f"{ds.stem}_scdiff2.h5ad", "counts")
    obs_mask = np.isfinite(observed) & (observed > 0)
    held_mask = np.zeros_like(gt, dtype=bool)
    for ci, fi in heldout_cells:
        if 0 <= ci < held_mask.shape[0] and 0 <= fi < held_mask.shape[1]:
            held_mask[ci, fi] = True
    rows = [cell_metrics(pred[c], gt[c], obs_mask[c], held_mask[c])
            for c in range(pred.shape[0])]
    return summarize(rows)


def main() -> int:
    args = parse_args()
    datasets = args.datasets or [d.stem for d in DATASETS]
    if not datasets:
        raise ValueError("No datasets")
    held_npz = args.data_dir / HELDOUT_NPZ
    heldout = np.empty((0, 2), dtype=np.int64)
    if held_npz.exists():
        with np.load(held_npz, allow_pickle=False) as d:
            heldout = np.asarray(d["heldout"], dtype=np.int64)
        print(f"[eval] heldout entries: {heldout.shape}", flush=True)
    ds_map = {d.stem: d for d in DATASETS}
    rows = []
    for stem in datasets:
        ds = ds_map.get(stem)
        if ds is None:
            raise ValueError(f"Unknown dataset {stem}")
        summ = evaluate(ds, args, heldout)
        method = args.method + (f" {args.config_tag}" if args.config_tag else "")
        rows.append({"method": args.method, "config_tag": args.config_tag,
                     "data_name": ds.stem, "tag": parse_tag(ds.stem),
                     "n_cells": ds.n_cells, "n_features": ds.n_features,
                     "pred_file": str(args.pred_dir / args.pred_pattern.format(stem=ds.stem)),
                     **summ})
        print(f"[eval] {stem}: PCC all={rows[-1]['pcc_all_mean']:.4f} "
              f"obs={rows[-1]['pcc_observed_mean']:.4f} held={rows[-1]['pcc_heldout_mean']:.4f}",
              flush=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if args.append and args.output_csv.exists():
        old = pd.read_csv(args.output_csv)
        cols = list(old.columns) + [c for c in new.columns if c not in old.columns]
        comb = pd.concat([old, new], ignore_index=True, sort=False).reindex(columns=cols)
        comb.to_csv(args.output_csv, index=False)
        print(f"[eval] appended {len(new)} -> {args.output_csv} ({len(comb)} rows)", flush=True)
    else:
        new.to_csv(args.output_csv, index=False)
        print(f"[eval] saved {args.output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())