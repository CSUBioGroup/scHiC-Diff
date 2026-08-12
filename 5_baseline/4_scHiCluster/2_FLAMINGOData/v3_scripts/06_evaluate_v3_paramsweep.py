#!/usr/bin/env python3
"""Evaluate scHiCluster imputation quality for FLAMINGO v3 paramsweep data.

For each of the 6 paramsweep datasets we compare the scHiCluster imputed
contacts against the ground truth stored in ``layers['gt']`` of the
corresponding ``*_scdiff2.h5ad`` file.  Metrics are computed cell-wise on
``log1p(max(x, 0))`` transformed contacts, following the convention of
``12_hpc_cal_FLAMINGOData_ALL_Pearson_MAE.py``:

  * log1p Pearson (PCC) mean / std
  * log1p Spearman (SCC) mean / std
  * log1p MAE mean / std

Three evaluation masks are reported:
  * all      - every feature
  * observed - features present in the input (counts > 0)
  * heldout  - features absent from the input but present in GT (held-out)

The held-out indices come from ``fixed_heldout_indices.npz`` (array
``heldout``, shape ``(n_heldout, 2)`` with columns ``[cell_index,
feature_index]``).  The observed mask is derived per cell from
``layers['counts']`` of the heldout-masked h5ad so that the input fed to
scHiCluster (which was prepared from the non-masked counts) matches what the
imputer actually saw.

Predictions are loaded from the per-dataset feature NPZ written by
``05_collect_v3_paramsweep.py`` (shape ``(n_cells, n_features)``).
"""

from __future__ import annotations

import argparse
import re
from multiprocessing import Pool
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import pearsonr, spearmanr


DEFAULT_DATA_DIR = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/"
    "5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/"
    "3_fixed_flamingoGen_datasets/5_paramsweep_datasets"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "4_scHiCluster/2_FLAMINGOData/v3_outputData"
)
DEFAULT_OUTPUT_CSV = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "4_scHiCluster/2_FLAMINGOData/v3_outputData/metrics/"
    "scHiCluster_FLAMINGO_v3_paramsweep_quality_metrics.csv"
)
HELDOUT_NPZ_NAME = "fixed_heldout_indices.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate cell-wise PCC/Spearman/MAE for v3 paramsweep scHiCluster imputation."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--method", default="scHiCluster")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--append", action="store_true",
                        help="Append results to the output CSV instead of overwriting it")
    return parser.parse_args()


def discover_datasets(data_dir: Path) -> list[str]:
    stems = []
    for path in sorted(data_dir.glob("v3_hybrid_*_scdiff2.h5ad")):
        m = re.match(r"^(v3_hybrid_.+)_scdiff2\.h5ad$", path.name)
        if m:
            stems.append(m.group(1))
    return stems


def load_pred_npz(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return load_npz(path).toarray().astype(np.float64, copy=False)


def load_layer(h5ad: Path, layer: str) -> np.ndarray:
    if not h5ad.exists():
        raise FileNotFoundError(h5ad)
    adata = ad.read_h5ad(h5ad)
    mat = adata.layers[layer]
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    return np.asarray(mat, dtype=np.float64)


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
                 observed_mask: np.ndarray, heldout_mask: np.ndarray) -> dict:
    gt_mask = np.isfinite(gt_row) & (gt_row > 0)
    obs_mask = gt_mask & observed_mask
    held_mask = gt_mask & heldout_mask
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


def evaluate_dataset(dataset: str, args: argparse.Namespace,
                     heldout_cells: np.ndarray) -> dict:
    gt_h5ad = args.data_dir / f"{dataset}_scdiff2.h5ad"
    masked_h5ad = None
    masked_candidates = sorted(args.data_dir.glob(f"{dataset}_heldout_masked_*.h5ad"))
    if masked_candidates:
        masked_h5ad = masked_candidates[0]
    pred_path = args.output_root / "2_lower_tri_npz" / f"{dataset}_scHiCluster_imputed.npz"

    print(f"[eval] {dataset}: loading GT {gt_h5ad.name}", flush=True)
    gt = load_layer(gt_h5ad, "gt")
    print(f"[eval] {dataset}: loading pred {pred_path}", flush=True)
    pred = load_pred_npz(pred_path)
    if pred.shape != gt.shape:
        raise ValueError(f"{dataset} shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    # Observed mask: scHiCluster input was prepared from the GT h5ad counts
    # layer, so that counts layer IS the observed mask.  We read it directly
    # rather than picking an arbitrary heldout-masked variant.
    print(f"[eval] {dataset}: using GT counts layer as observed mask", flush=True)
    observed = load_layer(gt_h5ad, "counts")
    if observed.shape != gt.shape:
        raise ValueError(f"{dataset} observed shape {observed.shape} != gt {gt.shape}")
    observed_mask = np.isfinite(observed) & (observed > 0)

    # Held-out mask from the global heldout index table.
    heldout_mask = np.zeros_like(gt, dtype=bool)
    # heldout_cells is the set of cell indices that have held-out entries.
    for cell_idx, feat_idx in heldout_cells:
        if 0 <= cell_idx < heldout_mask.shape[0] and 0 <= feat_idx < heldout_mask.shape[1]:
            heldout_mask[cell_idx, feat_idx] = True

    rows = []
    for c in range(pred.shape[0]):
        rows.append(cell_metrics(pred[c], gt[c], observed_mask[c], heldout_mask[c]))
    summ = summarize(rows)
    return {
        "method": args.method,
        "data_name": dataset,
        "n_cells": pred.shape[0],
        "n_features": int(pred.shape[1]),
        "gt_h5ad": str(gt_h5ad),
        "pred_file": str(pred_path),
        **summ,
    }


def main() -> int:
    args = parse_args()
    datasets = args.datasets or discover_datasets(args.data_dir)
    if not datasets:
        raise ValueError(f"No *_scdiff2.h5ad found under {args.data_dir}")
    heldout_npz = args.data_dir / HELDOUT_NPZ_NAME
    heldout_cells = np.empty((0, 2), dtype=np.int64)
    if heldout_npz.exists():
        with np.load(heldout_npz, allow_pickle=False) as d:
            heldout_cells = np.asarray(d["heldout"], dtype=np.int64)
        print(f"[eval] heldout entries: {heldout_cells.shape}", flush=True)
    else:
        print(f"[eval] WARNING: {heldout_npz} not found; heldout metrics will be empty", flush=True)

    rows = []
    for ds in datasets:
        row = evaluate_dataset(ds, args, heldout_cells)
        rows.append(row)
        print(f"[eval] {ds}: PCC all={row['pcc_all_mean']:.4f} "
              f"obs={row['pcc_observed_mean']:.4f} held={row['pcc_heldout_mean']:.4f}",
              flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if args.append and args.output_csv.exists():
        old_df = pd.read_csv(args.output_csv)
        combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)
        # Reorder so existing columns come first, new columns append at the end.
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