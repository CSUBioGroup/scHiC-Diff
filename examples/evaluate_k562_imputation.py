#!/usr/bin/env python3
"""
Evaluation script for K562 imputation results across all 12 conditions.
Computes cell-wise PCC, MAE, and SCC on:
  - all: all 1830 triangle features
  - obs: observed > 0
  - held: GT > 0 and observed <= 0 (held-out / imputed positions)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import spearmanr


DATASETS = [
    "K562_T1_1k", "K562_T1_2k", "K562_T1_4k", "K562_T1_7k",
    "K562_T2_1k", "K562_T2_2k", "K562_T2_4k", "K562_T2_7k",
    "K562_T3_1k", "K562_T3_2k", "K562_T3_4k", "K562_T3_7k",
]
N_CELLS, N_FEATURES = 100, 1830


def resolve_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for p in [current] + list(current.parents):
        if (p / "main.py").exists() and (p / "schicdiff").is_dir():
            return p
    raise RuntimeError("Could not find repository root containing main.py and schicdiff/")


def load_matrix(path: Path) -> np.ndarray:
    path = Path(path)
    if path.is_file():
        m = load_npz(path)
        if hasattr(m, "toarray"):
            m = m.toarray()
        return np.asarray(m, dtype=np.float64)
    if path.is_dir():
        tril_r, tril_c = np.tril_indices(61, k=-1)
        cells = []
        for i in range(1, 101):
            f = path / f"cell_{i}_chr19.txt"
            mat = np.zeros((61, 61), dtype=np.float64)
            if f.exists() and f.stat().st_size > 0:
                data = np.loadtxt(f, dtype=int)
                if data.ndim == 1 and data.size == 3:
                    mat[data[0], data[1]] = data[2]
                elif data.ndim == 2:
                    mat[data[:, 0], data[:, 1]] = data[:, 2]
            cells.append(mat[tril_r, tril_c])
        return np.asarray(cells, dtype=np.float64)
    stem = path.stem.removesuffix("_sim")
    alt_dir = path.parent / stem
    if alt_dir.is_dir():
        return load_matrix(alt_dir)
    raise FileNotFoundError(f"Data file or directory not found: {path}")


def safe_pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def safe_mae(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return np.nan
    return float(np.mean(np.abs(a - b)))


def safe_spearman(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(spearmanr(a, b)[0])


def evaluate_dataset(ds: str, gt_dir: Path, obs_dir: Path, save_root: Path) -> tuple[dict, dict]:
    gt_file = gt_dir / f"{ds}_true.npz"
    obs_file = obs_dir / f"{ds}_sim.npz"
    pred_file = save_root / f"{ds}_sim" / "denoise_recon_inv.npz"

    if not pred_file.exists():
        raise FileNotFoundError(f"Imputed result missing: {pred_file}")

    gt = load_matrix(gt_file)
    obs = load_matrix(obs_file)
    pred = load_matrix(pred_file)
    assert gt.shape == obs.shape == pred.shape == (N_CELLS, N_FEATURES), \
        f"Shape mismatch in {ds}: gt={gt.shape}, obs={obs.shape}, pred={pred.shape}"

    per_cell = {f"{m}_{s}": [] for m in ("pcc", "mae", "scc") for s in ("all", "obs", "held")}
    n_held = []
    for i in range(gt.shape[0]):
        g, o, p = gt[i], obs[i], pred[i]
        masks = {
            "all": np.ones_like(g, dtype=bool),
            "obs": o > 0,
            "held": (g > 0) & ~(o > 0),
        }
        n_held.append(int(masks["held"].sum()))
        for s, m in masks.items():
            per_cell[f"pcc_{s}"].append(safe_pearson(p[m], g[m]))
            per_cell[f"mae_{s}"].append(safe_mae(p[m], g[m]))
            per_cell[f"scc_{s}"].append(safe_spearman(p[m], g[m]))

    row = {
        "data_name": ds,
        "ctype": ds.split("_")[1],
        "cdepth": ds.split("_")[2],
        "n_held_mean": float(np.mean(n_held)),
    }
    for k, v in per_cell.items():
        row[f"{k}_mean"] = float(np.nanmean(v))
        row[f"{k}_std"] = float(np.nanstd(v))

    return row, per_cell


def main():
    parser = argparse.ArgumentParser(description="Evaluate K562 scHiC-Diff imputation results")
    parser.add_argument("--save-root", type=str, default=None,
                        help="Root directory containing {dataset}_sim/denoise_recon_inv.npz")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for metrics tables")
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    save_root = Path(args.save_root) if args.save_root else repo_root / "results/training_results_v5fast_bs128"
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "results/metrics_v5fast_bs128"
    examples_dir = repo_root / "examples"

    gt_dir = repo_root / "5_baseline/0_gtData/1_Gt_HiCImputeData"
    obs_dir = repo_root / "5_baseline/0_gtData/0_downsampled_HiCImputeData"

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("K562 Imputation Evaluation (All 12 Datasets)")
    print(f"Repository Root: {repo_root}")
    print(f"Results Root:    {save_root}")
    print(f"GT Directory:    {gt_dir}")
    print(f"OBS Directory:   {obs_dir}")
    print(f"Output Directory:{out_dir}")
    print("=" * 70)

    available = []
    missing = []
    for ds in DATASETS:
        target_npz = save_root / f"{ds}_sim" / "denoise_recon_inv.npz"
        if target_npz.exists():
            available.append(ds)
        else:
            missing.append(ds)

    print(f"Found {len(available)} / {len(DATASETS)} imputed results.")
    if missing:
        print(f"Missing datasets ({len(missing)}):", missing)

    if not available:
        print("No completed datasets found to evaluate. Exiting.")
        return

    rows = []
    for ds in available:
        row, _ = evaluate_dataset(ds, gt_dir, obs_dir, save_root)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save outputs
    metrics_csv = out_dir / "HiCImputeData_PCC_MAE_SCC_metrics.csv"
    df.to_csv(metrics_csv, index=False)
    # Also save a copy in examples/
    examples_csv = examples_dir / "HiCImputeData_PCC_MAE_SCC_metrics.csv"
    df.to_csv(examples_csv, index=False)

    cols_to_show = ["data_name", "pcc_all_mean", "pcc_held_mean", "mae_all_mean", "mae_held_mean", "scc_held_mean", "n_held_mean"]
    print("\n" + "=" * 85)
    print("EVALUATION SUMMARY TABLE:")
    print("=" * 85)
    print(df[cols_to_show].round(4).to_string(index=False))
    print("-" * 85)
    print(f"AVERAGE (across {len(df)} datasets):")
    print(f"  PCC (all):      {df['pcc_all_mean'].mean():.4f}")
    print(f"  PCC (held-out): {df['pcc_held_mean'].mean():.4f}")
    print(f"  MAE (all):      {df['mae_all_mean'].mean():.4f}")
    print(f"  MAE (held-out): {df['mae_held_mean'].mean():.4f}")
    print(f"  SCC (held-out): {df['scc_held_mean'].mean():.4f}")
    print("=" * 85)
    print(f"Saved metrics to:\n  - {metrics_csv}\n  - {examples_csv}")


if __name__ == "__main__":
    main()
