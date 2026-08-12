#!/usr/bin/env python3
"""
Calculate PCC and MAE for FLAMINGO Baseline (BS=128) results.
"""
import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import load_npz


def load_sparse_as_dense(path):
    """Load sparse matrix and convert to DataFrame."""
    mat = load_npz(path)
    return pd.DataFrame(mat.todense())


def cal_PCC_cell_wise(true, impute_result):
    """Calculate cell-wise Pearson correlation coefficient."""
    pearson_list = []
    cnum = true.shape[0]

    for cellid in range(cnum):
        result_vector = impute_result.iloc[cellid, :]
        true_vector = true.iloc[cellid, :]

        # Calculate Pearson correlation
        corr = np.corrcoef(true_vector, result_vector)
        pearson_list.append(corr[0][1])

    mean_pcc = np.mean(pearson_list)
    std_pcc = np.std(pearson_list)
    print(f"  Cell-wise PCC: {mean_pcc:.4f} ± {std_pcc:.4f}")
    return mean_pcc, std_pcc


def cal_MAE_cell_wise(true, impute_result):
    """Calculate cell-wise Mean Absolute Error."""
    mae_list = []
    cnum = true.shape[0]

    for cellid in range(cnum):
        result_vector = impute_result.iloc[cellid, :]
        true_vector = true.iloc[cellid, :]
        mae = np.mean(np.abs(true_vector - result_vector))
        mae_list.append(mae)

    mean_mae = np.mean(mae_list)
    std_mae = np.std(mae_list)
    print(f"  Cell-wise MAE: {mean_mae:.4f} ± {std_mae:.4f}")
    return mean_mae, std_mae


def csr_group_row(group: h5py.Group, row_idx: int, n_features: int) -> np.ndarray:
    indptr = group["indptr"]
    start = int(indptr[row_idx])
    end = int(indptr[row_idx + 1])
    row = np.zeros(n_features, dtype=np.float64)
    row[group["indices"][start:end]] = group["data"][start:end]
    return row


def cal_row_pcc_mae(true_vector: np.ndarray, result_vector: np.ndarray) -> Tuple[float, float]:
    corr = np.corrcoef(true_vector, result_vector)
    pcc = float(corr[0][1])
    mae = float(np.mean(np.abs(true_vector - result_vector)))
    return pcc, mae


def cal_row_pcc_mae_log1p(true_vector: np.ndarray, result_vector: np.ndarray) -> Tuple[float, float]:
    if np.nanmin(true_vector) < -1 or np.nanmin(result_vector) < -1:
        raise ValueError("values < -1 encountered; log1p invalid")
    return cal_row_pcc_mae(np.log1p(true_vector), np.log1p(result_vector))


def parse_higashi_filename(path: Path) -> Tuple[str, int]:
    match = re.match(r"(.+)_higashi_nbr_([0-9]+)_lower_tri\.npz$", path.name)
    if not match:
        raise ValueError(f"Unexpected Higashi npz name: {path.name}")
    return match.group(1), int(match.group(2))


def parse_v3_lower_tri_filename(path: Path) -> Dict[str, object]:
    higashi = re.match(r"(.+)_higashi_nbr_([0-9]+)_lower_tri\.npz$", path.name)
    if higashi:
        neighbor = int(higashi.group(2))
        return {
            "dataset": higashi.group(1),
            "method": "Higashi",
            "config_tag": f"nbr_{neighbor}",
            "neighbor": neighbor,
            "impute_variant": "",
        }

    hicimpute = re.match(r"(.+)_hicimpute_(.+)_lower_tri\.npz$", path.name)
    if hicimpute:
        return {
            "dataset": hicimpute.group(1),
            "method": "HiCImpute",
            "config_tag": hicimpute.group(2),
            "neighbor": "",
            "impute_variant": hicimpute.group(2),
        }

    scvi3d = re.match(r"(.+)_scVI3D_imputed\.npz$", path.name)
    if scvi3d:
        return {
            "dataset": scvi3d.group(1),
            "method": "scVI-3D",
            "config_tag": "scVI3D_imputed",
            "neighbor": "",
            "impute_variant": "scVI3D_imputed",
        }

    raise ValueError(f"Unexpected v3 lower-triangle npz name: {path.name}")


def weight_tag(dataset: str) -> str:
    match = re.search(r"_W([0-9]p[0-9]+)_", dataset)
    return f"W{match.group(1)}" if match else ""


def retention_tag(dataset: str) -> str:
    match = re.search(r"_(r0p[0-9]+)(?:_|$)", dataset)
    return match.group(1) if match else ""


def process_higashi_v3_npz(pred_path: Path, gt_dir: Path, log1p_transform: bool = False) -> Dict[str, object]:
    meta = parse_v3_lower_tri_filename(pred_path)
    dataset = meta["dataset"]
    gt_path = gt_dir / f"{dataset}_scdiff2.h5ad"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT h5ad for {dataset}: {gt_path}")

    pred = load_npz(pred_path).tocsr()
    with h5py.File(gt_path, "r") as handle:
        gt_group = handle["layers"]["gt"]
        n_cells = len(gt_group["indptr"]) - 1
        n_features = pred.shape[1]
        if pred.shape[0] != n_cells:
            raise ValueError(
                f"{pred_path.name}: pred shape {pred.shape}, gt rows {n_cells}"
            )

        pccs = []
        maes = []
        for row_idx in range(n_cells):
            true_vector = csr_group_row(gt_group, row_idx, n_features)
            result_vector = pred.getrow(row_idx).toarray().ravel().astype(np.float64, copy=False)
            if log1p_transform:
                pcc, mae = cal_row_pcc_mae_log1p(true_vector, result_vector)
            else:
                pcc, mae = cal_row_pcc_mae(true_vector, result_vector)
            pccs.append(pcc)
            maes.append(mae)

    pccs_arr = np.asarray(pccs, dtype=np.float64)
    maes_arr = np.asarray(maes, dtype=np.float64)
    return {
        "method": meta["method"],
        "config_tag": meta["config_tag"],
        "dataset": dataset,
        "h5ad": gt_path.name,
        "imputed_file": pred_path.name,
        "transform": "log1p_gt_and_log1p_prediction" if log1p_transform else "raw_gt_and_raw_prediction",
        "n_cells": int(n_cells),
        "n_features": int(n_features),
        "pcc_mean": float(np.nanmean(pccs_arr)),
        "pcc_std": float(np.nanstd(pccs_arr)),
        "mae_mean": float(np.nanmean(maes_arr)),
        "mae_std": float(np.nanstd(maes_arr)),
        "weight": weight_tag(dataset),
        "retention": retention_tag(dataset),
        "neighbor": meta["neighbor"],
        "impute_variant": meta["impute_variant"],
    }


def process_higashi_v3(
    pred_dir: Path,
    gt_dir: Path,
    output_csv: Path = None,
    log1p_transform: bool = False,
) -> None:
    npz_files = sorted({
        *pred_dir.glob("*_lower_tri.npz"),
        *pred_dir.glob("*_scVI3D_imputed.npz"),
    })
    if not npz_files:
        raise RuntimeError(f"No v3 lower-triangle npz files found in {pred_dir}")

    print(f"\n{'='*60}")
    print("Processing FLAMINGO v3 lower-triangle outputs")
    print(f"Pred dir: {pred_dir}")
    print(f"GT dir:   {gt_dir}")
    print(f"Transform: {'log1p_gt_and_log1p_prediction' if log1p_transform else 'raw_gt_and_raw_prediction'}")
    print(f"{'='*60}\n")

    results = []
    for pred_path in npz_files:
        print(f"[{pred_path.name}]")
        row = process_higashi_v3_npz(pred_path, gt_dir, log1p_transform=log1p_transform)
        results.append(row)
        print(
            f"  Shape: ({row['n_cells']}, {row['n_features']}) "
            f"PCC={row['pcc_mean']:.6f} ± {row['pcc_std']:.6f}, "
            f"MAE={row['mae_mean']:.6f} ± {row['mae_std']:.6f}"
        )

    df = pd.DataFrame(results)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved results to: {output_csv}")
    else:
        print("\nNo output CSV requested; printing results only.")
    print(f"\n{df.to_string(index=False)}")


def process_dataset(data_root, output_dir, tag, cell_suffix=""):
    """Process one retention rate dataset."""
    result_dir = data_root / f"beads_500_W0.7_level1_{tag}{cell_suffix}"

    if not result_dir.exists():
        print(f"  [SKIP] Directory not found: {result_dir}")
        return None

    # Load files
    imputed_path = result_dir / "denoise_recon.npz"
    target_path = result_dir / "denoise_target.npz"

    if not imputed_path.exists() or not target_path.exists():
        print(f"  [SKIP] Missing files in: {result_dir}")
        return None

    print(f"  Loading: {result_dir.name}")
    imputed = load_sparse_as_dense(imputed_path)
    target = load_sparse_as_dense(target_path)

    if imputed.shape != target.shape:
        print(f"  [ERROR] Shape mismatch: imputed {imputed.shape}, target {target.shape}")
        return None

    print(f"  Shape: {imputed.shape} (cells × features)")

    # Calculate metrics
    pcc_mean, pcc_std = cal_PCC_cell_wise(target, imputed)
    mae_mean, mae_std = cal_MAE_cell_wise(target, imputed)

    return {
        "tag": tag,
        "n_cells": imputed.shape[0],
        "n_features": imputed.shape[1],
        "pcc_mean": pcc_mean,
        "pcc_std": pcc_std,
        "mae_mean": mae_mean,
        "mae_std": mae_std,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate FLAMINGO Baseline metrics")
    parser.add_argument(
        "--cell-count",
        type=int,
        choices=[30, 100],
        default=None,
        help="Number of cells (30 or 100)"
    )
    parser.add_argument("--higashi-v3-pred-dir", type=Path, default=None)
    parser.add_argument("--higashi-v3-gt-dir", type=Path, default=None)
    parser.add_argument("--higashi-v3-output-csv", type=Path, default=None)
    parser.add_argument("--v3-lower-tri-pred-dir", type=Path, default=None)
    parser.add_argument("--v3-lower-tri-gt-dir", type=Path, default=None)
    parser.add_argument("--v3-lower-tri-output-csv", type=Path, default=None)
    parser.add_argument("--v3-log1p", action="store_true")
    args = parser.parse_args()

    v3_pred_dir = args.v3_lower_tri_pred_dir or args.higashi_v3_pred_dir
    if v3_pred_dir is not None:
        v3_gt_dir = args.v3_lower_tri_gt_dir or args.higashi_v3_gt_dir
        v3_output_csv = args.v3_lower_tri_output_csv or args.higashi_v3_output_csv
        if v3_gt_dir is None:
            raise ValueError(
                "--v3-lower-tri-gt-dir is required with --v3-lower-tri-pred-dir"
            )
        process_higashi_v3(
            v3_pred_dir,
            v3_gt_dir,
            v3_output_csv,
            log1p_transform=args.v3_log1p,
        )
        return

    if args.cell_count is None:
        raise ValueError("--cell-count is required unless --higashi-v3-pred-dir is used")

    # Setup paths
    base_dir = Path("/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData")

    if args.cell_count == 30:
        data_root = base_dir / "retention_sweep_h5ad/training_results"
        cell_suffix = ""
    else:  # 100
        data_root = base_dir / "retention_sweep_100cells_h5ad/training_results"
        cell_suffix = "_100cells"

    output_dir = Path(__file__).resolve().parent / "results_pcc_all"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Retention rates to process
    tags = ["r0p005", "r0p01", "r0p02", "r0p05", "r0p1", "r0p2"]

    print(f"\n{'='*60}")
    print(f"Processing FLAMINGO Baseline (BS=128) - {args.cell_count} cells")
    print(f"{'='*60}\n")

    results = []
    for tag in tags:
        print(f"[{tag}]")
        result = process_dataset(data_root, output_dir, tag, cell_suffix)
        if result:
            results.append(result)
        print()

    # Save results
    if results:
        df = pd.DataFrame(results)
        output_csv = output_dir / f"FLAMINGO_Baseline_BS128_{args.cell_count}cells_PCC_MAE.csv"
        df.to_csv(output_csv, index=False)
        print(f"✓ Saved results to: {output_csv}")
        print(f"\n{df.to_string(index=False)}")
    else:
        print("[WARNING] No results generated")


if __name__ == "__main__":
    main()
