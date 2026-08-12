#!/usr/bin/env python3
"""Convert HiCImpute FLAMINGOData Impute_SZ bin files to tril-ordered npz."""
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, save_npz
from scipy.stats import pearsonr

PROJECT_ROOT = Path("/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline")
sys.path.insert(0, str(PROJECT_ROOT))

from paperplots.recalc_eval_common import (
    load_h5ad_gt_and_observed,
    FLAMINGO_500CELLS_ROOT,
    FLAMINGO_PARAMSWEEP_ROOT,
)

N_BEADS = 500
N_FEATURES = N_BEADS * (N_BEADS - 1) // 2  # 124750
N_CELLS = 1500

DATASETS = [
    ("v3_hybrid_W0p5_500cells_level0", FLAMINGO_PARAMSWEEP_ROOT),
    ("v3_hybrid_W0p6_500cells_level0", FLAMINGO_PARAMSWEEP_ROOT),
    ("v3_hybrid_W0p7_500cells_level0", FLAMINGO_500CELLS_ROOT),
    ("v3_hybrid_W0p7_500cells_level0_r0p01", FLAMINGO_PARAMSWEEP_ROOT),
    ("v3_hybrid_W0p7_500cells_level0_r0p05", FLAMINGO_PARAMSWEEP_ROOT),
    ("v3_hybrid_W0p8_500cells_level0", FLAMINGO_PARAMSWEEP_ROOT),
    ("v3_hybrid_W0p9_500cells_level0", FLAMINGO_PARAMSWEEP_ROOT),
]

BIN_DIR = PROJECT_ROOT / "3_HiCImpute/2_FLAMINGOData/v3_outputData/bin"
INPUT_DIR = PROJECT_ROOT / "3_HiCImpute/2_FLAMINGOData/v3_inputData"
OUT_DIR = PROJECT_ROOT / "3_HiCImpute/2_FLAMINGOData/v3_outputData/npz_dxy"

iu, ju = np.triu_indices(N_BEADS, k=1)
il, jl = np.tril_indices(N_BEADS, k=-1)


def process_one(dataset, gt_root):
    print(f"\n=== {dataset} ===", flush=True)

    # Load Impute_SZ bin (features x cells, R col-major)
    sz_bin = BIN_DIR / f"{dataset}_Impute_SZ.bin"
    all_bin = BIN_DIR / f"{dataset}_Impute_All.bin"
    if not sz_bin.exists():
        print(f"  SKIP: {sz_bin} not found")
        return

    impute_sz = np.fromfile(sz_bin, dtype=np.float64).reshape(N_FEATURES, N_CELLS, order='F')
    print(f"  Impute_SZ: min={impute_sz.min():.4f}, max={impute_sz.max():.4f}, mean={impute_sz.mean():.4f}, std={impute_sz.std():.4f}", flush=True)

    # Load feature_order for inv_order transform
    order_file = INPUT_DIR / dataset / "feature_order.npy"
    order = np.load(order_file)

    # Transform to numpy triu row-major: impute_numpy = impute_r[order, :]
    sz_numpy = impute_sz[order, :]  # features x cells, numpy triu
    del impute_sz

    # Transpose to cells x features
    pred_triu = sz_numpy.T  # cells x features, numpy triu
    del sz_numpy

    # triu -> tril reorder (cell by cell to save memory)
    pred_tril = np.empty_like(pred_triu)
    for c in range(N_CELLS):
        full = np.zeros((N_BEADS, N_BEADS), dtype=np.float64)
        full[iu, ju] = pred_triu[c]
        full = full + full.T
        pred_tril[c] = full[il, jl]
    del pred_triu

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{dataset}_niter50000_burnin5000_Impute_SZ.npz"
    save_npz(out_path, coo_matrix(pred_tril).tocsr())
    print(f"  Saved -> {out_path}", flush=True)

    # Quick PCC check on cell0
    gt, _ = load_h5ad_gt_and_observed(gt_root / f"{dataset}_scdiff2.h5ad")
    r, _ = pearsonr(pred_tril[0], gt[0])
    print(f"  Cell0 PCC (Impute_SZ tril vs GT): {r:.4f}", flush=True)

    del pred_tril, gt


def main():
    dataset_idx = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    if dataset_idx >= 0:
        ds, root = DATASETS[dataset_idx]
        process_one(ds, root)
    else:
        for ds, root in DATASETS:
            process_one(ds, root)


if __name__ == "__main__":
    main()