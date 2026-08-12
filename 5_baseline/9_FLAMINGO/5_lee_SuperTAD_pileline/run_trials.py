"""
Stage 2: 100-trial sampling + PCC computation
- Reads imputed per-cell NPZ from imputed/{method}/
- 100 trials × 4 cell types, each sampling 30 cells
- PCC between trial pseudo-bulk and cool-based target
Usage: python run_trials.py --method scHiC-Diff
"""
import os
import csv
import glob
import argparse
import numpy as np
from scipy.sparse import load_npz, save_npz, csr_matrix
from scipy.stats import pearsonr

import config as cfg


def compute_pcc_full(a, b):
    a = a.copy(); b = b.copy()
    np.fill_diagonal(a, 0); np.fill_diagonal(b, 0)
    if np.std(a) == 0 or np.std(b) == 0: return 0.0
    return pearsonr(a.flatten(), b.flatten())[0]


def compute_pcc_upper(a, b):
    n = a.shape[0]
    triu = np.triu_indices(n, k=1)
    av, bv = a[triu], b[triu]
    if np.std(av) == 0 or np.std(bv) == 0: return 0.0
    return pearsonr(av, bv)[0]


def sample_cell_indices(n_cells, n_sample, seed):
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    if n_sample < 0:
        raise ValueError("n_sample must be nonnegative")
    rng = np.random.RandomState(seed)
    return rng.choice(n_cells, size=min(n_sample, n_cells), replace=False)


def load_cell_stack(files):
    if not files:
        raise ValueError("files must contain at least one cell matrix")

    matrices = []
    expected_shape = None
    for path in files:
        matrix = load_npz(path).toarray()
        if expected_shape is None:
            expected_shape = matrix.shape
        elif matrix.shape != expected_shape:
            raise ValueError(
                f"shape mismatch for {path}: expected {expected_shape}, got {matrix.shape}"
            )
        matrices.append(matrix)
    return np.stack(matrices, axis=0)


def sum_sampled_stack(stack, sampled):
    return stack[np.asarray(sampled, dtype=np.intp)].sum(axis=0, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="100-trial PCC computation")
    parser.add_argument("--method", required=True, help="Imputation method name (e.g. scHiC-Diff)")
    args = parser.parse_args()
    method = args.method

    imputed_dir = os.path.join(cfg.IMPUTED_DIR, method)
    trials_dir = os.path.join(cfg.TRIALS_DIR, method)
    matrices_dir = os.path.join(trials_dir, "matrices")
    os.makedirs(matrices_dir, exist_ok=True)

    print("=" * 60)
    print(f"Stage 2: 100-trial Sampling + PCC (method={method})")
    print("=" * 60)

    csv_path = os.path.join(trials_dir, "pcc_results.csv")
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["trial_id", "cell_type", "seed",
                     "pcc_8x8_full", "pcc_8x8_upper", "pcc_49x49_full",
                     "n_sampled", "n_total"])

    s, e = cfg.PDGFRA_SUB_BINS

    for cell_type in cfg.CELL_TYPES:
        print(f"\n--- {cell_type} ---")
        pattern = os.path.join(imputed_dir, f"{cell_type}_cell_*.npz")
        files = sorted(glob.glob(pattern))
        n_cells = len(files)
        if n_cells == 0:
            print(f"  WARNING: No files at {pattern}")
            continue
        print(f"  {n_cells} cells, sampling {cfg.N_SAMPLE}, {cfg.N_TRIALS} trials")
        cell_stack = load_cell_stack(files)

        target = load_npz(os.path.join(cfg.TARGET_DIR, f"{cell_type}_target.npz")).toarray()
        target_8x8 = target[s:e, s:e]

        for trial_id in range(cfg.N_TRIALS):
            seed = cfg.BASE_SEED + trial_id
            sampled = sample_cell_indices(n_cells, cfg.N_SAMPLE, seed)
            pseudo_bulk = sum_sampled_stack(cell_stack, sampled)

            trial_8x8 = pseudo_bulk[s:e, s:e]
            pcc_8f = compute_pcc_full(trial_8x8, target_8x8)
            pcc_8u = compute_pcc_upper(trial_8x8, target_8x8)
            pcc_49f = compute_pcc_full(pseudo_bulk, target)

            writer.writerow([trial_id, cell_type, seed,
                             f"{pcc_8f:.6f}", f"{pcc_8u:.6f}", f"{pcc_49f:.6f}",
                             len(sampled), n_cells])

            save_npz(os.path.join(matrices_dir, f"{cell_type}_trial{trial_id:03d}.npz"),
                     csr_matrix(pseudo_bulk))

            if trial_id % 20 == 0:
                print(f"  Trial {trial_id:3d}: PCC_8x8={pcc_8f:.3f}")

    csv_file.close()
    print(f"\nResults saved: {csv_path}")

    import pandas as pd
    df = pd.read_csv(csv_path)
    print("\n=== PCC Summary (mean ± std) ===")
    for ct in cfg.CELL_TYPES:
        sub = df[df.cell_type == ct]
        if len(sub) == 0: continue
        print(f"  {ct:6s}: 8x8_full={sub.pcc_8x8_full.mean():.3f}±{sub.pcc_8x8_full.std():.3f}, "
              f"49x49={sub.pcc_49x49_full.mean():.3f}±{sub.pcc_49x49_full.std():.3f}")
    print("\nStage 2 complete.")


if __name__ == "__main__":
    main()
