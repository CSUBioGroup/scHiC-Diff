"""
Stage 1: Data preparation (v2 - cool-based target)
- Load scHiC-Diff raw_x.npz and denoise_recon_inv.npz for trial data
- Aggregate per-cell .cool files to build target pseudo-bulk maps (correct target source)
- Save per-cell matrices to input_lee/ for other imputation methods
"""
import os
import json
import glob
import numpy as np
import cooler
from scipy.sparse import load_npz, save_npz, csr_matrix
from scipy.stats import pearsonr

import config as cfg


def upper_vector2matrix(upper_vector, n=cfg.N_BINS):
    """Reconstruct n x n symmetric matrix from upper-triangle vector (k=1)."""
    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[np.triu_indices(n, k=1)] = np.asarray(upper_vector, dtype=np.float64)
    matrix += matrix.T
    return matrix


def load_cell_data(cell_type):
    """Load raw and imputed data for a cell type from scHiC-Diff outputs."""
    data_dir = os.path.join(cfg.DATA_SOURCE, cell_type)
    raw = load_npz(os.path.join(data_dir, "raw_x.npz")).toarray()
    imputed = load_npz(os.path.join(data_dir, "denoise_recon_inv.npz")).toarray()
    print(f"  raw_x: {raw.shape} (nnz={np.count_nonzero(raw)}), "
          f"imputed: {imputed.shape} (nnz={np.count_nonzero(imputed)})")
    return raw, imputed


def aggregate_cool_target(cell_type):
    """Aggregate all per-cell .cool files to build target pseudo-bulk map.
    Fetches chr4:54890000-55380000 (49 bins at 10kb) from each cell."""
    cool_dir = os.path.join(cfg.COOL_DIR, cell_type)
    cool_files = sorted(glob.glob(os.path.join(cool_dir, "*.cool")))
    n_cells = len(cool_files)
    print(f"  Aggregating {n_cells} cool files...")

    fetch_str = f"{cfg.CHROM}:{cfg.REGION_START}-{cfg.REGION_START + cfg.N_BINS * cfg.RESOLUTION}"
    target_49 = np.zeros((cfg.N_BINS, cfg.N_BINS), dtype=np.float64)
    n_ok = 0
    n_fail = 0

    for i, fpath in enumerate(cool_files):
        try:
            c = cooler.Cooler(fpath)
            mat = c.matrix(balance=False).fetch(fetch_str)
            target_49 += np.array(mat, dtype=np.float64)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                print(f"    WARNING: skipped {os.path.basename(fpath)}: {e}")
        if (i + 1) % 200 == 0 or (i + 1) == n_cells:
            print(f"    {i+1}/{n_cells} done ({n_ok} ok, {n_fail} skipped)")

    return target_49, n_ok


def save_per_cell_npz(cell_type, cell_idx, matrix_49x49, output_dir):
    """Save a single cell's 49x49 matrix as NPZ."""
    fname = f"{cell_type}_cell_{cell_idx:04d}.npz"
    save_npz(os.path.join(output_dir, fname), csr_matrix(matrix_49x49))
    return fname


def save_per_cell_bedpe(cell_type, cell_idx, matrix_49x49, output_dir):
    """Save a single cell's contacts in bedpe-like format (non-zero only)."""
    fname = f"{cell_type}_cell_{cell_idx:04d}.txt"
    out_path = os.path.join(output_dir, fname)
    n = matrix_49x49.shape[0]
    lines = []
    for i in range(n):
        for j in range(i + 1, n):
            val = matrix_49x49[i, j]
            if val > 0:
                lines.append(f"{cfg.CHROM}\t{cfg.get_bin_start(i)}\t{cfg.CHROM}\t{cfg.get_bin_start(j)}\t{val:.6f}")
    with open(out_path, 'w') as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return fname


def main():
    print("=" * 60)
    print("Stage 1: Data Preparation (v2 - cool-based target)")
    print("=" * 60)

    for d in [cfg.PER_CELL_NPZ_DIR, cfg.PER_CELL_BEDPE_DIR, cfg.TARGET_DIR]:
        os.makedirs(d, exist_ok=True)

    metadata = {
        "chromosome": cfg.CHROM,
        "resolution": cfg.RESOLUTION,
        "n_bins": cfg.N_BINS,
        "region_start": cfg.REGION_START,
        "region_end": cfg.REGION_START + cfg.N_BINS * cfg.RESOLUTION,
        "pdgfra_start": cfg.PDGFRA_START,
        "pdgfra_end": cfg.PDGFRA_END,
        "pdgfra_sub_bins": list(cfg.PDGFRA_SUB_BINS),
        "target_source": "per-cell .cool files aggregated",
        "trial_source": "scHiC-Diff raw_x.npz and denoise_recon_inv.npz",
        "cell_types": {}
    }

    for cell_type in cfg.CELL_TYPES:
        print(f"\n--- {cell_type} ---")

        # 1. Load scHiC-Diff data (for trials)
        raw, imputed = load_cell_data(cell_type)
        n_cells_raw = raw.shape[0]

        # 2. Save per-cell matrices to input_lee/ (from raw_x, for other methods)
        print(f"  Saving {n_cells_raw} per-cell matrices to input_lee/...")
        for cell_idx in range(n_cells_raw):
            raw_matrix = upper_vector2matrix(raw[cell_idx])
            save_per_cell_npz(cell_type, cell_idx, raw_matrix, cfg.PER_CELL_NPZ_DIR)
            save_per_cell_bedpe(cell_type, cell_idx, raw_matrix, cfg.PER_CELL_BEDPE_DIR)

        # 3. Build target from cool files (correct target source)
        cool_target, n_cool_cells = aggregate_cool_target(cell_type)
        target_path = os.path.join(cfg.TARGET_DIR, f"{cell_type}_target.npz")
        save_npz(target_path, csr_matrix(cool_target))

        # 4. Sanity check: PCC between raw_x target and cool target
        raw_sum = raw.sum(axis=0)
        raw_target = upper_vector2matrix(raw_sum)
        triu = np.triu_indices(cfg.N_BINS, k=1)
        pcc_raw_cool = pearsonr(cool_target[triu], raw_target[triu])[0]

        s, e = cfg.PDGFRA_SUB_BINS
        cool_8x8 = cool_target[s:e, s:e].copy()
        raw_8x8 = raw_target[s:e, s:e].copy()
        np.fill_diagonal(cool_8x8, 0)
        np.fill_diagonal(raw_8x8, 0)
        pcc_8x8 = pearsonr(cool_8x8.flatten(), raw_8x8.flatten())[0]

        print(f"  Cool target: {n_cool_cells} cells, nnz={np.count_nonzero(cool_target)}/{cfg.N_BINS**2}")
        print(f"  PCC(cool_target, raw_x_target) 49x49={pcc_raw_cool:.4f}, 8x8={pcc_8x8:.4f}")

        metadata["cell_types"][cell_type] = {
            "n_cells_raw_x": n_cells_raw,
            "n_cells_cool": n_cool_cells,
            "cool_target_nnz": int(np.count_nonzero(cool_target)),
            "pcc_cool_vs_raw_49x49": float(pcc_raw_cool),
            "pcc_cool_vs_raw_8x8": float(pcc_8x8),
        }

    meta_path = os.path.join(cfg.INPUT_LEE_DIR, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved: {meta_path}")
    print("\nStage 1 complete.")


if __name__ == "__main__":
    main()
