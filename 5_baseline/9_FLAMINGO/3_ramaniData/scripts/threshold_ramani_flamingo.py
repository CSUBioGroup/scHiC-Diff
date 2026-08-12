#!/usr/bin/env python3
"""Thresholded post-processing for Ramani FLAMINGO completed tensors.

The original FLAMINGO contact-space run fills ~86% of the matrix with a uniform
~1.0 background, which homogenizes cells (per-cell sum CV drops from 0.96 to
0.03) and destroys clustering (ARI ≈ 0). This script re-processes the existing
``completed_tensor.npy`` files by:

1. Thresholding imputed entries: only keep imputed values above ``--threshold``
   (default 1.0). Values below the threshold are set to 0 (no contact).
2. Restoring observed entries from the original RawCount input.
3. Saving thresholded chrom_npz and embedding, compatible with the existing
   clustering evaluation pipeline.

Usage::

    python threshold_ramani_flamingo.py \\
        --output-root output \\
        --threshold 1.0 \\
        --make-embedding --log1p

This does NOT re-run FLAMINGO — it only post-processes existing outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import sparse


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output"
DEFAULT_INPUT_ROOT = BASE_DIR / "input"
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]

# Shared clustering common module
COMMON_DIR = (
    Path("/public/home/hpc254701055/2_projects/10_schicdiff")
    / "1_scHiC/5_baseline/paperplots/4_ramani_clustering_metrics/scripts"
)
sys.path.insert(0, str(COMMON_DIR.parent))
from scripts import ramani_imputation_common as common  # noqa: E402


def load_observed(input_root: Path, chrom: str) -> np.ndarray | None:
    """Load observed contact tensor from input/{chrom}/contact_matrices/."""
    matrix_dir = input_root / chrom / "contact_matrices"
    if not matrix_dir.exists():
        return None
    files = sorted(
        matrix_dir.glob("RawCount_Cell_*.txt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not files:
        return None
    matrices = [np.loadtxt(f, delimiter="\t", dtype=np.float64) for f in files]
    return np.stack(matrices, axis=0)


def threshold_tensor(
    completed: np.ndarray,
    observed: np.ndarray | None,
    threshold: float,
) -> np.ndarray:
    """Apply threshold to imputed entries and restore observed values."""
    out = np.real(completed).astype(np.float64)
    out[~np.isfinite(out)] = 0.0
    out[out < 0] = 0.0

    if observed is not None and observed.shape == out.shape:
        obs_mask = observed > 0
        # Zero out imputed entries below threshold
        imp_mask = ~obs_mask
        out[imp_mask & (out < threshold)] = 0.0
        # Restore observed values
        out[obs_mask] = observed[obs_mask]
    else:
        # No observed data — just threshold everything
        out[out < threshold] = 0.0

    return out


def process_chrom(
    chrom: str,
    output_root: Path,
    input_root: Path,
    threshold: float,
    out_dir: Path,
) -> dict:
    comp_path = output_root / chrom / "completed_tensor.npy"
    if not comp_path.exists():
        return {"chrom": chrom, "status": "missing"}

    completed = np.load(comp_path)
    observed = load_observed(input_root, chrom)
    thresholded = threshold_tensor(completed, observed, threshold)

    upper = np.triu_indices(thresholded.shape[1], k=1)
    sparse.save_npz(
        out_dir / f"{chrom}.npz",
        sparse.csr_matrix(thresholded[:, upper[0], upper[1]]),
    )

    density = np.count_nonzero(thresholded) / thresholded.size
    cell_sums = thresholded.reshape(thresholded.shape[0], -1).sum(axis=1)
    cv = float(np.std(cell_sums) / np.mean(cell_sums)) if np.mean(cell_sums) > 0 else 0.0
    return {
        "chrom": chrom,
        "status": "ok",
        "density": f"{density:.4f}",
        "per_cell_cv": f"{cv:.4f}",
        "max": f"{float(thresholded.max()):.4g}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Imputed values below this are set to 0 (default 1.0)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir for chrom_npz (default: output/chrom_npz_thresh{N})")
    parser.add_argument("--chroms", default=None)
    parser.add_argument("--make-embedding", action="store_true")
    parser.add_argument("--per-chrom-dim", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--log1p", action="store_true", default=True)
    parser.add_argument("--no-log1p", dest="log1p", action="store_false")
    parser.add_argument("--cell-list", type=Path, default=common.DEFAULT_CELL_LIST)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.output_root / f"chrom_npz_thresh{args.threshold}")
    out_dir.mkdir(parents=True, exist_ok=True)

    chroms = common.chroms_from_arg(args.chroms)
    rows = []
    for chrom in chroms:
        row = process_chrom(chrom, args.output_root, args.input_root, args.threshold, out_dir)
        rows.append(row)
        print(f"  {chrom}: {row}")

    # Validate
    summary = common.validate_chrom_npz(out_dir, cell_list_path=args.cell_list, chroms=chroms)
    common.save_validation_json(
        summary,
        args.output_root / f"ramani_flamingo_thresh{args.threshold}_validation.json",
    )

    if args.make_embedding:
        emb_path = args.output_root / f"ramani_embedding_thresh{args.threshold}.npz"
        common.save_embedding_from_chrom_npz(
            out_dir,
            emb_path,
            cell_list_path=args.cell_list,
            chroms=chroms,
            per_chrom_dim=args.per_chrom_dim,
            seed=args.seed,
            log1p=args.log1p,
        )
        common.write_manifest_snippet(
            args.output_root / f"ramani_method_thresh{args.threshold}_manifest.csv",
            method=f"Tensor-FLAMINGO-thresh{args.threshold}",
            display_name=f"FLAMINGO (thresh={args.threshold})",
            source_type="embedding",
            source_path=emb_path,
            per_chrom_dim=args.per_chrom_dim,
            log1p=args.log1p,
            notes=f"Thresholded at {args.threshold} to remove uniform background.",
        )
        print(f"Embedding: {emb_path}")

    print(f"chrom_npz: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())