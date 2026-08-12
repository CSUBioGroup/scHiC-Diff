#!/usr/bin/env python3
"""Prepare 1Mb case-study h5ad data for Tensor-FLAMINGO contact-space imputation.

For each chromosome h5ad:
  - Read sparse X (cells x triu_k0_features) and var/bin1_id, bin2_id
  - Reconstruct symmetric contact matrix per cell (nonneg, diag=0)
  - Write RawCount_Cell_NNN.txt (the format the shared LRTC runner expects)
  - Save observed_contact_tensor.npy and metadata.json

A .complete marker prevents duplicate runs unless --force.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


CASE_DIR = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "9_FLAMINGO/6_caseStudy/1_cluster1mb"
)
DEFAULT_INPUT_ROOT = CASE_DIR.parent.parent.parent / "paperplots/7_caseData/1_cluster1mb/inputData/6_hires_h5ad_1mb"
DEFAULT_WORK_ROOT = CASE_DIR / "work"

# Chromosomes present in the input directory (chr20-22 missing from h5ad).
CHROMOSOMES = [f"chr{i}" for i in range(1, 20)] + ["chrX", "chrY"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    p.add_argument("--chroms", nargs="*", default=None,
                   help="Subset of chromosomes (default: all found)")
    p.add_argument("--force", action="store_true",
                   help="Re-prepare even if .complete exists")
    return p.parse_args(argv)


def load_h5ad(path: Path) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, int, int]:
    """Return (X_csr, bin1_id, bin2_id, n_cells, n_bins)."""
    with h5py.File(path, "r") as f:
        data = f["X/data"][:]
        indices = f["X/indices"][:]
        indptr = f["X/indptr"][:]
        n_cells = f["obs/_index"].shape[0]
        n_features = f["var/_index"].shape[0]
        bin1 = f["var/bin1_id"][:]
        bin2 = f["var/bin2_id"][:]
    X = sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_features))
    n_bins = int(max(bin1.max(), bin2.max())) + 1
    return X, bin1.astype(int), bin2.astype(int), n_cells, n_bins


def reconstruct_symmetric(features: np.ndarray, bin1: np.ndarray, bin2: np.ndarray,
                          n_bins: int) -> np.ndarray:
    """Build a nonneg symmetric matrix with zero diagonal from triu(k=0) features."""
    M = np.zeros((n_bins, n_bins), dtype=np.float64)
    M[bin1, bin2] = features.astype(np.float64)
    M[bin2, bin1] = features.astype(np.float64)
    M[~np.isfinite(M)] = 0.0
    M[M < 0] = 0.0
    np.fill_diagonal(M, 0.0)
    return M


def prepare_chrom(chrom: str, args: argparse.Namespace) -> Path:
    input_path = args.input_root / f"hires_1Mb_{chrom}.h5ad"
    if not input_path.exists():
        print(f"[prep] {chrom}: SKIP (no h5ad at {input_path})", flush=True)
        return Path()

    input_dir = args.work_root / "input" / chrom
    matrix_dir = input_dir / "contact_matrices"
    complete_marker = input_dir / ".complete"
    observed_path = input_dir / "observed_contact_tensor.npy"

    if complete_marker.exists() and observed_path.exists() and not args.force:
        print(f"[prep] {chrom}: already prepared (use --force to redo)", flush=True)
        return input_dir

    matrix_dir.mkdir(parents=True, exist_ok=True)

    X, bin1, bin2, n_cells, n_bins = load_h5ad(input_path)
    print(f"[prep] {chrom}: {n_cells} cells, {n_bins} bins, "
          f"{X.nnz} total nnz", flush=True)

    observed_tensor = np.zeros((n_cells, n_bins, n_bins), dtype=np.float64)
    for cell_idx in range(n_cells):
        row = X[cell_idx].toarray().flatten()
        M = reconstruct_symmetric(row, bin1, bin2, n_bins)
        observed_tensor[cell_idx] = M
        dest = matrix_dir / f"RawCount_Cell_{cell_idx + 1:03d}.txt"
        np.savetxt(dest, M, fmt="%.10g", delimiter="\t")
        if (cell_idx + 1) % 1000 == 0:
            print(f"[prep] {chrom}: {cell_idx + 1}/{n_cells}", flush=True)

    np.save(observed_path, observed_tensor)

    metadata = {
        "chromosome": chrom,
        "resolution": 1000000,
        "n_cells": n_cells,
        "n_bins": n_bins,
        "method": "Tensor-FLAMINGO",
        "space": "contact_count",
        "source_h5ad": str(input_path),
        "input_subdir": "contact_matrices",
    }
    with (input_dir / "metadata.json").open("w") as fh:
        json.dump(metadata, fh, indent=2)
    complete_marker.write_text("complete\n")

    print(f"[prep] {chrom}: done, wrote {n_cells} matrices to {matrix_dir}",
          flush=True)
    return input_dir


def main() -> int:
    args = parse_args()
    chroms = args.chroms or [c for c in CHROMOSOMES
                             if (args.input_root / f"hires_1Mb_{c}.h5ad").exists()]
    print(f"[prep] chromosomes: {chroms}", flush=True)
    for chrom in chroms:
        prepare_chrom(chrom, args)
    print("[prep] all done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())