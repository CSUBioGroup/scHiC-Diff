#!/usr/bin/env python3
"""Collect FLAMINGO completed tensors into per-chromosome sparse NPZ.

Each completed_tensor.npy has shape (n_cells, n_bins, n_bins) with a symmetric
nonneg contact matrix per cell. We extract the upper triangle (k=1, excluding
diagonal) per cell and stack into a (n_cells, n_features) sparse CSR matrix,
saved as <chrom>.npz under the imputedData/FLAMINGO directory.

This matches the scVI-3D / input_by_chr_binpairs_npz format:
  - triu(k=1), excluding diagonal
  - (n_cells, n_bins*(n_bins-1)/2)
  - scipy.sparse.csr_matrix saved via save_npz

The bin-pair order follows NumPy's np.triu_indices(n_bins, k=1) row-major order.

Also dumps a per-cell dense (n_cells, n_bins, n_bins) .npy under
imputedData/FLAMINGO/contact_tensors/ for downstream APA / loop calling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse


CASE_DIR = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "9_FLAMINGO/6_caseStudy/2_call_loops"
)
DEFAULT_WORK_ROOT = CASE_DIR / "work"
DEFAULT_OUTPUT_ROOT = CASE_DIR / "imputedData" / "FLAMINGO"

CHROMOSOMES = ["chr1"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--chroms", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-dense-tensor", action="store_true", default=True,
                   help="Also save (n_cells, n_bins, n_bins) dense .npy for APA/loop calling")
    return p.parse_args(argv)


def collect_chrom(chrom: str, args: argparse.Namespace,
                  triu_i: np.ndarray, triu_j: np.ndarray) -> None:
    output_dir = args.output_root / "chrom_npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{chrom}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"[collect] {chrom}: skip existing {out_path}", flush=True)
        return

    output_subdir = args.work_root / "output" / chrom
    completed_path = output_subdir / "completed_tensor.npy"
    observed_path = args.work_root / "input" / chrom / "observed_contact_tensor.npy"

    if not completed_path.exists():
        print(f"[collect] {chrom}: SKIP (no completed_tensor at {completed_path})",
              flush=True)
        return

    completed = np.load(completed_path)
    n_cells, n_bins, _ = completed.shape
    n_features = len(triu_i)
    print(f"[collect] {chrom}: {n_cells} cells, {n_bins} bins, "
          f"{n_features} features", flush=True)

    completed = np.real(completed).astype(np.float64)
    completed[~np.isfinite(completed)] = 0.0
    completed[completed < 0] = 0.0
    for idx in range(n_cells):
        np.fill_diagonal(completed[idx], 0.0)
        completed[idx] = np.maximum(completed[idx], completed[idx].T)

    if observed_path.exists():
        observed = np.load(observed_path)
        omega = observed > 0
        completed[omega] = observed[omega]

    if args.save_dense_tensor:
        tensor_dir = args.output_root / "contact_tensors"
        tensor_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = tensor_dir / f"{chrom}_completed.npy"
        np.save(tensor_path, completed)
        print(f"[collect] {chrom}: saved dense tensor {tensor_path} "
              f"shape={completed.shape}", flush=True)

    features = completed[:, triu_i, triu_j]

    mat = sparse.csr_matrix(features)
    sparse.save_npz(out_path, mat)
    print(f"[collect] {chrom}: saved {out_path} shape={mat.shape} "
          f"nnz={mat.nnz}", flush=True)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    chroms = args.chroms or [c for c in CHROMOSOMES
                             if (args.work_root / "output" / c / "completed_tensor.npy").exists()]
    if not chroms:
        print("[collect] no completed chromosomes found", flush=True)
        return 1

    print(f"[collect] chromosomes: {chroms}", flush=True)
    for chrom in chroms:
        meta_path = args.work_root / "input" / chrom / "metadata.json"
        if meta_path.exists():
            with meta_path.open() as fh:
                n_bins = json.load(fh)["n_bins"]
        else:
            completed = np.load(args.work_root / "output" / chrom / "completed_tensor.npy",
                                mmap_mode="r")
            n_bins = completed.shape[1]
        triu_i, triu_j = np.triu_indices(n_bins, k=1)
        collect_chrom(chrom, args, triu_i, triu_j)

    print("[collect] all done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())