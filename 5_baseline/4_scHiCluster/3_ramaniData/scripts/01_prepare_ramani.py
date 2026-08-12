#!/usr/bin/env python3
"""Prepare scHiCluster input from 626-cell Ramani raw NPZ.

For each chromosome, reads the (626, n_features) CSR matrix from
``raw_626_chrom_npz/{chrom}.npz``, reconstructs the full NxN symmetric
contact matrix per cell, and writes non-zero upper-triangle contacts as
``cell_{id}_{chrom}.txt`` (row, col, value tab-separated).

Also writes ``ramani_{chrom}.chrom.sizes`` per chromosome.

Cell IDs 1..626 follow ``ML1_ML3_cell_list.txt`` order.
"""

from __future__ import annotations

import argparse
import math
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = BASE_DIR / "input"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "input" / "schicluster_input"
DEFAULT_CELL_LIST = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/"
    "2-Ramani-GSE84920-ML1-ML3/upper_npz/1000000bp/ML1_ML3_cell_list.txt"
)

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cell-list", type=Path, default=DEFAULT_CELL_LIST)
    parser.add_argument("--chroms", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def n_bins_from_features(n_features: int) -> int:
    n = int((1 + math.sqrt(1 + 8 * n_features)) / 2)
    assert n * (n - 1) // 2 == n_features
    return n


_W_UPPER: tuple[np.ndarray, np.ndarray] = (np.empty(0), np.empty(0))
_W_OUT_DIR: Path = Path()
_W_CHROM: str = ""
_W_MATRIX: np.ndarray = np.empty((0, 0))
_W_OVERWRITE: bool = False


def _worker_init(upper_idx, out_dir, chrom, matrix, overwrite):
    global _W_UPPER, _W_OUT_DIR, _W_CHROM, _W_MATRIX, _W_OVERWRITE
    _W_UPPER = upper_idx
    _W_OUT_DIR = out_dir
    _W_CHROM = chrom
    _W_MATRIX = matrix
    _W_OVERWRITE = overwrite


def _write_cell(cell_id_1based: int) -> int:
    cell_id = cell_id_1based - 1
    row_vec = _W_MATRIX[cell_id]
    ui, uj = _W_UPPER
    keep = np.nonzero(row_vec)[0]
    dest = _W_OUT_DIR / f"cell_{cell_id_1based}_{_W_CHROM}.txt"
    if dest.exists() and not _W_OVERWRITE:
        return cell_id_1based, -1
    dest.parent.mkdir(parents=True, exist_ok=True)
    if keep.size == 0:
        dest.write_text("")
        return cell_id_1based, 0
    r = ui[keep]
    c = uj[keep]
    v = row_vec[keep]
    np.savetxt(dest, np.column_stack((r, c, v)),
               fmt=["%d", "%d", "%.10g"], delimiter="\t")
    return cell_id_1based, int(keep.size)


def prepare_chrom(chrom: str, args: argparse.Namespace, workers: int) -> None:
    src = args.input_root / "raw_626_chrom_npz" / f"{chrom}.npz"
    if not src.exists():
        print(f"[prepare] {chrom}: WARNING {src} not found, skipping", flush=True)
        return
    matrix = load_npz(src).toarray().astype(np.float64)
    n_cells, n_features = matrix.shape
    n_bins = n_bins_from_features(n_features)
    print(f"[prepare] {chrom}: {n_cells} cells, {n_bins} bins, "
          f"{n_features} features", flush=True)

    out_dir = args.output_root / chrom
    out_dir.mkdir(parents=True, exist_ok=True)
    chrom_file = out_dir / f"ramani_{chrom}.chrom.sizes"
    chrom_file.write_text(f"{chrom}\t{n_bins - 1}\n")

    upper_idx = np.triu_indices(n_bins, k=1)
    init_kwargs = (upper_idx, out_dir, chrom, matrix, args.overwrite)
    items = list(range(1, n_cells + 1))
    total_nnz = 0
    if workers <= 1:
        _worker_init(*init_kwargs)
        for cid in items:
            _, nnz = _write_cell(cid)
            if nnz >= 0:
                total_nnz += nnz
            if cid % 100 == 0:
                print(f"  {chrom}: cell {cid}/{n_cells}", flush=True)
    else:
        with Pool(processes=workers, initializer=_worker_init,
                  initargs=init_kwargs) as pool:
            for cid, nnz in pool.imap_unordered(_write_cell, items,
                    chunksize=max(1, len(items) // (workers * 4))):
                if nnz >= 0:
                    total_nnz += nnz
    print(f"[prepare] {chrom}: done, nnz={total_nnz}", flush=True)


def main() -> int:
    args = parse_args()
    chroms = args.chroms or CHROMS
    workers = args.workers
    if workers <= 0:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] chroms={chroms}, workers={workers}", flush=True)
    for chrom in chroms:
        prepare_chrom(chrom, args, workers)
    print("[prepare] all done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
