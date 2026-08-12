#!/usr/bin/env python3
"""Collect scHiCluster Ramani imputed hdf5 into per-chromosome feature NPZ.

Reads per-cell hdf5 files from ``output/1_imputed_hdf5/{chrom}/`` (626 cells,
IDs 1..626 in ML1_ML3_cell_list.txt order), extracts upper-triangle features
via ``np.triu_indices(n_bins, k=1)``, and stacks into ``(626, n_features)``
CSR per chromosome.

Also computes a merged SVD embedding:
  log1p(chrom_npz) → TruncatedSVD(n_components=5, random_state=100) per chrom
  → hstack → (626, 115)

Output:
  output/chrom_npz/chrN.npz       — (626, n_features) CSR
  output/ramani_embedding.npz     — (626, 115) SVD embedding
"""

from __future__ import annotations

import argparse
import math
import os
import re
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz, vstack, load_npz
from sklearn.decomposition import TruncatedSVD


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = BASE_DIR / "input"
DEFAULT_IMPUTED_ROOT = BASE_DIR / "output" / "1_imputed_hdf5"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output"
DEFAULT_CELL_LIST = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/"
    "2-Ramani-GSE84920-ML1-ML3/upper_npz/1000000bp/ML1_ML3_cell_list.txt"
)

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
PAD = 1
STD = 1.0
RP = 0.5
PER_CHROM_DIM = 5
SVD_SEED = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--imputed-root", type=Path, default=DEFAULT_IMPUTED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cell-list", type=Path, default=DEFAULT_CELL_LIST)
    parser.add_argument("--chroms", nargs="*", default=None)
    parser.add_argument("--pad", type=int, default=PAD)
    parser.add_argument("--std", type=float, default=STD)
    parser.add_argument("--rp", type=float, default=RP)
    parser.add_argument("--per-chrom-dim", type=int, default=PER_CHROM_DIM)
    parser.add_argument("--seed", type=int, default=SVD_SEED)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def mode_name(pad: int, std: float, rp: float) -> str:
    return f"pad{pad}_std{std:g}_rp{rp:g}_sqrtvc"


def n_bins_from_features(n_features: int) -> int:
    n = int((1 + math.sqrt(1 + 8 * n_features)) / 2)
    assert n * (n - 1) // 2 == n_features
    return n


_W_UPPER: tuple[np.ndarray, np.ndarray] = (np.empty(0), np.empty(0))
_W_N_BINS: int = 0


def _worker_init(upper_idx, n_bins):
    global _W_UPPER, _W_N_BINS
    _W_UPPER = upper_idx
    _W_N_BINS = n_bins


def _load_cell(args_tuple):
    cell_id, path = args_tuple
    with h5py.File(path, "r") as f:
        m = f["Matrix"]
        shape = tuple(int(s) for s in m.attrs["shape"])
        csr = csr_matrix(
            (m["data"][:], m["indices"][:], m["indptr"][:]),
            shape=shape,
        )
    dense = csr.toarray().astype(np.float64, copy=False)
    nb = _W_N_BINS
    if dense.shape[0] > nb:
        dense = dense[:nb, :nb]
    ui, uj = _W_UPPER
    row = dense[ui, uj].reshape(1, -1)
    row[~np.isfinite(row)] = 0.0
    row[row < 0] = 0.0
    return coo_matrix(row)


def collect_chrom(chrom: str, args, mode: str, workers: int) -> None:
    # Get n_bins from raw input
    raw_path = args.input_root / "raw_626_chrom_npz" / f"{chrom}.npz"
    if not raw_path.exists():
        print(f"[collect] {chrom}: raw input not found, skipping", flush=True)
        return
    raw_mat = load_npz(str(raw_path))
    n_cells, n_features = raw_mat.shape
    n_bins = n_bins_from_features(n_features)
    upper_idx = np.triu_indices(n_bins, k=1)

    chrom_npz_dir = args.output_root / "chrom_npz"
    chrom_npz_dir.mkdir(parents=True, exist_ok=True)
    out_path = chrom_npz_dir / f"{chrom}.npz"
    if out_path.exists() and not args.overwrite:
        print(f"[collect] {chrom}: skip existing {out_path}", flush=True)
        return

    imp_dir = args.imputed_root / chrom
    if not imp_dir.is_dir():
        print(f"[collect] {chrom}: imputed dir {imp_dir} not found, skipping",
              flush=True)
        return

    pattern = re.compile(rf"^cell_(\d+)_{re.escape(chrom)}_{re.escape(mode)}\.hdf5$")
    files = []
    for p in imp_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda t: t[0])
    if not files:
        print(f"[collect] {chrom}: no imputed hdf5 found in {imp_dir}", flush=True)
        return
    print(f"[collect] {chrom}: {len(files)} cells, {n_features} features, "
          f"n_bins={n_bins}", flush=True)

    if workers <= 1:
        _worker_init(upper_idx, n_bins)
        rows = [_load_cell(it) for it in files]
    else:
        with Pool(processes=workers, initializer=_worker_init,
                  initargs=(upper_idx, n_bins)) as pool:
            rows = pool.map(_load_cell, files,
                            chunksize=max(1, len(files) // (workers * 4)))

    stacked = vstack(rows, format="csr")
    save_npz(out_path, stacked)
    print(f"[collect] {chrom}: saved {out_path} shape={stacked.shape}", flush=True)


def main() -> int:
    args = parse_args()
    workers = args.workers
    if workers <= 0:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    chroms = args.chroms or CHROMS
    mode = mode_name(args.pad, args.std, args.rp)

    print(f"[collect] chroms={len(chroms)}, mode={mode}, workers={workers}",
          flush=True)
    for chrom in chroms:
        collect_chrom(chrom, args, mode, workers)

    # SVD embedding
    print("[collect] building SVD embedding...", flush=True)
    with open(args.cell_list) as f:
        cells = [l.strip() for l in f if l.strip()]
    embeddings = []
    for chrom in chroms:
        npz_path = args.output_root / "chrom_npz" / f"{chrom}.npz"
        if not npz_path.exists():
            continue
        mat = load_npz(str(npz_path)).toarray().astype(np.float64)
        log_data = np.log1p(mat)
        svd = TruncatedSVD(n_components=args.per_chrom_dim,
                           random_state=args.seed)
        emb = svd.fit_transform(log_data)
        embeddings.append(emb)
        print(f"  {chrom}: SVD {args.per_chrom_dim} dim, "
              f"explained var={svd.explained_variance_ratio_.sum():.4f}",
              flush=True)

    merged = np.hstack(embeddings).astype(np.float32)
    emb_path = args.output_root / "ramani_embedding.npz"
    np.savez_compressed(emb_path, data=merged,
                        cells=np.asarray(cells, dtype=object))
    print(f"[collect] embedding: {merged.shape} -> {emb_path}", flush=True)
    print("[collect] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
