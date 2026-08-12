#!/usr/bin/env python3
"""Collect per-cell scHiCluster imputed hdf5 into per-dataset feature NPZ.

Each ``cell_<id>_<chrom>_<mode>.hdf5`` stores a dense imputed contact matrix
as a CSR sparse matrix under the ``Matrix`` group (with ``data``,
``indices``, ``indptr`` sub-datasets and a ``shape`` attribute).  The hdf5
matrix is ``impute_bins x impute_bins`` (62x62) because scHiCluster read
``chrom.sizes`` value 61 and created 62 bins; only the first ``n_bins``
(61) rows/cols carry real signal, so we extract the top-left ``n_bins x
n_bins`` submatrix.

The Simu_Data simulation stores features in **lower-triangle** order
(``np.tril_indices(n_bins, k=-1)``), matching the GT/sim NPZ files from the
dxy processed data.  This is different from the FLAMINGO v3 pipeline which
uses upper-triangle ordering.

Features are stacked into a ``(n_cells, n_features)`` sparse matrix saved as
``<dataset>_scHiCluster_imputed.npz`` under ``output/2_lower_tri_npz``.
"""

from __future__ import annotations

import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz, vstack


DEFAULT_INPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "4_scHiCluster/1_HiCImputeDate/result/1_Simu_Data"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "4_scHiCluster/1_HiCImputeDate/output"
)

CHROM = "chr19"
N_BINS = 61
IMPUTE_BINS = 62
PAD = 1
STD = 1.0
RP = 0.5

DATASETS = [f"K562_T{t}_{d}" for t in (1, 2, 3) for d in ("1k", "2k", "4k", "7k")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect scHiCluster per-cell hdf5 into per-dataset lower-tri NPZ."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chrom", default=CHROM)
    parser.add_argument("--n-bins", type=int, default=N_BINS,
                        help="Real number of bins (61); extract top-left submatrix")
    parser.add_argument("--impute-bins", type=int, default=IMPUTE_BINS,
                        help="hdf5 matrix size (62); scHiCluster padded chrom.sizes")
    parser.add_argument("--pad", type=int, default=PAD)
    parser.add_argument("--std", type=float, default=STD)
    parser.add_argument("--rp", type=float, default=RP)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def mode_name(pad: int, std: float, rp: float) -> str:
    return f"pad{pad}_std{std:g}_rp{rp:g}_sqrtvc"


def discover_datasets(input_root: Path) -> list[str]:
    found = []
    for d in sorted(input_root.iterdir()):
        if d.is_dir() and d.name.startswith("K562_T"):
            found.append(d.name)
    return found or list(DATASETS)


_W_LOWER: tuple[np.ndarray, np.ndarray] = (np.empty(0, dtype=np.int64),
                                             np.empty(0, dtype=np.int64))
_W_N_BINS: int = N_BINS
_W_IMPUTE_BINS: int = IMPUTE_BINS


def _worker_init(lower_idx: tuple[np.ndarray, np.ndarray],
                 n_bins: int, impute_bins: int) -> None:
    global _W_LOWER, _W_N_BINS, _W_IMPUTE_BINS
    _W_LOWER = lower_idx
    _W_N_BINS = n_bins
    _W_IMPUTE_BINS = impute_bins


def _load_cell(args: tuple[int, Path]) -> coo_matrix:
    cell_id, path = args
    with h5py.File(path, "r") as f:
        m = f["Matrix"]
        shape = tuple(int(s) for s in m.attrs["shape"])
        csr = csr_matrix(
            (m["data"][:], m["indices"][:], m["indptr"][:]),
            shape=shape,
        )
    dense = csr.toarray().astype(np.float64, copy=False)
    # Extract the meaningful n_bins x n_bins submatrix
    nb = _W_N_BINS
    if dense.shape[0] > nb:
        dense = dense[:nb, :nb]
    li, lj = _W_LOWER
    # scHiCluster stores the upper triangle (i<j) in the hdf5; the lower
    # triangle is zero.  The Simu_Data GT uses lower-triangle ordering
    # (i>j via np.tril_indices).  For a symmetric contact matrix the value
    # at (i,j) equals (j,i), so we read the transposed positions to get
    # the lower-triangle feature values from the upper-triangle storage.
    row = dense[lj, li].reshape(1, -1)
    return coo_matrix(row)


def collect_dataset(dataset: str, args: argparse.Namespace, mode: str,
                    lower_idx: tuple[np.ndarray, np.ndarray]) -> None:
    out_dir = args.output_root / "2_lower_tri_npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_scHiCluster_imputed.npz"
    if out_path.exists() and not args.overwrite:
        print(f"[collect] {dataset}: skip existing {out_path}", flush=True)
        return

    imp_dir = args.input_root / dataset
    if not imp_dir.is_dir():
        raise FileNotFoundError(imp_dir)
    pattern = re.compile(
        rf"^cell_(\d+)_{re.escape(args.chrom)}_{re.escape(mode)}\.hdf5$"
    )
    files = []
    for p in imp_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda t: t[0])
    if not files:
        raise ValueError(
            f"No imputed hdf5 in {imp_dir} matching {pattern.pattern}"
        )
    n_cells = len(files)
    n_features = len(lower_idx[0])
    print(f"[collect] {dataset}: {n_cells} cells, {n_features} features",
          flush=True)

    workers = args.workers
    if workers <= 0:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    items = [(cid, p) for cid, p in files]
    if workers <= 1:
        _worker_init(lower_idx, args.n_bins, args.impute_bins)
        rows = [_load_cell(it) for it in items]
    else:
        with Pool(processes=workers, initializer=_worker_init,
                  initargs=(lower_idx, args.n_bins, args.impute_bins)) as pool:
            rows = pool.map(_load_cell, items,
                            chunksize=max(1, len(items) // (workers * 4)))
    stacked = vstack(rows, format="coo")
    save_npz(out_path, stacked)
    print(f"[collect] {dataset}: saved {out_path} shape={stacked.shape}",
          flush=True)


def main() -> int:
    args = parse_args()
    mode = mode_name(args.pad, args.std, args.rp)
    datasets = args.datasets or discover_datasets(args.input_root)
    if not datasets:
        raise ValueError(f"No dataset dirs in {args.input_root}")
    lower_idx = np.tril_indices(args.n_bins, k=-1)
    print(f"[collect] mode={mode}, n_bins={args.n_bins}, "
          f"impute_bins={args.impute_bins}, datasets={len(datasets)}",
          flush=True)
    for ds in datasets:
        collect_dataset(ds, args, mode, lower_idx)
    print("[collect] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
