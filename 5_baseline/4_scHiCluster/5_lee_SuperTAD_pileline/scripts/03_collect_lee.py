#!/usr/bin/env python3
"""Collect per-cell scHiCluster imputed NPZ into per-cell-type feature NPZ.

Each ``cell_<id>_<chrom>_<mode>.npz`` is a sparse CSR N x N imputed contact
matrix.  We extract the upper-triangle features (i<j) via
``np.triu_indices(n_bins, k=1)`` and stack into a ``(n_cells, n_features)``
sparse matrix saved as ``<CellType>_scHiCluster_imputed.npz`` under
``output/2_upper_tri_npz``.

Data geometry:
  n_bins      = 49
  n_features  = 49*48/2 = 1176
"""

from __future__ import annotations

import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, load_npz, save_npz, vstack


DEFAULT_INPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "4_scHiCluster/5_lee_SuperTAD_pileline/result"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "4_scHiCluster/5_lee_SuperTAD_pileline/output"
)

CHROM = "chr4"
N_BINS = 49
PAD = 1
STD = 1.0
RP = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect scHiCluster per-cell NPZ into per-cell-type feature NPZ."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chrom", default=CHROM)
    parser.add_argument("--n-bins", type=int, default=N_BINS)
    parser.add_argument("--pad", type=int, default=PAD)
    parser.add_argument("--std", type=float, default=STD)
    parser.add_argument("--rp", type=float, default=RP)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def mode_name(pad: int, std: float, rp: float) -> str:
    return f"pad{pad}_std{std:g}_rp{rp:g}_sqrtvc"


_W_UPPER: tuple[np.ndarray, np.ndarray] = (np.empty(0, dtype=np.int64),
                                            np.empty(0, dtype=np.int64))
_W_N_BINS: int = N_BINS


def _worker_init(upper_idx: tuple[np.ndarray, np.ndarray], n_bins: int) -> None:
    global _W_UPPER, _W_N_BINS
    _W_UPPER = upper_idx
    _W_N_BINS = n_bins


def _load_cell(args: tuple[int, Path]) -> coo_matrix:
    cell_id, path = args
    arr = load_npz(path).toarray().astype(np.float64, copy=False)
    nb = _W_N_BINS
    if arr.shape[0] > nb:
        arr = arr[:nb, :nb]
    ui, uj = _W_UPPER
    row = arr[ui, uj].reshape(1, -1)
    return coo_matrix(row)


def collect_dataset(dataset: str, args: argparse.Namespace, mode: str,
                    upper_idx: tuple[np.ndarray, np.ndarray]) -> None:
    out_dir = args.output_root / "2_upper_tri_npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_scHiCluster_imputed.npz"
    if out_path.exists() and not args.overwrite:
        print(f"[collect] {dataset}: skip existing {out_path}", flush=True)
        return

    imp_dir = args.input_root / dataset
    if not imp_dir.is_dir():
        raise FileNotFoundError(imp_dir)
    pattern = re.compile(rf"^cell_(\d+)_{re.escape(args.chrom)}_{re.escape(mode)}\.npz$")
    files = []
    for p in imp_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda t: t[0])
    if not files:
        raise ValueError(f"No imputed npz in {imp_dir} matching {pattern.pattern}")
    n_cells = len(files)
    n_features = len(upper_idx[0])
    print(f"[collect] {dataset}: {n_cells} cells, {n_features} features", flush=True)

    workers = args.workers
    if workers <= 0:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    items = [(cid, p) for cid, p in files]
    if workers <= 1:
        _worker_init(upper_idx, args.n_bins)
        rows = [_load_cell(it) for it in items]
    else:
        with Pool(processes=workers, initializer=_worker_init,
                  initargs=(upper_idx, args.n_bins)) as pool:
            rows = pool.map(_load_cell, items,
                            chunksize=max(1, len(items) // (workers * 4)))
    stacked = vstack(rows, format="coo")
    save_npz(out_path, stacked)
    print(f"[collect] {dataset}: saved {out_path} shape={stacked.shape}", flush=True)


def discover_datasets(input_root: Path) -> list[str]:
    return sorted(d.name for d in input_root.iterdir() if d.is_dir())


def main() -> int:
    args = parse_args()
    mode = mode_name(args.pad, args.std, args.rp)
    datasets = args.datasets or discover_datasets(args.input_root)
    if not datasets:
        raise ValueError(f"No dataset dirs in {args.input_root}")
    upper_idx = np.triu_indices(args.n_bins, k=1)
    print(f"[collect] mode={mode}, n_bins={args.n_bins}, "
          f"n_features={len(upper_idx[0])}, datasets={len(datasets)}", flush=True)
    for ds in datasets:
        collect_dataset(ds, args, mode, upper_idx)
    print("[collect] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())