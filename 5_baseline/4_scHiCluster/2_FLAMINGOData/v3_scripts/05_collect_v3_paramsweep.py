#!/usr/bin/env python3
"""Collect per-cell scHiCluster imputed NPZ into per-dataset feature NPZ.

Each ``cell_<id>_<chrom>_<mode>.npz`` is a dense N x N imputed contact matrix.
We extract the upper-triangle features (i<j, row-major over i) which matches
the FLAMINGO h5ad ``layers['gt']`` / ``layers['counts']`` feature ordering
(var_names ``chrFLAMINGO_i_j`` with ``i < j``, k = i*(2N-1-i)//2 + (j-i-1)).
The features are stacked into a ``(n_cells, n_features)`` sparse matrix saved
as ``<dataset>_scHiCluster_imputed.npz`` under ``v3_outputData/2_lower_tri_npz``
(the directory name is kept for consistency with the reference pipeline).
"""

from __future__ import annotations

import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, save_npz, vstack


DEFAULT_INPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "4_scHiCluster/2_FLAMINGOData/v3_inputData"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "4_scHiCluster/2_FLAMINGOData/v3_outputData"
)

CHROM = "chr19"
N_BINS = 500
PAD = 1
STD = 1.0
RP = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect scHiCluster per-cell NPZ into per-dataset lower-tri NPZ."
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


def _worker_init(upper_idx: tuple[np.ndarray, np.ndarray]) -> None:
    global _W_UPPER
    _W_UPPER = upper_idx


def _load_lower(args: tuple[int, Path, int]) -> coo_matrix:
    cell_id, path, n_bins = args
    from scipy.sparse import load_npz
    arr = load_npz(path).toarray().astype(np.float64, copy=False)
    if arr.shape != (n_bins, n_bins):
        raise ValueError(f"{path} shape {arr.shape} != ({n_bins},{n_bins})")
    ui, uj = _W_UPPER
    row = arr[ui, uj].reshape(1, -1)
    return coo_matrix(row)


def collect_dataset(dataset: str, args: argparse.Namespace, mode: str,
                    upper_idx: tuple[np.ndarray, np.ndarray]) -> None:
    out_dir = args.output_root / "2_lower_tri_npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_scHiCluster_imputed.npz"
    if out_path.exists() and not args.overwrite:
        print(f"[collect] {dataset}: skip existing {out_path}", flush=True)
        return

    imp_dir = args.output_root / "1_imputed_npz" / dataset
    if not imp_dir.is_dir():
        raise FileNotFoundError(imp_dir)
    # Discover imputed cell files and sort by cell id.
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
    print(f"[collect] {dataset}: {n_cells} cells, {n_features} features",
          flush=True)

    workers = args.workers
    if workers <= 0:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    items = [(cid, p, args.n_bins) for cid, p in files]
    if workers <= 1:
        _worker_init(upper_idx)
        rows = [_load_lower(it) for it in items]
    else:
        with Pool(processes=workers, initializer=_worker_init,
                  initargs=(upper_idx,)) as pool:
            rows = pool.map(_load_lower, items,
                            chunksize=max(1, len(items) // (workers * 4)))
    stacked = vstack(rows, format="coo")
    save_npz(out_path, stacked)
    print(f"[collect] {dataset}: saved {out_path} shape={stacked.shape}",
          flush=True)


def discover_datasets(input_root: Path) -> list[str]:
    return sorted(d.name for d in input_root.iterdir() if d.is_dir())


def main() -> int:
    args = parse_args()
    mode = mode_name(args.pad, args.std, args.rp)
    datasets = args.datasets or discover_datasets(args.input_root)
    if not datasets:
        raise ValueError(f"No dataset dirs in {args.input_root}")
    lower_idx = np.triu_indices(args.n_bins, k=1)
    for ds in datasets:
        collect_dataset(ds, args, mode, lower_idx)
    print("[collect] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())