#!/usr/bin/env python3
"""Prepare FLAMINGO v3 paramsweep h5ad datasets for scHiCluster input.

Each h5ad stores the per-cell lower-triangle contact feature vector in
``layers['counts']`` (observed, noisy contacts used as scHiCluster input) and
``layers['gt']`` (clean ground truth).  Feature ``k`` maps to the upper-triangle
matrix coordinate ``(i, j)`` with ``i < j`` over an N x N contact matrix where
``N = 500`` and the feature index follows row-major ordering
``k = i*(2N-1-i)//2 + (j - i - 1)``.

This script reconstructs the full NxN contact matrix per cell from the counts
layer, extracts the non-zero upper-triangle entries (row col value) and writes
one ``cell_<id>_<chrom>.txt`` per cell plus a ``simu_<chrom>.chrom.sizes`` file,
matching the input format expected by ``hicluster impute-cell``.
"""

from __future__ import annotations

import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

import anndata as ad
import numpy as np


DEFAULT_DATA_DIR = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/"
    "5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/"
    "3_fixed_flamingoGen_datasets/5_paramsweep_datasets"
)
DEFAULT_INPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "4_scHiCluster/2_FLAMINGOData/v3_inputData"
)
CHROM = "chr19"
N_BINS = 500
LAYER_INPUT = "counts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FLAMINGO v3 paramsweep h5ad to scHiCluster input."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--chrom", default=CHROM)
    parser.add_argument("--n-bins", type=int, default=N_BINS)
    parser.add_argument("--layer", default=LAYER_INPUT)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Dataset stems to process; default: all *_scdiff2.h5ad")
    parser.add_argument("--max-cells", type=int, default=0,
                        help="Cap cells per dataset; 0 means all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers for cell writing; 0 = SLURM_CPUS_PER_TASK or 1")
    parser.add_argument("--single-dataset", default=None,
                        help="Process only this dataset stem (used by SLURM array jobs)")
    return parser.parse_args()


def discover_datasets(data_dir: Path) -> list[str]:
    stems = []
    for path in sorted(data_dir.glob("v3_hybrid_*_scdiff2.h5ad")):
        name = path.name
        m = re.match(r"^(v3_hybrid_.+)_scdiff2\.h5ad$", name)
        if m:
            stems.append(m.group(1))
    return stems


def upper_triangle_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def cell_id_digits(n_cells: int) -> int:
    return max(3, len(str(n_cells)))


def write_cell_file(row: np.ndarray, i_idx: np.ndarray, j_idx: np.ndarray,
                   dest: Path, overwrite: bool) -> int:
    if dest.exists() and not overwrite:
        return -1
    keep = np.nonzero(row)[0]
    if keep.size == 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("")
        return 0
    r = i_idx[keep]
    c = j_idx[keep]
    v = row[keep]
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(dest, np.column_stack((r, c, v)),
               fmt=["%d", "%d", "%.10g"], delimiter="\t")
    return int(keep.size)


# Worker globals set by the initializer for the multiprocessing pool.
_W_I: np.ndarray = np.empty(0, dtype=np.int64)
_W_J: np.ndarray = np.empty(0, dtype=np.int64)
_W_OVERWRITE: bool = False
_W_DIGITS: int = 3
_W_CHROM: str = CHROM
_W_DIR: Path = Path()
_W_H5AD: Path = Path()
_W_LAYER_NAME: str = "counts"
_W_ADATA = None


def _worker_open() -> None:
    global _W_ADATA
    if _W_ADATA is None:
        _W_ADATA = ad.read_h5ad(_W_H5AD, backed="r")


def _worker_init(kwargs: dict) -> None:
    global _W_I, _W_J, _W_OVERWRITE, _W_DIGITS, _W_CHROM, _W_DIR, _W_H5AD, _W_LAYER_NAME
    _W_I = kwargs["i_idx"]
    _W_J = kwargs["j_idx"]
    _W_OVERWRITE = kwargs["overwrite"]
    _W_DIGITS = kwargs["digits"]
    _W_CHROM = kwargs["chrom"]
    _W_DIR = kwargs["out_dir"]
    _W_H5AD = kwargs["h5ad"]
    _W_LAYER_NAME = kwargs["layer_name"]


def _worker_cell(cell_id: int) -> tuple[int, int]:
    _worker_open()
    row = _W_ADATA.layers[_W_LAYER_NAME][cell_id - 1:cell_id]
    if hasattr(row, "toarray"):
        row = row.toarray()
    row = np.asarray(row, dtype=np.float64).ravel()
    dest = _W_DIR / f"cell_{cell_id}_{_W_CHROM}.txt"
    nnz = write_cell_file(row, _W_I, _W_J, dest, _W_OVERWRITE)
    return cell_id, nnz


def _worker_close() -> None:
    global _W_ADATA
    if _W_ADATA is not None:
        try:
            _W_ADATA.file.close()
        except Exception:
            pass
        _W_ADATA = None


def prepare_dataset(stem: str, data_dir: Path, input_root: Path,
                    chrom: str, n_bins: int, layer: str,
                    max_cells: int, overwrite: bool, workers: int) -> None:
    h5ad_path = data_dir / f"{stem}_scdiff2.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(h5ad_path)
    print(f"[prepare] {stem}: opening (backed) {h5ad_path.name}", flush=True)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    n_total = adata.shape[0]
    selected = n_total if max_cells == 0 else min(max_cells, n_total)
    print(f"[prepare] {stem}: {n_total} cells available, using {selected}",
          flush=True)

    i_idx, j_idx = upper_triangle_indices(n_bins)
    n_features = i_idx.size
    if adata.shape[1] != n_features:
        raise ValueError(
            f"{stem}: feature count {adata.shape[1]} != expected {n_features}"
        )

    out_dir = input_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    chrom_file = out_dir / f"simu_{chrom}.chrom.sizes"
    chrom_file.write_text(f"{chrom}\t{n_bins - 1}\n")

    digits = cell_id_digits(selected)
    init_kwargs = {
        "i_idx": i_idx,
        "j_idx": j_idx,
        "out_dir": out_dir,
        "chrom": chrom,
        "digits": digits,
        "overwrite": overwrite,
        "h5ad": h5ad_path,
        "layer_name": layer,
    }
    items = list(range(1, selected + 1))
    total_nnz = 0
    if workers <= 1:
        _worker_init(init_kwargs)
        try:
            for cell_id in items:
                _, nnz = _worker_cell(cell_id)
                if nnz >= 0:
                    total_nnz += nnz
                if cell_id % 100 == 0:
                    print(f"[prepare] {stem}: cell {cell_id}/{selected}", flush=True)
        finally:
            _worker_close()
    else:
        with Pool(processes=workers, initializer=_worker_init,
                  initargs=(init_kwargs,)) as pool:
            done = 0
            for cell_id, nnz in pool.imap_unordered(_worker_cell, items,
                                                    chunksize=max(1, len(items) // (workers * 4))):
                if nnz >= 0:
                    total_nnz += nnz
                done += 1
                if done % 100 == 0:
                    print(f"[prepare] {stem}: {done}/{selected} cells done",
                          flush=True)
    adata.file.close()
    print(f"[prepare] {stem}: {selected} cells -> {out_dir} (chrom {chrom}, "
          f"{n_bins} bins, total nnz written={total_nnz})", flush=True)


def write_manifest(stems: list[str], input_root: Path, chrom: str,
                   n_bins: int, max_cells: int, data_dir: Path) -> None:
    manifest = input_root / "manifest.tsv"
    # Preserve existing manifest entries and only add/update the requested stems.
    existing: dict[str, str] = {}
    if manifest.exists():
        for line_no, line in enumerate(manifest.read_text().splitlines(), start=1):
            if line_no == 1:
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                existing[parts[0]] = line
    lines = ["dataset\tn_bins\tn_cells\th5ad"]
    requested = set(stems)
    kept = 0
    for stem, line in existing.items():
        if stem not in requested:
            lines.append(line)
            kept += 1
    for stem in stems:
        h5ad = data_dir / f"{stem}_scdiff2.h5ad"
        adata = ad.read_h5ad(h5ad, backed="r")
        n_cells = adata.shape[0]
        n_cells = n_cells if max_cells == 0 else min(max_cells, n_cells)
        adata.file.close()
        lines.append(f"{stem}\t{n_bins}\t{n_cells}\t{h5ad}")
    manifest.write_text("\n".join(lines) + "\n")
    print(f"[prepare] wrote manifest {manifest} ({len(stems)} new + {kept} kept)",
          flush=True)


def main() -> int:
    args = parse_args()
    args.input_root.mkdir(parents=True, exist_ok=True)
    stems = args.datasets or discover_datasets(args.data_dir)
    if args.single_dataset:
        stems = [s for s in stems if s == args.single_dataset]
        if not stems:
            raise ValueError(f"--single-dataset {args.single_dataset} not found")
    if not stems:
        raise ValueError(f"No *_scdiff2.h5ad found under {args.data_dir}")
    workers = args.workers
    if workers <= 0:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"[prepare] datasets: {stems} workers={workers}", flush=True)
    # Manifest written once by the first job; guard with a lock-free check.
    if not args.single_dataset or not (args.input_root / "manifest.tsv").exists():
        write_manifest(stems, args.input_root, args.chrom,
                       args.n_bins, args.max_cells, args.data_dir)
    for stem in stems:
        prepare_dataset(stem, args.data_dir, args.input_root, args.chrom,
                        args.n_bins, args.layer, args.max_cells, args.overwrite,
                        workers)
    print("[prepare] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())