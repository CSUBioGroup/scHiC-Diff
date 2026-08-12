#!/usr/bin/env python3
"""Prepare Lee SuperTAD pipeline BEDPE data for scVI-3D input.

Input:  per_cell_bedpe/<CellType>_cell_<NNNN>.txt
        5-col BEDPE:  chrA  startA  chrB  startB  contact

Output per cell type:
  cell_1.txt ... cell_N.txt  - 5-col: chrA  binA_bp  chrB  binB_bp  count
  genome.txt                  - chr4  (n_bins-1)*resolution
  cell_summary.txt            - name  batch  cell_type

Data geometry (from metadata.json):
  chromosome      = chr4
  resolution      = 10000
  n_bins          = 49
  region_start    = 54890000

The count column is rounded up to integers (ceil) because scVI uses ZINB
likelihood which requires non-negative integer counts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np


DEFAULT_INPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "1_scVI-3D/5_lee_SuperTAD_pileline/input_lee"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "1_scVI-3D/5_lee_SuperTAD_pileline/scvi3d_input"
)
DEFAULT_METADATA = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "1_scVI-3D/5_lee_SuperTAD_pileline/input_lee/metadata.json"
)

CELL_TYPES = ["Astro", "Endo", "ODC", "OPC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Lee SuperTAD BEDPE for scVI-3D."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--cell-types", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=0,
                        help="0 = SLURM_CPUS_PER_TASK or 1")
    return parser.parse_args()


_W_CHROM: str = "chr4"
_W_RES: int = 10000
_W_REGION_START: int = 54890000
_W_BEDPE_DIR: Path = Path()
_W_OUT_DIR: Path = Path()
_W_OVERWRITE: bool = False


def _worker_init(kwargs: dict) -> None:
    global _W_CHROM, _W_RES, _W_REGION_START, _W_BEDPE_DIR, _W_OUT_DIR, _W_OVERWRITE
    _W_CHROM = kwargs["chrom"]
    _W_RES = kwargs["resolution"]
    _W_REGION_START = kwargs["region_start"]
    _W_BEDPE_DIR = kwargs["bedpe_dir"]
    _W_OUT_DIR = kwargs["out_dir"]
    _W_OVERWRITE = kwargs["overwrite"]


def _convert_cell(args: tuple[str, int]) -> tuple[str, int, int]:
    cell_type, cell_id = args
    bedpe_idx = cell_id - 1  # 0-based BEDPE -> 1-based scVI-3D
    bedpe_name = f"{cell_type}_cell_{bedpe_idx:04d}.txt"
    bedpe_path = _W_BEDPE_DIR / bedpe_name
    out_path = _W_OUT_DIR / f"cell_{cell_id}.txt"

    if out_path.exists() and not _W_OVERWRITE:
        return cell_type, cell_id, -1

    if not bedpe_path.exists():
        return cell_type, cell_id, -2

    if bedpe_path.stat().st_size == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        return cell_type, cell_id, 0

    data = np.loadtxt(bedpe_path, dtype=str)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        return cell_type, cell_id, 0

    starts_a = data[:, 1].astype(np.int64)
    starts_b = data[:, 3].astype(np.int64)
    counts = data[:, 4].astype(np.float64)

    # Convert genomic coordinates to bin indices, then to bp for scVI-3D
    bin_a = (starts_a - _W_REGION_START) // _W_RES
    bin_b = (starts_b - _W_REGION_START) // _W_RES
    bin_a_bp = bin_a * _W_RES
    bin_b_bp = bin_b * _W_RES

    # ZINB requires non-negative integer counts -> ceil
    counts_int = np.ceil(counts).astype(np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack([
        np.full(len(counts_int), _W_CHROM, dtype=object),
        bin_a_bp,
        np.full(len(counts_int), _W_CHROM, dtype=object),
        bin_b_bp,
        counts_int,
    ])
    np.savetxt(out_path, rows, fmt=["%s", "%d", "%s", "%d", "%d"], delimiter="\t")
    return cell_type, cell_id, len(counts_int)


def prepare_cell_type(cell_type: str, input_root: Path, output_root: Path,
                      chrom: str, n_bins: int, resolution: int,
                      region_start: int, workers: int, overwrite: bool) -> None:
    bedpe_dir = input_root / "per_cell_bedpe"
    out_dir = output_root / cell_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # genome.txt: (n_bins - 1) * resolution
    chrom_size_bp = (n_bins - 1) * resolution
    (out_dir / "genome.txt").write_text(f"{chrom}\t{chrom_size_bp}\n")

    # Discover cells
    pattern = re.compile(rf"^{cell_type}_cell_(\d+)\.txt$")
    cell_ids = []
    for p in sorted(bedpe_dir.iterdir()):
        m = pattern.match(p.name)
        if m:
            cell_ids.append(int(m.group(1)) + 1)  # 1-based
    cell_ids.sort()
    if not cell_ids:
        raise FileNotFoundError(f"No BEDPE files for {cell_type}")

    n_cells = len(cell_ids)
    print(f"[scvi-prep] {cell_type}: {n_cells} cells, chrom={chrom}, "
          f"n_bins={n_bins}, res={resolution}", flush=True)

    # cell_summary.txt
    with (out_dir / "cell_summary.txt").open("w") as f:
        f.write("name\tbatch\tcell_type\n")
        for cid in cell_ids:
            f.write(f"cell_{cid}.txt\tbatch1\t{cell_type}\n")

    # Convert cells
    init_kwargs = {
        "chrom": chrom,
        "resolution": resolution,
        "region_start": region_start,
        "bedpe_dir": bedpe_dir,
        "out_dir": out_dir,
        "overwrite": overwrite,
    }

    items = [(cell_type, cid) for cid in cell_ids]
    total_nnz = 0
    skipped = 0

    if workers <= 1:
        _worker_init(init_kwargs)
        for item in items:
            _, _, nnz = _convert_cell(item)
            if nnz >= 0:
                total_nnz += nnz
            elif nnz == -1:
                skipped += 1
    else:
        with Pool(processes=workers, initializer=_worker_init,
                  initargs=(init_kwargs,)) as pool:
            done = 0
            for ct, cid, nnz in pool.imap_unordered(_convert_cell, items,
                                                    chunksize=max(1, len(items) // (workers * 4))):
                if nnz >= 0:
                    total_nnz += nnz
                elif nnz == -1:
                    skipped += 1
                done += 1
                if done % 200 == 0:
                    print(f"[scvi-prep] {cell_type}: {done}/{n_cells}", flush=True)

    print(f"[scvi-prep] {cell_type}: done, nnz={total_nnz}, skipped={skipped}",
          flush=True)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    with open(args.metadata) as f:
        meta = json.load(f)
    chrom = meta["chromosome"]
    resolution = meta["resolution"]
    n_bins = meta["n_bins"]
    region_start = meta["region_start"]

    cell_types = args.cell_types or list(meta.get("cell_types", {}).keys())
    if not cell_types:
        cell_types = CELL_TYPES

    workers = args.workers
    if workers <= 0:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    print(f"[scvi-prep] chrom={chrom}, res={resolution}, n_bins={n_bins}, "
          f"cell_types={cell_types}, workers={workers}", flush=True)

    for ct in cell_types:
        prepare_cell_type(ct, args.input_root, args.output_root,
                          chrom, n_bins, resolution, region_start,
                          workers, args.overwrite)

    print("[scvi-prep] all done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    raise SystemExit(main())