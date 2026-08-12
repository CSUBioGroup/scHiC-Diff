#!/usr/bin/env python3
"""Prepare Lee SuperTAD pipeline BEDPE data for scHiCluster impute-cell input.

Input:  per_cell_bedpe/<CellType>_cell_<NNNN>.txt
        5-col BEDPE:  chrA  startA  chrB  startB  contact

Output: schicluster_input/<CellType>/cell_<id>_chr4.txt
        3-col triplet:  row_bin  col_bin  contact  (0-based, row <= col)

        schicluster_input/<CellType>/simu_chr4.chrom.sizes
        chr4\t<n_bins-1>   (scHiCluster internally creates n_bins bins)

Data geometry (from metadata.json):
  chromosome      = chr4
  resolution      = 10000
  n_bins          = 49
  region_start    = 54890000
  region_end      = 55380000

Cell types and counts:
  Astro: 449   Endo: 202   ODC: 1244   OPC: 203   total: 2098
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
    "4_scHiCluster/5_lee_SuperTAD_pileline/input_lee"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "4_scHiCluster/5_lee_SuperTAD_pileline/schicluster_input"
)
DEFAULT_METADATA = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "4_scHiCluster/5_lee_SuperTAD_pileline/input_lee/metadata.json"
)

CELL_TYPES = ["Astro", "Endo", "ODC", "OPC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Lee SuperTAD BEDPE to scHiCluster triplet input."
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
_W_N_BINS: int = 49
_W_BEDPE_DIR: Path = Path()
_W_OUT_DIR: Path = Path()
_W_OVERWRITE: bool = False


def _worker_init(kwargs: dict) -> None:
    global _W_CHROM, _W_RES, _W_REGION_START, _W_N_BINS
    global _W_BEDPE_DIR, _W_OUT_DIR, _W_OVERWRITE
    _W_CHROM = kwargs["chrom"]
    _W_RES = kwargs["resolution"]
    _W_REGION_START = kwargs["region_start"]
    _W_N_BINS = kwargs["n_bins"]
    _W_BEDPE_DIR = kwargs["bedpe_dir"]
    _W_OUT_DIR = kwargs["out_dir"]
    _W_OVERWRITE = kwargs["overwrite"]


def _convert_cell(args: tuple[str, int]) -> tuple[str, int, int]:
    cell_type, cell_id = args
    # cell_id is 1-based (scHiCluster convention); BEDPE uses 0-based
    bedpe_idx = cell_id - 1
    bedpe_name = f"{cell_type}_cell_{bedpe_idx:04d}.txt"
    bedpe_path = _W_BEDPE_DIR / bedpe_name
    out_path = _W_OUT_DIR / f"cell_{cell_id}_{_W_CHROM}.txt"

    if out_path.exists() and not _W_OVERWRITE:
        return cell_type, cell_id, -1

    if not bedpe_path.exists():
        return cell_type, cell_id, -2

    # Read BEDPE: chrA  startA  chrB  startB  contact
    data = np.loadtxt(bedpe_path, dtype=str)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        return cell_type, cell_id, 0

    starts_a = data[:, 1].astype(np.int64)
    starts_b = data[:, 3].astype(np.int64)
    contacts = data[:, 4].astype(np.float64)

    # Convert genomic coordinates to 0-based bin indices
    bin_a = (starts_a - _W_REGION_START) // _W_RES
    bin_b = (starts_b - _W_REGION_START) // _W_RES

    # Ensure row <= col (upper triangle)
    swapped = bin_a > bin_b
    bin_a[swapped], bin_b[swapped] = bin_b[swapped], bin_a[swapped]

    # Filter valid bins
    valid = (bin_a >= 0) & (bin_b < _W_N_BINS) & (bin_a < _W_N_BINS) & (bin_b >= 0)
    bin_a = bin_a[valid]
    bin_b = bin_b[valid]
    contacts = contacts[valid]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(contacts) == 0:
        out_path.write_text("")
        return cell_type, cell_id, 0

    arr = np.column_stack((bin_a, bin_b, contacts))
    np.savetxt(out_path, arr, fmt=["%d", "%d", "%.10g"], delimiter="\t")
    return cell_type, cell_id, len(contacts)


def prepare_cell_type(cell_type: str, args: argparse.Namespace,
                      chrom: str, resolution: int, region_start: int,
                      n_bins: int) -> None:
    bedpe_dir = args.input_root / "per_cell_bedpe"
    out_dir = args.output_root / cell_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write chrom.sizes: scHiCluster interprets value as max bp coordinate.
    # ngene = value // res + 1.  For n_bins bins at given resolution:
    #   value = (n_bins - 1) * resolution  ->  ngene = (n_bins-1)*res // res + 1 = n_bins
    chrom_file = out_dir / f"simu_{chrom}.chrom.sizes"
    chrom_size_bp = (n_bins - 1) * resolution
    chrom_file.write_text(f"{chrom}\t{chrom_size_bp}\n")

    # Discover cells (BEDPE uses 0-based IDs, convert to 1-based for scHiCluster)
    pattern = re.compile(rf"^{cell_type}_cell_(\d+)\.txt$")
    cell_ids = []
    for p in sorted(bedpe_dir.iterdir()):
        m = pattern.match(p.name)
        if m:
            cell_ids.append(int(m.group(1)))
    cell_ids.sort()
    # Convert 0-based BEDPE IDs to 1-based scHiCluster IDs
    cell_ids = [cid + 1 for cid in cell_ids]

    if not cell_ids:
        print(f"[prepare] {cell_type}: no BEDPE files found", flush=True)
        return

    print(f"[prepare] {cell_type}: {len(cell_ids)} cells, "
          f"chrom={chrom}, n_bins={n_bins}, res={resolution}", flush=True)

    workers = args.workers
    if workers <= 0:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    init_kwargs = {
        "chrom": chrom,
        "resolution": resolution,
        "region_start": region_start,
        "n_bins": n_bins,
        "bedpe_dir": bedpe_dir,
        "out_dir": out_dir,
        "overwrite": args.overwrite,
    }

    items = [(cell_type, cid) for cid in cell_ids]
    total_nnz = 0
    skipped = 0
    missing = 0

    if workers <= 1:
        _worker_init(init_kwargs)
        for item in items:
            _, _, nnz = _convert_cell(item)
            if nnz >= 0:
                total_nnz += nnz
            elif nnz == -1:
                skipped += 1
            elif nnz == -2:
                missing += 1
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
                elif nnz == -2:
                    missing += 1
                done += 1
                if done % 200 == 0:
                    print(f"[prepare] {cell_type}: {done}/{len(items)}", flush=True)

    print(f"[prepare] {cell_type}: done, nnz={total_nnz}, "
          f"skipped={skipped}, missing={missing}", flush=True)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    # Read metadata
    with open(args.metadata) as f:
        meta = json.load(f)
    chrom = meta["chromosome"]
    resolution = meta["resolution"]
    n_bins = meta["n_bins"]
    region_start = meta["region_start"]

    cell_types = args.cell_types or list(meta.get("cell_types", {}).keys())
    if not cell_types:
        cell_types = CELL_TYPES

    print(f"[prepare] chrom={chrom}, res={resolution}, n_bins={n_bins}, "
          f"region_start={region_start}, cell_types={cell_types}", flush=True)

    for ct in cell_types:
        prepare_cell_type(ct, args, chrom, resolution, region_start, n_bins)

    print("[prepare] all done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())