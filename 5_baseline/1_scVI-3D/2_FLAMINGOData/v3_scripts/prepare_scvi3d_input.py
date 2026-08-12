#!/usr/bin/env python3
"""Prepare FLAMINGO v3 raw simulation data for scVI-3D input.

Reads the raw 500x500 dense contact matrices from::

    <raw_root>/sim_<stem>/downsampled_contact_data/type_X_cell_Y.txt

Each file is a dense symmetric matrix (500 rows x 500 columns, tab-separated,
float IF values).  This script extracts the upper-triangle non-zero contacts,
applies ``ceil()`` to convert float IF values to integer counts (required by
scVI's Zero-Inflated Negative Binomial likelihood), and writes the per-cell
tab-separated files expected by scVI-3D::

    chrA  binA(bp)  chrB  binB(bp)  counts

For each dataset this script produces::

    <input_root>/<stem>/cell_1.txt ... cell_N.txt   (per-cell contact lists)
    <input_root>/<stem>/genome.txt                   (chrom \t size_bp)
    <input_root>/<stem>/cell_summary.txt             (name \t batch \t cell_type)

Usage::

    python prepare_scvi3d_input.py \\
        --raw-root  <dir with sim_v3_hybrid_*> \\
        --input-root <output dir> \\
        --datasets v3_hybrid_W0p7_500cells_level0_r0p01 \\
        --workers 20 --overwrite
"""

from __future__ import annotations

import argparse
import os
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Constants — FLAMINGO v3 paramsweep datasets
# ---------------------------------------------------------------------------
N_BINS = 500
CHROM_NAME = "chrFLAMINGO"
RESOLUTION = 1_000_000

DEFAULT_RAW_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/"
    "1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/1_RawData/"
    "3_fixed_flamnigoGen"
)
DEFAULT_INPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/"
    "5_baseline/1_scVI-3D/2_FLAMINGOData/v3_inputData"
)

# Precomputed upper-triangle indices (i < j) for the 500x500 contact matrix
_IU, _JU = np.triu_indices(N_BINS, k=1)


# ---------------------------------------------------------------------------
# Cell name parsing: "type_1_cell_42.txt" -> ("T1", 42)
# ---------------------------------------------------------------------------
_TYPE_MAP = {1: "T1", 2: "T2", 3: "T3"}
_NAME_RE = re.compile(r"type_(\d+)_cell_(\d+)\.txt$")


def parse_cell_name(filename: str) -> tuple[str, int]:
    """Parse ``type_X_cell_Y.txt`` -> (cell_type, cell_number)."""
    m = _NAME_RE.match(filename)
    if not m:
        raise ValueError(f"Cannot parse cell name: {filename}")
    type_id = int(m.group(1))
    cell_num = int(m.group(2))
    return _TYPE_MAP.get(type_id, f"T{type_id}"), cell_num


def _sort_key(path: Path) -> tuple[int, int]:
    """Sort key: (type_id, cell_num) so order is type_1/1..500, type_2/1..500, ..."""
    m = _NAME_RE.match(path.name)
    return (int(m.group(1)), int(m.group(2)))


# ---------------------------------------------------------------------------
# Per-cell writer (top-level for Pool pickling)
# ---------------------------------------------------------------------------
def write_cell(args_tuple) -> tuple[int, int]:
    """Read a raw 500x500 matrix, extract upper-tri contacts, ceil, write txt."""
    cell_idx, raw_path, dest_dir, resolution, overwrite = args_tuple
    cell_name = f"cell_{cell_idx + 1}.txt"
    dest = dest_dir / cell_name
    if dest.exists() and not overwrite:
        return cell_idx, -1  # skipped

    # Load dense 500x500 matrix and extract upper-triangle non-zero contacts
    mat = np.loadtxt(raw_path)
    vals = mat[_IU, _JU]
    mask = vals != 0
    if mask.sum() == 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("")
        return cell_idx, 0

    i_nz = _IU[mask]
    j_nz = _JU[mask]
    v_nz = np.ceil(vals[mask]).astype(int)  # ceil to integer for scVI ZINB

    # Convert bin indices to bp coordinates
    b1 = i_nz * resolution
    b2 = j_nz * resolution

    rows = np.column_stack([
        np.full(mask.sum(), CHROM_NAME, dtype=object),
        b1,
        np.full(mask.sum(), CHROM_NAME, dtype=object),
        b2,
        v_nz,
    ])
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(dest, rows, fmt=["%s", "%d", "%s", "%d", "%d"], delimiter="\t")
    return cell_idx, int(mask.sum())


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def discover_datasets(raw_root: Path) -> list[str]:
    """Find all ``sim_v3_hybrid_*`` directories and return the stem names."""
    stems = []
    for d in sorted(raw_root.iterdir()):
        if d.is_dir() and d.name.startswith("sim_v3_hybrid_"):
            contact_dir = d / "downsampled_contact_data"
            if contact_dir.is_dir():
                stems.append(d.name[len("sim_"):])  # strip "sim_" prefix
    return stems


def prepare_dataset(
    stem: str,
    raw_root: Path,
    input_root: Path,
    resolution: int,
    workers: int,
    overwrite: bool,
) -> int:
    """Convert one raw simulation directory into scVI-3D input files."""
    raw_dir = raw_root / f"sim_{stem}" / "downsampled_contact_data"
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)

    # Collect and sort cell files by (type_id, cell_num) for deterministic order
    raw_files = sorted(
        raw_dir.glob("type_*_cell_*.txt"),
        key=_sort_key,
    )
    n_cells = len(raw_files)
    print(f"[scvi-prep] {stem}: {n_cells} cells from {raw_dir}", flush=True)

    # Derive cell types from filenames
    cell_types = [parse_cell_name(p.name)[0] for p in raw_files]

    dest_dir = input_root / stem
    dest_dir.mkdir(parents=True, exist_ok=True)

    # genome.txt: chrom \t size_bp
    # size_bp = (N_BINS - 1) * resolution so that n_bins = size_bp // res + 1 = N_BINS
    (dest_dir / "genome.txt").write_text(
        f"{CHROM_NAME}\t{(N_BINS - 1) * resolution}\n"
    )

    # cell_summary.txt: name \t batch \t cell_type
    with (dest_dir / "cell_summary.txt").open("w") as f:
        f.write("name\tbatch\tcell_type\n")
        for c in range(n_cells):
            f.write(f"cell_{c + 1}.txt\tbatch1\t{cell_types[c]}\n")

    # Per-cell txt files
    items = [
        (c, raw_files[c], dest_dir, resolution, overwrite)
        for c in range(n_cells)
    ]
    if workers <= 1:
        results = [write_cell(it) for it in items]
    else:
        with Pool(processes=workers) as p:
            results = p.map(
                write_cell, items,
                chunksize=max(1, n_cells // (workers * 4)),
            )

    n_written = sum(1 for _, n in results if n >= 0)
    n_skipped = sum(1 for _, n in results if n == -1)
    nnz = sum(n for _, n in results if n > 0)
    print(
        f"[scvi-prep] {stem}: {n_cells} cells, "
        f"{n_written} written, {n_skipped} skipped, total nnz={nnz}",
        flush=True,
    )
    return n_cells


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT,
                        help="Directory with sim_v3_hybrid_* subdirectories")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT,
                        help="Output root for per-dataset input directories")
    parser.add_argument("--resolution", type=int, default=RESOLUTION,
                        help="Resolution in bp (default: 1Mb)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Dataset stems to process (default: all)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = SLURM_CPUS_PER_TASK or 1)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing cell txt files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input_root.mkdir(parents=True, exist_ok=True)
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    stems = args.datasets or discover_datasets(args.raw_root)
    print(f"[scvi-prep] datasets={stems} workers={workers}", flush=True)
    for stem in stems:
        prepare_dataset(
            stem, args.raw_root, args.input_root,
            args.resolution, workers, args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())