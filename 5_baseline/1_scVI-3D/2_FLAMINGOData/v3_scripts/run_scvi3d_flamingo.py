#!/usr/bin/env python3
"""scVI-3D imputation for FLAMINGO v3 paramsweep data.

This is a self-contained re-implementation of the scVI-3D algorithm
(Zheng et al., 2022) specialised for the FLAMINGO single-chromosome
simulation datasets.  It reads the per-cell txt files produced by
``prepare_scvi3d_input.py``, builds band matrices, trains a scVI VAE per
band, imputes the denoised contact matrix, and writes:

  * ``<output_dir>/scVI-3D_norm/cell_*.txt``  — per-cell imputed contacts
    (same 5-column format as input: chrA binA chrB binB count)
  * ``<output_dir>/<stem>_scVI3D_imputed.npz``  — (n_cells, n_features)
    sparse matrix in the FLAMINGO upper-triangle feature ordering, ready
    for direct comparison with ``layers['gt']`` of the h5ad.

Algorithm overview (matching the original scVI-3D.py):

1. Read per-cell contact txts → DataFrame per cell
2. Rescale: binA = bp // resolution, diag = |binB - binA|
3. Build band matrices: for each band distance d, stack cells into
   a (n_cells, n_bins - d) matrix where column k = contact (k, k+d)
4. For each band matrix: train scvi.model.SCVI, then
   ``get_normalized_expression(library_size=bandDepth)`` to impute
5. Convert imputed values back to contact coordinates and write out

Usage::

    python run_scvi3d_flamingo.py \\
        --input-dir  <dir with cell_*.txt, genome.txt, cell_summary.txt> \\
        --output-dir <work dir> \\
        --stem v3_hybrid_W0p7_500cells_level0_r0p01 \\
        --resolution 1000000 \\
        --band-max 10 --n-latent 100 --gpu
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scipy.sparse import coo_matrix, save_npz

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class BandMatrixBuilder:
    """Build per-band (cell × locus) matrices from per-cell contact DataFrames.

    Reproduces ``Process.rescale`` + ``band_all`` from the original scVI-3D.py.
    """

    def __init__(self, resolution: int, chrom_size_bp: int):
        self.resolution = resolution
        self.chrom = None
        self.n_bins = chrom_size_bp // resolution + 1

    def rescale(self, chrA, binA_bp, binB_bp, counts):
        """Convert bp coordinates to bin indices and compute diagonal distance."""
        binA = binA_bp // self.resolution
        binB = binB_bp // self.resolution
        diag = np.abs(binB - binA)
        self.df = pd.DataFrame(
            {"chrA": chrA, "x": binA, "y": binB, "counts": counts, "diag": diag}
        )

    def band_all(self, used_diags) -> dict:
        """Group contacts by diagonal distance and build band vectors.

        Returns ``{diag: np.ndarray(n_bins - diag)}`` for a single cell.
        """
        cell_band = {}
        for diag, sub in self.df.groupby("diag"):
            if used_diags != "whole" and diag not in used_diags:
                continue
            band_len = self.n_bins - diag
            band_vec = np.zeros(band_len, dtype=np.float64)
            # x = min(binA, binB) -> position along the band
            x = np.minimum(sub["x"].values, sub["y"].values)
            valid = x < band_len
            np.add.at(band_vec, x[valid], sub["counts"].values[valid])
            cell_band[diag] = band_vec
        return cell_band


# ---------------------------------------------------------------------------
# Core scVI-3D normalization (per band)
# ---------------------------------------------------------------------------
def normalize_band(
    band_matrix: np.ndarray,
    chrom: str,
    band_dist: int,
    n_latent: int = 100,
    max_epochs: int | None = None,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Train scVI on one band matrix and return (latent, imputed) arrays.

    ``band_matrix`` has shape (n_cells, n_loci).  The function:
    1. Creates an AnnData object
    2. Sets up and trains scVI
    3. Imputes with ``get_normalized_expression(library_size=bandDepth)``
    4. Returns the latent representation and the imputed matrix

    ``max_epochs=None`` uses the scVI default (400).  Pass a smaller int for
    faster testing.
    """
    n_cells, n_loci = band_matrix.shape
    print(f"  band {band_dist}: shape={band_matrix.shape}", flush=True)

    if n_loci == 0:
        return np.zeros((n_cells, n_latent)), np.empty((n_cells, 0))

    # bandDepth = mean total counts per cell (library size reference)
    band_depth = band_matrix.sum(axis=1).mean()

    adata = sc.AnnData(band_matrix)
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata, n_latent=n_latent)
    train_kwargs = {"use_gpu": use_gpu}
    if max_epochs is not None:
        train_kwargs["max_epochs"] = max_epochs
    model.train(**train_kwargs)

    # Impute with fixed library size = bandDepth
    imputed = model.get_normalized_expression(
        library_size=band_depth, return_numpy=True
    )
    latent = model.get_latent_representation()

    return latent, imputed


# ---------------------------------------------------------------------------
# Contact list I/O
# ---------------------------------------------------------------------------
def read_contact_txt(path: Path) -> pd.DataFrame:
    """Read a 5-column tab-separated contact file.

    Columns: chrA  binA(bp)  chrB  binB(bp)  counts
    """
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["chrA", "binA", "chrB", "binB", "counts"],
    )
    return df


def write_contact_txt(
    path: Path, contacts: list, chrom: str, resolution: int
):
    """Write imputed contacts to a per-cell txt file (appending per band)."""
    if not contacts:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        return
    rows = pd.concat(contacts, ignore_index=True)
    # Convert bin index back to bp coordinates
    rows["binA"] = rows["binA"].astype(int) * resolution
    rows["binB"] = rows["binB"].astype(int) * resolution
    rows[["chrA", "binA", "chrB", "binB", "count"]].to_csv(
        path, sep="\t", header=False, index=False, mode="a"
    )


def imputed_to_contacts(
    imputed: np.ndarray, chrom: str, band_dist: int
) -> pd.DataFrame:
    """Convert (n_cells, n_loci) imputed matrix to a long contact DataFrame.

    Each column k in the band matrix corresponds to contact (k, k + band_dist).
    """
    n_cells, n_loci = imputed.shape
    if n_loci == 0:
        return pd.DataFrame(columns=["chrA", "binA", "chrB", "binB", "count", "cellID"])
    # Melt: cellID, binA(=column index), count
    tmp = pd.DataFrame(imputed.T)  # (n_loci, n_cells)
    tmp.index.name = "binA"
    long = pd.melt(
        tmp.reset_index(), id_vars=["binA"],
        var_name="cellID", value_name="count",
    )
    long["binA"] = long["binA"].astype(int)
    long["binB"] = long["binA"] + band_dist
    long["chrA"] = chrom
    long["chrB"] = chrom
    return long[["chrA", "binA", "chrB", "binB", "count", "cellID"]]


# ---------------------------------------------------------------------------
# Collect: per-cell txt -> (n_cells, n_features) sparse npz
# ---------------------------------------------------------------------------
def collect_imputed(
    impute_dir: Path,
    output_npz: Path,
    n_bins: int,
    resolution: int,
    chrom: str,
):
    """Read per-cell imputed txts and build a (n_cells, n_features) CSR matrix.

    The feature ordering uses canonical upper-triangle indices:
    ``np.triu_indices(n_bins, k=1)`` in NumPy row-major order.
    """
    files = sorted(
        impute_dir.glob("cell_*.txt"),
        key=lambda p: int(re.match(r"cell_(\d+)", p.stem).group(1)),
    )
    n_cells = len(files)
    i_idx, j_idx = np.triu_indices(n_bins, k=1)
    n_features = len(i_idx)
    print(f"[scvi-collect] {n_cells} cells, {n_features} features", flush=True)

    rows = np.zeros((n_cells, n_features), dtype=np.float64)
    for ci, fpath in enumerate(files):
        if fpath.stat().st_size == 0:
            continue
        df = pd.read_csv(
            fpath, sep="\t", header=None,
            names=["chrA", "binA", "chrB", "binB", "count"],
        )
        a = (df["binA"].to_numpy() // resolution).astype(np.int64)
        b = (df["binB"].to_numpy() // resolution).astype(np.int64)
        c = df["count"].to_numpy(dtype=np.float64)
        valid = (a >= 0) & (a < n_bins) & (b >= 0) & (b < n_bins)
        a, b, c = a[valid], b[valid], c[valid]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        mat = np.zeros((n_bins, n_bins), dtype=np.float64)
        np.add.at(mat, (lo, hi), c)
        mat = mat + mat.T
        rows[ci] = mat[i_idx, j_idx]

    rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
    rows[rows < 0] = 0.0
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_npz, coo_matrix(rows).tocsr())
    print(f"[scvi-collect] saved {output_npz} shape={rows.shape}", flush=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory with cell_*.txt, genome.txt, cell_summary.txt")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output working directory")
    parser.add_argument("--collect-dir", type=Path, default=None,
                        help="Directory for the final npz (default: output-dir/..)")
    parser.add_argument("--stem", required=True,
                        help="Dataset stem (e.g. v3_hybrid_W0p7_500cells_level0_r0p01)")
    parser.add_argument("--resolution", type=int, default=1_000_000,
                        help="Resolution in bp (default: 1Mb)")
    parser.add_argument("--n-bins", type=int, default=500,
                        help="Number of bins along the chromosome (default: 500)")
    parser.add_argument("--band-max", default="whole",
                        help='Max band distance, or "whole" (default: whole)')
    parser.add_argument("--include-diag", action="store_true",
                        help="Include diagonal (band 0). Default: exclude.")
    parser.add_argument("--n-latent", type=int, default=100,
                        help="Latent dimension (default: 100)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Max training epochs per band (default: scVI=400)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for scVI training")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolution = args.resolution
    n_bins = args.n_bins
    chrom_size_bp = n_bins * resolution
    stem = args.stem

    # Chromosome name: read from genome.txt
    genome_path = args.input_dir / "genome.txt"
    if not genome_path.exists():
        print(f"ERROR: {genome_path} not found", file=sys.stderr)
        return 1
    genome_df = pd.read_csv(genome_path, sep="\t", header=None)
    chrom = genome_df.iloc[0, 0]
    chrom_size_bp = int(genome_df.iloc[0, 1])
    n_bins = chrom_size_bp // resolution + 1
    print(f"[scvi3d] chrom={chrom} size={chrom_size_bp} n_bins={n_bins}", flush=True)

    # Cell list (sorted by filename = cell_1, cell_2, ..., cell_N).
    # Only match files whose stem is cell_<digits>; exclude cell_summary etc.
    def _cell_sort_key(p: Path) -> int:
        m = re.match(r"cell_(\d+)$", p.stem)
        return int(m.group(1)) if m else -1

    cell_files = sorted(
        (p for p in args.input_dir.glob("cell_*.txt") if _cell_sort_key(p) >= 0),
        key=_cell_sort_key,
    )
    n_cells = len(cell_files)
    print(f"[scvi3d] {n_cells} cells from {args.input_dir}", flush=True)

    # Band range
    if args.band_max == "whole":
        used_diags = "whole"
        band_start = 1
        band_end = n_bins
    else:
        band_max = int(args.band_max)
        if args.include_diag:
            used_diags = list(range(0, band_max + 1))
            band_start = 0
        else:
            used_diags = list(range(1, band_max + 1))
            band_start = 1
        band_end = band_max + 1
    print(f"[scvi3d] bands: {band_start}..{band_end-1} (used_diags={used_diags})", flush=True)

    # ------------------------------------------------------------------
    # Step 1: Read all per-cell contact txts and build band matrices
    # ------------------------------------------------------------------
    print("[scvi3d] Step 1: reading contacts and building band matrices", flush=True)
    builder = BandMatrixBuilder(resolution, chrom_size_bp)

    # raw_cells[ci] = {diag: band_vec}
    raw_cells = []
    for ci, fpath in enumerate(cell_files):
        if fpath.stat().st_size == 0:
            raw_cells.append({})
            continue
        df = read_contact_txt(fpath)
        builder.rescale(df["chrA"], df["binA"], df["binB"], df["counts"])
        cell_band = builder.band_all(used_diags)
        raw_cells.append(cell_band)
        if (ci + 1) % 200 == 0:
            print(f"  read {ci + 1}/{n_cells} cells", flush=True)

    # ------------------------------------------------------------------
    # Step 2: Stack into (n_cells, n_loci) per band
    # ------------------------------------------------------------------
    print("[scvi3d] Step 2: stacking band matrices", flush=True)
    band_chrom_diag = {}  # {band_dist: (n_cells, n_loci) array}
    for band in range(band_start, band_end):
        if used_diags != "whole" and band not in used_diags:
            continue
        band_len = n_bins - band
        mat = np.zeros((n_cells, band_len), dtype=np.float64)
        for ci in range(n_cells):
            if band in raw_cells[ci]:
                mat[ci] = raw_cells[ci][band]
        band_chrom_diag[band] = mat
    del raw_cells

    # ------------------------------------------------------------------
    # Step 3: Train scVI per band and impute
    # ------------------------------------------------------------------
    print("[scvi3d] Step 3: scVI training and imputation per band", flush=True)
    norm_dir = args.output_dir / "scVI-3D_norm"
    if args.overwrite and norm_dir.exists():
        import shutil
        shutil.rmtree(norm_dir)
    norm_dir.mkdir(parents=True, exist_ok=True)

    latent_list = []
    for band_dist, band_mat in band_chrom_diag.items():
        latent, imputed = normalize_band(
            band_mat, chrom, band_dist,
            n_latent=args.n_latent, max_epochs=args.max_epochs,
            use_gpu=args.gpu,
        )
        latent_list.append(latent)

        # Convert imputed matrix -> per-cell contacts
        contacts_long = imputed_to_contacts(imputed, chrom, band_dist)
        for cell_id, cell_df in contacts_long.groupby("cellID"):
            cell_name = cell_files[int(cell_id)].name
            out_path = norm_dir / cell_name
            if band_dist == band_start:
                # First band: start fresh (remove existing file if overwrite)
                if out_path.exists():
                    out_path.unlink()
            sub = cell_df[["chrA", "binA", "chrB", "binB", "count"]].copy()
            sub["binA"] = sub["binA"].astype(int) * resolution
            sub["binB"] = sub["binB"].astype(int) * resolution
            sub.to_csv(out_path, sep="\t", header=False, index=False, mode="a")

    # ------------------------------------------------------------------
    # Step 4: Save latent embeddings
    # ------------------------------------------------------------------
    print("[scvi3d] Step 4: saving latent embeddings", flush=True)
    latent_full = np.hstack(latent_list) if latent_list else np.zeros((n_cells, 0))
    latent_dir = args.output_dir / "latentEmbeddings"
    latent_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(latent_dir / "scVI-3D_latentFull.txt",
               latent_full, delimiter="\t")

    # ------------------------------------------------------------------
    # Step 5: Collect per-cell txts into feature npz
    # ------------------------------------------------------------------
    print("[scvi3d] Step 5: collecting into feature npz", flush=True)
    collect_dir = args.collect_dir or args.output_dir.parent
    output_npz = collect_dir / f"{stem}_scVI3D_imputed.npz"
    collect_imputed(norm_dir, output_npz, n_bins, resolution, chrom)

    print(f"[scvi3d] DONE: {stem}", flush=True)
    print(f"  per-cell txt:  {norm_dir}/", flush=True)
    print(f"  feature npz:   {output_npz}", flush=True)
    print(f"  latent embed:  {latent_dir}/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
