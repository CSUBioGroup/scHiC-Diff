#!/usr/bin/env python3
"""Collect Higashi impute.hdf5 outputs into per-dataset lower-triangle NPZ.

Reads ``<temp_dir>/<chrom>_<embedding_name>_nbr_<K>_impute.hdf5`` and converts
each cell's imputed values into a ``(n_cells, n_features)`` feature matrix
(upper-triangle i<j, row-major) matching the FLAMINGO h5ad ``layers['gt']``
ordering.  Saves as scipy sparse CSR NPZ for evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import coo_matrix, save_npz

from v3_common import (CHROM_NAME, DEFAULT_DATA_DIR, DATASETS, N_BINS,
                       N_FEATURES, feature_to_bins)

DEFAULT_INPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "6_Higashi/2_FLAMINGOData/v3_inputData")
DEFAULT_OUTPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "6_Higashi/2_FLAMINGOData/v3_outputData")
EMBEDDING_NAME = "flamingo_higashi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--neighbor", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def hdf5_to_lower_tri(hdf5_path: Path, n_cells: int, n_beads: int,
                      i_idx, j_idx) -> np.ndarray:
    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)
    out = np.zeros((n_cells, n_beads * (n_beads - 1) // 2), dtype=np.float32)
    with h5py.File(hdf5_path, "r") as h:
        if CHROM_NAME in h:
            g = h[CHROM_NAME]
            if "coordinates" in g:
                coords = np.asarray(g["coordinates"])
                for cid in range(n_cells):
                    key = str(cid)
                    if key not in g:
                        continue
                    mat = np.zeros((n_beads, n_beads), dtype=np.float32)
                    vals = np.asarray(g[key], dtype=np.float32)
                    mat[coords[:, 0].astype(int), coords[:, 1].astype(int)] = vals
                    mat = mat + mat.T
                    out[cid] = mat[i_idx, j_idx]
            else:
                for cid in range(n_cells):
                    key = str(cid)
                    if key in g:
                        mat = np.asarray(g[key], dtype=np.float32)
                        out[cid] = mat[i_idx, j_idx]
        else:
            coords = np.asarray(h["coordinates"])
            for cid in range(n_cells):
                key = f"cell_{cid}"
                if key not in h:
                    continue
                mat = np.zeros((n_beads, n_beads), dtype=np.float32)
                vals = np.asarray(h[key], dtype=np.float32)
                mat[coords[:, 0].astype(int), coords[:, 1].astype(int)] = vals
                mat = mat + mat.T
                out[cid] = mat[i_idx, j_idx]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out[out < 0] = 0.0
    return out


def collect_dataset(stem, neighbor, args, i_idx, j_idx):
    record_dir = args.input_root / f"{stem}_nbr{neighbor}"
    hdf5_path = record_dir / "temp" / f"{CHROM_NAME}_{EMBEDDING_NAME}_nbr_{neighbor}_impute.hdf5"
    ds = next(d for d in DATASETS if d.stem == stem)
    print(f"[higashi-collect] {stem} nbr{neighbor}: {hdf5_path}", flush=True)
    data = hdf5_to_lower_tri(hdf5_path, ds.n_cells, N_BINS, i_idx, j_idx)
    out_dir = args.output_root / "npz_lower_tri"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_higashi_nbr_{neighbor}_lower_tri.npz"
    save_npz(out_path, coo_matrix(data).tocsr())
    print(f"[higashi-collect] {stem} nbr{neighbor}: saved {out_path} shape={data.shape}",
          flush=True)


def main() -> int:
    args = parse_args()
    stems = args.datasets or [d.stem for d in DATASETS]
    i_idx, j_idx = feature_to_bins(N_BINS)
    for stem in stems:
        collect_dataset(stem, args.neighbor, args, i_idx, j_idx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())