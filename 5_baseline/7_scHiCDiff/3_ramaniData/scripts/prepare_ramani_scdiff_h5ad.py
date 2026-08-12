#!/usr/bin/env python3
"""Prepare Ramani per-chromosome h5ad files for scHiC-Diff v5 fast."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


COMMON_DIR = (
    Path("/public/home/hpc254701055/2_projects/10_schicdiff")
    / "1_scHiC/5_baseline/paperplots/4_ramani_clustering_metrics/scripts"
)
sys.path.insert(0, str(COMMON_DIR.parent))
from scripts import ramani_imputation_common as common  # noqa: E402


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = BASE_DIR / "input"


def write_h5ad(path: Path, matrix, obs: pd.DataFrame, var: pd.DataFrame) -> None:
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("anndata is required to write h5ad files; use the scdiff2 environment") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(X=matrix, obs=obs.copy(), var=var.copy()).write_h5ad(path)


def prepare(args: argparse.Namespace) -> Path:
    chroms = common.chroms_from_arg(args.chroms)
    cells = common.load_cell_list(args.cell_list)
    aligned_dir = args.input_root / "raw_626_chrom_npz"
    common.filter_630_to_626(args.source_dir, aligned_dir, args.cell_list, chroms, force=args.force)
    rows: list[dict[str, object]] = []
    obs = pd.DataFrame(
        {
            "cell_name": cells,
            "cell_type": [cell.split("_", 1)[0] for cell in cells],
            "dataset": "Ramani_ML1_ML3",
        },
        index=cells,
    )
    for chrom in chroms:
        matrix = sparse.load_npz(aligned_dir / f"{chrom}.npz").tocsr().astype(np.float32)
        common.assert_row_count(matrix, len(cells), chrom)
        n_bins = common.n_bins_from_upper_triangle_size(matrix.shape[1])
        var = pd.DataFrame(
            {
                "chrom": chrom,
                "feature_index": np.arange(matrix.shape[1], dtype=np.int64),
                "n_bins": n_bins,
            },
            index=[f"{chrom}_upper_{idx}" for idx in range(matrix.shape[1])],
        )
        h5ad_path = args.input_root / f"{chrom}_ramani_scdiff2.h5ad"
        if args.force or not h5ad_path.exists():
            write_h5ad(h5ad_path, matrix, obs, var)
        rows.append(
            {
                "chrom": chrom,
                "h5ad": str(h5ad_path.resolve()),
                "n_cells": len(cells),
                "n_features": int(matrix.shape[1]),
                "n_bins": int(n_bins),
            }
        )
    manifest = args.input_root / "ramani_scdiff_h5ad_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=common.DEFAULT_RAW_CHROM_DIR)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--cell-list", type=Path, default=common.DEFAULT_CELL_LIST)
    parser.add_argument("--chroms", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    prepare(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
