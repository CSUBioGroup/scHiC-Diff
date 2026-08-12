#!/usr/bin/env python3
"""Concatenate per-chromosome Higashi sparse matrices for the 1Mb plot workflow."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from scipy import sparse


def chrom_key(path: Path) -> tuple[int, str]:
    name = path.stem.removeprefix("chr")
    if name.isdigit():
        return int(name), ""
    return 10_000, name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrom-npz-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("cell_labels.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.chrom_npz_dir.glob("chr*.npz"), key=chrom_key)
    if not paths:
        raise FileNotFoundError(f"No chr*.npz files in {args.chrom_npz_dir}")
    matrices = [sparse.load_npz(path).tocsr() for path in paths]
    n_cells = matrices[0].shape[0]
    if any(matrix.shape[0] != n_cells for matrix in matrices):
        raise ValueError("Chromosome matrices do not have a consistent cell count")
    if args.labels.exists() and len(pd.read_csv(args.labels)) != n_cells:
        raise ValueError(f"{args.labels} does not match {n_cells} matrix rows")

    merged = sparse.hstack(matrices, format="csr")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(args.output, merged)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(
            {
                "chrom_npz_dir": str(args.chrom_npz_dir),
                "chromosomes": [path.stem for path in paths],
                "matrix_shape": list(merged.shape),
                "nnz": int(merged.nnz),
                "labels": str(args.labels),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {args.output}: shape={merged.shape}, nnz={merged.nnz}")


if __name__ == "__main__":
    main()
