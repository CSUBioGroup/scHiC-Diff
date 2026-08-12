#!/usr/bin/env python3
"""Convert selected_cells.npz or an entire loop-result directory into local .hic files."""

import argparse
from pathlib import Path
import subprocess

import numpy as np
from scipy.sparse import csr_matrix


def infer_n_bins(n_features, triu_k=None):
    """Infer matrix dimension from upper-triangle feature count. Returns (n_bins, triu_k)."""
    if triu_k is not None:
        if triu_k == 0:
            n_bins = int((-1 + np.sqrt(1 + 8 * n_features)) / 2)
            if n_bins * (n_bins + 1) // 2 != n_features:
                raise ValueError(f"n_features={n_features} is not valid triu(k=0)")
            return n_bins, 0
        elif triu_k == 1:
            n_bins = int((1 + np.sqrt(1 + 8 * n_features)) / 2)
            if n_bins * (n_bins - 1) // 2 != n_features:
                raise ValueError(f"n_features={n_features} is not valid triu(k=1)")
            return n_bins, 1
        else:
            raise ValueError(f"triu_k must be 0 or 1, got {triu_k}")
    n_bins_k0 = int((-1 + np.sqrt(1 + 8 * n_features)) / 2)
    if n_bins_k0 * (n_bins_k0 + 1) // 2 == n_features:
        return n_bins_k0, 0
    n_bins_k1 = int((1 + np.sqrt(1 + 8 * n_features)) / 2)
    if n_bins_k1 * (n_bins_k1 - 1) // 2 == n_features:
        return n_bins_k1, 1
    raise ValueError(f"Cannot infer a valid upper-triangle matrix size from n_features={n_features}")


def triu_to_matrix(triu_flat, n_bins, triu_k=0):
    mat = np.zeros((n_bins, n_bins))
    idx = np.triu_indices(n_bins, k=triu_k)
    mat[idx] = triu_flat
    mat = mat + mat.T - np.diag(np.diag(mat))
    return mat


def npz_to_pre(npz_file, pre_file, resolution=20000, chrom="chr1", triu_k=None):
    payload = np.load(npz_file, allow_pickle=True)
    mat = csr_matrix((payload["data"], payload["indices"], payload["indptr"]), shape=payload["shape"])
    n_cells, n_features = mat.shape
    n_bins, triu_k = infer_n_bins(n_features, triu_k=triu_k)

    merged = np.zeros((n_bins, n_bins))
    for cell_idx in range(n_cells):
        merged += triu_to_matrix(mat[cell_idx].toarray().ravel(), n_bins, triu_k=triu_k)

    with open(pre_file, "w") as handle:
        for i in range(n_bins):
            for j in range(i, n_bins):
                count = merged[i, j]
                if count > 0:
                    pos1 = i * resolution
                    pos2 = j * resolution
                    handle.write(f"0 {chrom} {pos1} 0 1 {chrom} {pos2} 1 {int(count)}\n")
    return n_bins


def pre_to_hic(pre_file, hic_file, chrom_sizes_file, juicer_jar, resolution):
    cmd = [
        "java",
        "-Xmx8g",
        "-jar",
        str(juicer_jar),
        "pre",
        "-n",
        "-r",
        str(resolution),
        str(pre_file),
        str(hic_file),
        str(chrom_sizes_file),
    ]
    subprocess.run(cmd, check=True)


def convert_single(npz_file, hic_file, juicer_jar, resolution=20000, chrom="chr1", chrom_size=None, keep_pre=False, triu_k=None):
    hic_file = Path(hic_file)
    hic_file.parent.mkdir(parents=True, exist_ok=True)
    pre_file = hic_file.with_suffix(".pre")
    n_bins = npz_to_pre(npz_file, pre_file, resolution=resolution, chrom=chrom, triu_k=triu_k)

    inferred_chrom_size = chrom_size or (n_bins * resolution)
    chrom_sizes_file = hic_file.parent / "chrom.sizes"
    chrom_sizes_file.write_text(f"{chrom}\t{inferred_chrom_size}\n")

    if hic_file.exists():
        hic_file.unlink()
    pre_to_hic(pre_file, hic_file, chrom_sizes_file, juicer_jar=juicer_jar, resolution=resolution)
    if not keep_pre and pre_file.exists():
        pre_file.unlink()


def convert_directory(samples_dir, hic_dir, juicer_jar, resolution=20000, chrom="chr1", chrom_size=None, keep_pre=False, triu_k=None):
    samples_dir = Path(samples_dir)
    hic_dir = Path(hic_dir)
    hic_dir.mkdir(parents=True, exist_ok=True)
    for sample_dir in sorted(path for path in samples_dir.iterdir() if path.is_dir()):
        npz_file = sample_dir / "selected_cells.npz"
        if not npz_file.exists():
            continue
        hic_file = hic_dir / f"{sample_dir.name}.hic"
        print(f"Converting {npz_file} -> {hic_file}")
        convert_single(
            npz_file=npz_file,
            hic_file=hic_file,
            juicer_jar=juicer_jar,
            resolution=resolution,
            chrom=chrom,
            chrom_size=chrom_size,
            keep_pre=keep_pre,
            triu_k=triu_k,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Convert selected_cells.npz to .hic")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-npz", help="Single selected_cells.npz file")
    group.add_argument("--samples-dir", help="Loop-result directory containing */selected_cells.npz")
    parser.add_argument("--hic-file", help="Output .hic file for --input-npz mode")
    parser.add_argument("--hic-dir", help="Output directory for --samples-dir mode")
    parser.add_argument("--juicer-jar", required=True)
    parser.add_argument("--resolution", type=int, default=20000)
    parser.add_argument("--chrom", default="chr1")
    parser.add_argument("--chrom-size", type=int, default=None)
    parser.add_argument("--keep-pre", action="store_true")
    parser.add_argument("--triu-k", type=int, default=None, choices=[0, 1],
                        help="Upper-triangle convention: 0=include diagonal, 1=exclude diagonal. Auto-detect if not specified.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input_npz:
        if not args.hic_file:
            raise SystemExit("--hic-file is required when using --input-npz")
        convert_single(
            npz_file=args.input_npz,
            hic_file=args.hic_file,
            juicer_jar=args.juicer_jar,
            resolution=args.resolution,
            chrom=args.chrom,
            chrom_size=args.chrom_size,
            keep_pre=args.keep_pre,
            triu_k=args.triu_k,
        )
    else:
        if not args.hic_dir:
            raise SystemExit("--hic-dir is required when using --samples-dir")
        convert_directory(
            samples_dir=args.samples_dir,
            hic_dir=args.hic_dir,
            juicer_jar=args.juicer_jar,
            resolution=args.resolution,
            chrom=args.chrom,
            chrom_size=args.chrom_size,
            keep_pre=args.keep_pre,
            triu_k=args.triu_k,
        )


if __name__ == "__main__":
    main()
