#!/usr/bin/env python3
"""Prepare FLAMINGO v3 paramsweep h5ad for Higashi (higashi_v1 format).

Builds per-dataset, per-neighbor input directories containing
``config.JSON``, ``data/`` (chrom.sizes + label_info.pickle) and ``temp/``
(data.npy, weight.npy, chrom_start_end.npy).  Two neighbor configs (0 and 5)
are generated per dataset.  Observed contacts come from h5ad
``layers['counts']``; GT (``layers['gt']``) is kept separately for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse

from v3_common import (CHROM_NAME, DEFAULT_DATA_DIR, DATASETS, N_BINS,
                       N_FEATURES, load_layer, feature_to_bins)

DEFAULT_INPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "6_Higashi/2_FLAMINGOData/v3_inputData")
EMBEDDING_NAME = "flamingo_higashi"
RESOLUTION = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--neighbors", type=int, nargs="*", default=[0, 5])
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--embedding-epoch", type=int, default=60)
    parser.add_argument("--no-nbr-epoch", type=int, default=45)
    parser.add_argument("--with-nbr-epoch", type=int, default=30)
    parser.add_argument("--cpu-num", type=int, default=20)
    parser.add_argument("--gpu-num", type=int, default=1)
    parser.add_argument("--loss-mode", default="zinb")
    parser.add_argument("--min-delta", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_config(record_dir, n_cells, neighbor_num, args):
    temp_dir = record_dir / "temp"
    data_dir = record_dir / "data"
    return {
        "data_dir": str(data_dir.resolve()),
        "temp_dir": str(temp_dir.resolve()),
        "genome_reference_path": str((data_dir / "flamingo.chrom.sizes").resolve()),
        "chrom_list": [CHROM_NAME],
        "impute_list": [CHROM_NAME],
        "resolution": args.resolution,
        "resolution_cell": args.resolution,
        "minimum_distance": 0, "maximum_distance": -1,
        "minimum_impute_distance": 0, "maximum_impute_distance": -1,
        "local_transfer_range": 1,
        "dimensions": args.dimensions,
        "embedding_name": EMBEDDING_NAME,
        "loss_mode": args.loss_mode,
        "neighbor_num": neighbor_num,
        "cpu_num": args.cpu_num,
        "cpu_num_torch": max(1, min(args.cpu_num, 4)),
        "gpu_num": args.gpu_num,
        "embedding_epoch": args.embedding_epoch,
        "no_nbr_epoch": args.no_nbr_epoch,
        "with_nbr_epoch": args.with_nbr_epoch,
        "impute_no_nbr": True,
        "impute_with_nbr": neighbor_num > 0,
        "correct_be_impute": False,
        "precompute_weighted_nbr": True,
        "input_format": "higashi_v1",
        "structured": False,
        "contact_header": ["cell_id", "chrom1", "chrom2", "pos1", "pos2", "count"],
        "header_included": True,
    }


def build_dataset(stem, data_dir, input_root, neighbor, args):
    record_dir = input_root / f"{stem}_nbr{neighbor}"
    cfg = record_dir / "config.JSON"
    data_npy = record_dir / "temp" / "data.npy"
    if cfg.exists() and data_npy.exists() and not args.overwrite:
        print(f"[higashi-prep] {stem} nbr{neighbor}: skip (exists)", flush=True)
        return
    h5ad = data_dir / f"{stem}_scdiff2.h5ad"
    print(f"[higashi-prep] {stem} nbr{neighbor}: loading counts", flush=True)
    observed = load_layer(h5ad, "counts").tocsr()
    n_cells = observed.shape[0]
    i_idx, j_idx = feature_to_bins(N_BINS)
    valid_delta = (j_idx - i_idx) >= args.min_delta
    rows_parts, weights_parts = [], []
    for cid in range(n_cells):
        s, e = observed.indptr[cid], observed.indptr[cid + 1]
        feat = observed.indices[s:e]
        w = observed.data[s:e]
        m = np.isfinite(w) & (w > 0) & valid_delta[feat]
        if not np.any(m):
            continue
        feat = feat[m]
        w = w[m].astype(np.float32, copy=False)
        rows = np.column_stack([
            np.full(feat.size, cid, dtype=np.int64),
            np.zeros(feat.size, dtype=np.int64),
            i_idx[feat], j_idx[feat]])
        rows_parts.append(rows)
        weights_parts.append(w)
    if not rows_parts:
        raise ValueError(f"{stem}: no positive contacts after filter")
    rows_all = np.concatenate(rows_parts, axis=0)
    w_all = np.concatenate(weights_parts, axis=0)
    temp = record_dir / "temp"
    data_dir2 = record_dir / "data"
    temp.mkdir(parents=True, exist_ok=True)
    (temp / "raw").mkdir(parents=True, exist_ok=True)
    data_dir2.mkdir(parents=True, exist_ok=True)
    np.save(temp / "data.npy", rows_all, allow_pickle=True)
    np.save(temp / "weight.npy", w_all.astype(np.float32, copy=False), allow_pickle=True)
    np.save(temp / "chrom_start_end.npy", np.array([[0, N_BINS]], dtype=np.int64))
    (data_dir2 / "flamingo.chrom.sizes").write_text(f"{CHROM_NAME}\t{N_BINS * args.resolution}\n")
    labels = {"cell_name": [f"{stem}_cell_{i + 1}" for i in range(n_cells)],
              "cell_type": [stem] * n_cells, "dataset_id": [stem] * n_cells}
    with (data_dir2 / "label_info.pickle").open("wb") as h:
        pickle.dump(labels, h, protocol=4)
    config = build_config(record_dir, n_cells, neighbor, args)
    cfg.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(f"[higashi-prep] {stem} nbr{neighbor}: {n_cells} cells, "
          f"{len(w_all)} contacts -> {record_dir}", flush=True)


def main() -> int:
    args = parse_args()
    args.input_root.mkdir(parents=True, exist_ok=True)
    stems = args.datasets or [d.stem for d in DATASETS]
    for stem in stems:
        for nbr in args.neighbors:
            build_dataset(stem, args.data_dir, args.input_root, nbr, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())