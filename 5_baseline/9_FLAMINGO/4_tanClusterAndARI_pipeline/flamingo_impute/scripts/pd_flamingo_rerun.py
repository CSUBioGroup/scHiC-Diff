#!/usr/bin/env python3
"""PD-space FLAMINGO t-SVD ADMM imputation for Ramani and Tan datasets.

This script applies the **successful Lee PD-space pipeline** to Ramani and Tan
data. It mirrors ``5_lee_SuperTAD_pileline/scripts/lee_flamingo_pipeline.py``
exactly:

    contact → PD (IF^(-0.25)) → RawCount txt
    → FLAMINGO t-SVD ADMM (mu=1e-4, selection=best, 500 iters)
    → PD → contact (PD^(-4)), clip outliers, restore observed

The original Ramani/Tan runs used **contact space** directly, which produced a
uniform ~1.0 background that homogenized cells and destroyed clustering. The
PD-space approach with ``selection=best`` produces moderate PD values for
missing entries → reasonable imputed contacts without uniform background.

Subcommands
-----------
- ``prep`` : read existing RawCount txt (contact) → convert to PD → write PD txt
- ``post`` : read completed PD tensor → convert to contact → clip → save output
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse


ALPHA = 0.25  # FLAMINGO PD<->IF conversion exponent (must match Lee/HiCImpute)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger("pd_flamingo")


# ---------------------------------------------------------------------------
# IF <-> PD conversion  (identical to Lee / HiCImputeData convention)
# ---------------------------------------------------------------------------

def contact_to_pd(contact: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    pd = np.zeros(contact.shape, dtype=np.float64)
    mask = np.isfinite(contact) & (contact > 0)
    pd[mask] = np.power(contact[mask], -alpha)
    return pd


def pd_to_contact(pd: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    contact = np.zeros(pd.shape, dtype=np.float64)
    mask = np.isfinite(pd) & (pd > 0)
    contact[mask] = np.power(pd[mask], -1.0 / alpha)
    contact[~np.isfinite(contact)] = 0.0
    contact[contact < 0] = 0.0
    return contact


# ---------------------------------------------------------------------------
# Manifest reading (shared format: dataset, n_cells, n_bins, ...)
# ---------------------------------------------------------------------------

def read_manifest(manifest: Path) -> list[dict]:
    with manifest.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


# ---------------------------------------------------------------------------
# prep: contact RawCount txt → PD RawCount txt
# ---------------------------------------------------------------------------

def prep_dataset(
    dataset: str,
    input_root: Path,
    output_input_root: Path,
    contact_subdir: str,
    pd_subdir: str,
    force: bool = False,
) -> Path:
    """Read contact RawCount txt, convert to PD, write PD RawCount txt."""
    contact_dir = input_root / dataset / contact_subdir
    pd_input_dir = output_input_root / dataset / pd_subdir
    marker = output_input_root / dataset / ".pd_complete"

    if marker.exists() and not force:
        LOGGER.info("prep %s: already done (use --force to redo)", dataset)
        return output_input_root / dataset

    files = sorted(
        contact_dir.glob("RawCount_Cell_*.txt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not files:
        raise FileNotFoundError(f"No RawCount txt in {contact_dir}")

    n_cells = len(files)
    n_bins = None
    pd_input_dir.mkdir(parents=True, exist_ok=True)

    observed_pd = []
    for cell_idx, fpath in enumerate(files):
        contact = np.loadtxt(fpath, delimiter="\t", dtype=np.float64)
        contact[~np.isfinite(contact)] = 0.0
        contact[contact < 0] = 0.0
        contact = np.maximum(contact, contact.T)
        np.fill_diagonal(contact, 0.0)
        n_bins = n_bins or contact.shape[0]

        pd_mat = contact_to_pd(contact)
        observed_pd.append(pd_mat)

        dst = pd_input_dir / f"RawCount_Cell_{cell_idx + 1:03d}.txt"
        np.savetxt(dst, pd_mat, fmt="%.10g", delimiter="\t")

    observed_pd_tensor = np.stack(observed_pd, axis=0)
    np.save(output_input_root / dataset / "observed_pd_tensor.npy", observed_pd_tensor)

    # Also save observed contact tensor for post-processing
    np.save(output_input_root / dataset / "observed_contact_tensor.npy",
            np.stack([np.loadtxt(f, delimiter="\t", dtype=np.float64) for f in files], axis=0))

    marker.write_text("complete\n")
    LOGGER.info("prep %s: %d cells, %d bins, PD tensor %s, nnz=%d",
                dataset, n_cells, n_bins, observed_pd_tensor.shape,
                int(np.count_nonzero(observed_pd_tensor)))
    return output_input_root / dataset


# ---------------------------------------------------------------------------
# post: completed PD tensor → contact → clip → save
# ---------------------------------------------------------------------------

def post_dataset(
    dataset: str,
    pd_input_root: Path,
    output_root: Path,
    final_output_dir: Path,
    output_format: str,
    npy_name: str = "completed_tensor.npy",
    clip_factor: float = 2.0,
    n_bins: int | None = None,
    n_cells: int | None = None,
) -> Path:
    """Read completed PD tensor, convert to contact, clip, save output."""
    completed_path = output_root / dataset / npy_name
    if not completed_path.exists():
        raise FileNotFoundError(f"Missing completed tensor: {completed_path}")

    completed_pd = np.real(np.load(completed_path)).astype(np.float64)
    observed_pd = np.load(pd_input_root / dataset / "observed_pd_tensor.npy").astype(np.float64)
    observed_contact = np.load(pd_input_root / dataset / "observed_contact_tensor.npy").astype(np.float64)

    if completed_pd.shape != observed_pd.shape:
        raise ValueError(f"Shape mismatch for {dataset}: completed={completed_pd.shape}, observed={observed_pd.shape}")

    n_cells_actual = completed_pd.shape[0]
    n_bins_actual = completed_pd.shape[1]
    LOGGER.info("post %s: completed %s, converting PD -> contact", dataset, completed_pd.shape)

    # ---- Clip: tiny PD -> contact=0 (same logic as Lee) ----
    obs_contact_positive = pd_to_contact(observed_pd)
    max_observed = float(obs_contact_positive[obs_contact_positive > 0].max()) if np.any(obs_contact_positive > 0) else 1.0
    max_contact_allowed = max_observed * clip_factor
    pd_threshold = float(np.power(max_contact_allowed, -ALPHA))
    tiny_pd_mask = (completed_pd > 0) & (completed_pd < pd_threshold)
    completed_pd = completed_pd.copy()
    completed_pd[tiny_pd_mask] = 0.0
    LOGGER.info("post %s: max_observed=%.4g, clip_factor=%.2f, PD threshold=%.6g, %d entries zeroed",
                dataset, max_observed, clip_factor, pd_threshold, int(tiny_pd_mask.sum()))

    # ---- PD -> contact ----
    completed_contact = pd_to_contact(completed_pd)

    # ---- Per-cell post-processing ----
    final_output_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "ramani_chrom_npz":
        # Ramani: save as (n_cells, n_upper_features) sparse CSR per dataset
        upper = np.triu_indices(n_bins_actual, k=1)
        result = completed_contact[:, upper[0], upper[1]].astype(np.float32)
        # restore observed
        obs_upper = observed_contact[:, upper[0], upper[1]]
        obs_mask = obs_upper > 0
        result[obs_mask] = obs_upper[obs_mask].astype(np.float32)
        result[~np.isfinite(result)] = 0.0
        result[result < 0] = 0.0
        out_path = final_output_dir / f"{dataset}.npz"
        sparse.save_npz(out_path, sparse.csr_matrix(result))
        LOGGER.info("post %s: saved %s, shape=%s, nnz=%d", dataset, out_path, result.shape, np.count_nonzero(result))
        return out_path

    elif output_format == "tan_unified_npz":
        # Tan: save as (n_cells, n_upper_features) sparse COO, named FLAMINGO_PD_{seg}.npz
        upper = np.triu_indices(n_bins_actual, k=1)
        vectors = []
        for c in range(n_cells_actual):
            mat = completed_contact[c].copy()
            mat[~np.isfinite(mat)] = 0.0
            mat[mat < 0] = 0.0
            mat = np.maximum(mat, mat.T)
            np.fill_diagonal(mat, 0.0)
            # restore observed
            obs_mask = observed_contact[c] > 0
            mat[obs_mask] = observed_contact[c][obs_mask]
            vec = mat[upper]
            vectors.append(vec)
        result = np.vstack(vectors).astype(np.float64)
        out_path = final_output_dir / f"FLAMINGO_PD_{dataset}.npz"
        sparse.save_npz(out_path, sparse.coo_matrix(result))
        LOGGER.info("post %s: saved %s, shape=%s, nnz=%d", dataset, out_path, result.shape, np.count_nonzero(result))
        return out_path

    else:
        raise ValueError(f"Unknown output_format: {output_format}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prep", help="contact RawCount -> PD RawCount")
    p_prep.add_argument("--dataset", required=True)
    p_prep.add_argument("--input-root", type=Path, required=True,
                        help="Root of existing contact-space input (e.g. input/)")
    p_prep.add_argument("--pd-input-root", type=Path, required=True,
                        help="Where to write PD-space input (e.g. input_pd_lee/)")
    p_prep.add_argument("--contact-subdir", default="contact_matrices")
    p_prep.add_argument("--pd-subdir", default="distance_matrices")
    p_prep.add_argument("--force", action="store_true")

    p_post = sub.add_parser("post", help="completed PD -> contact -> save")
    p_post.add_argument("--dataset", required=True)
    p_post.add_argument("--pd-input-root", type=Path, required=True)
    p_post.add_argument("--output-root", type=Path, required=True,
                        help="FLAMINGO output root (where completed_tensor.npy lives)")
    p_post.add_argument("--final-output-dir", type=Path, required=True,
                        help="Where to save the final contact NPZ")
    p_post.add_argument("--output-format", choices=("ramani_chrom_npz", "tan_unified_npz"), required=True)
    p_post.add_argument("--clip-factor", type=float, default=2.0)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prep":
        prep_dataset(
            dataset=args.dataset,
            input_root=args.input_root.resolve(),
            output_input_root=args.pd_input_root.resolve(),
            contact_subdir=args.contact_subdir,
            pd_subdir=args.pd_subdir,
            force=args.force,
        )
    elif args.command == "post":
        post_dataset(
            dataset=args.dataset,
            pd_input_root=args.pd_input_root.resolve(),
            output_root=args.output_root.resolve(),
            final_output_dir=args.final_output_dir.resolve(),
            output_format=args.output_format,
            clip_factor=args.clip_factor,
        )


if __name__ == "__main__":
    main(sys.argv[1:])