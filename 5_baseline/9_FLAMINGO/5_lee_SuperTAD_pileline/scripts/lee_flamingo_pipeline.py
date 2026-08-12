#!/usr/bin/env python3
"""Run contact-space FLAMINGO completion for Lee PDGFRA scHi-C data.

The pipeline has one data contract: read 49 x 49 contact matrices, complete the
contact tensor, restore observed contacts, and write symmetric nonnegative
contact NPZ files. No distance-space conversion is supported.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy import sparse


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR))
import config as cfg  # noqa: E402


LRTC_SCRIPT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/"
    "4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/"
    "run_flamingo_pyfftw_completion.py"
)
METHOD_NAME = "FLAMINGO_fixed_contact"
INPUT_SUBDIR = "contact_matrices"
WORK_DIR = PIPELINE_DIR / "work" / METHOD_NAME
FLAMINGO_INPUT_ROOT = WORK_DIR / "input"
FLAMINGO_OUTPUT_ROOT = WORK_DIR / "output"
IMPUTED_DIR = PIPELINE_DIR / "imputed" / METHOD_NAME

DEFAULT_MAX_ITER = 500
DEFAULT_TOL = 1e-4
DEFAULT_MU = 1e-4
DEFAULT_MAX_MU = 1e10
DEFAULT_RHO = 1.1
DEFAULT_PATIENCE = 0
DEFAULT_MIN_REL_IMPROVEMENT = 0.0
DEFAULT_N_THREADS = 8
DEFAULT_PYTHON = (
    "/public/home/hpc254701055/micromamba/envs/"
    "unicorn_and_flamingo_env/bin/python"
)

CELL_TYPES = list(cfg.CELL_TYPES)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger("lee_flamingo_contact")


def list_input_npz(cell_type: str) -> list[Path]:
    """Return the ordered source contact files for one cell type."""
    files = sorted(Path(cfg.PER_CELL_NPZ_DIR).glob(f"{cell_type}_cell_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No input NPZ at {cfg.PER_CELL_NPZ_DIR}/{cell_type}_cell_*.npz"
        )
    return files


def clean_contact_matrix(matrix: np.ndarray, label: str) -> np.ndarray:
    """Return a finite, symmetric, nonnegative contact matrix."""
    contact = np.asarray(matrix, dtype=np.float64)
    if contact.ndim != 2 or contact.shape[0] != contact.shape[1]:
        raise ValueError(f"{label} must be square, got {contact.shape}")
    contact = contact.copy()
    contact[~np.isfinite(contact)] = 0.0
    contact[contact < 0.0] = 0.0
    contact = np.maximum(contact, contact.T)
    np.fill_diagonal(contact, 0.0)
    return contact


def restore_observed_contacts(
    completed: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Restore positive observed contacts without breaking symmetry."""
    completed_contact = clean_contact_matrix(completed, "completed contact")
    observed_contact = clean_contact_matrix(observed, "observed contact")
    if completed_contact.shape != observed_contact.shape:
        raise ValueError(
            "completed/observed shape mismatch: "
            f"{completed_contact.shape} != {observed_contact.shape}"
        )
    observed_mask = observed_contact > 0.0
    completed_contact[observed_mask] = observed_contact[observed_mask]
    return completed_contact


def prep_cell_type(
    cell_type: str,
    force: bool = False,
    input_root: Path = FLAMINGO_INPUT_ROOT,
) -> Path:
    """Write source NPZ matrices as contact-space FLAMINGO input files."""
    input_dir = Path(input_root) / cell_type
    matrix_dir = input_dir / INPUT_SUBDIR
    complete_marker = input_dir / ".complete"
    observed_path = input_dir / "observed_contact_tensor.npy"

    if complete_marker.exists() and observed_path.exists() and not force:
        LOGGER.info("prep: %s already prepared (use --force to redo)", cell_type)
        return input_dir

    matrix_dir.mkdir(parents=True, exist_ok=True)
    npz_files = list_input_npz(cell_type)
    observed_tensor = np.zeros(
        (len(npz_files), cfg.N_BINS, cfg.N_BINS),
        dtype=np.float64,
    )
    index_rows: list[dict[str, object]] = []

    LOGGER.info("prep %s: %d contact matrices", cell_type, len(npz_files))
    for cell_idx, npz_path in enumerate(npz_files):
        contact = clean_contact_matrix(
            sparse.load_npz(npz_path).toarray(),
            str(npz_path),
        )
        if contact.shape != (cfg.N_BINS, cfg.N_BINS):
            raise ValueError(
                f"{npz_path} has shape {contact.shape}; "
                f"expected {(cfg.N_BINS, cfg.N_BINS)}"
            )
        observed_tensor[cell_idx] = contact
        destination_name = f"RawCount_Cell_{cell_idx + 1:03d}.txt"
        np.savetxt(
            matrix_dir / destination_name,
            contact,
            fmt="%.10g",
            delimiter="\t",
        )
        n_observed = int(np.count_nonzero(contact))
        index_rows.append(
            {
                "cell_idx": cell_idx,
                "cell_number": cell_idx + 1,
                "input_file": destination_name,
                "source_npz": str(npz_path),
                "n_observed": n_observed,
                "observed_fraction": n_observed / contact.size,
            }
        )

    np.save(observed_path, observed_tensor)
    with (input_dir / "input_file_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    metadata = {
        "cell_type": cell_type,
        "n_cells": len(npz_files),
        "n_bins": cfg.N_BINS,
        "method": METHOD_NAME,
        "space": "contact_count",
        "source_dir": cfg.PER_CELL_NPZ_DIR,
        "input_subdir": INPUT_SUBDIR,
    }
    with (input_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    complete_marker.write_text("complete\n")

    LOGGER.info(
        "prep %s: wrote %d contact files to %s",
        cell_type,
        len(npz_files),
        matrix_dir,
    )
    return input_dir


def load_completed_contact_tensor(path: Path) -> np.ndarray:
    """Load a completed tensor and reject meaningful imaginary residuals."""
    completed = np.load(path)
    if np.iscomplexobj(completed):
        imaginary_max = float(np.max(np.abs(completed.imag)))
        real_scale = max(1.0, float(np.max(np.abs(completed.real))))
        if imaginary_max > 1e-8 * real_scale:
            raise ValueError(
                f"Completed tensor has non-negligible imaginary residual: "
                f"{imaginary_max:.6g}"
            )
        completed = completed.real
    completed = np.asarray(completed, dtype=np.float64)
    if completed.ndim != 3 or completed.shape[1:] != (cfg.N_BINS, cfg.N_BINS):
        raise ValueError(
            "Completed tensor must have shape "
            f"(n_cells, {cfg.N_BINS}, {cfg.N_BINS}), got {completed.shape}"
        )
    return completed


def post_cell_type(
    cell_type: str,
    npy_name: str = "completed_tensor.npy",
    input_root: Path = FLAMINGO_INPUT_ROOT,
    output_root: Path = FLAMINGO_OUTPUT_ROOT,
    imputed_dir: Path = IMPUTED_DIR,
) -> Path:
    """Validate completed contacts, restore observations, and write NPZ files."""
    input_dir = Path(input_root) / cell_type
    output_dir = Path(output_root) / cell_type
    imputed_dir = Path(imputed_dir)
    completed_path = output_dir / npy_name
    observed_path = input_dir / "observed_contact_tensor.npy"
    if not completed_path.is_file():
        raise FileNotFoundError(f"Missing completed tensor: {completed_path}")
    if not observed_path.is_file():
        raise FileNotFoundError(f"Missing observed contact tensor: {observed_path}")

    completed_tensor = load_completed_contact_tensor(completed_path)
    observed_tensor = np.asarray(np.load(observed_path), dtype=np.float64)
    if completed_tensor.shape != observed_tensor.shape:
        raise ValueError(
            f"Shape mismatch for {cell_type}: completed={completed_tensor.shape}, "
            f"observed={observed_tensor.shape}"
        )

    npz_files = list_input_npz(cell_type)
    if len(npz_files) != completed_tensor.shape[0]:
        raise ValueError(
            f"Cell count mismatch for {cell_type}: NPZ={len(npz_files)}, "
            f"completed={completed_tensor.shape[0]}"
        )

    imputed_dir.mkdir(parents=True, exist_ok=True)
    completed_nnz = 0
    completed_max = 0.0
    completed_min_positive = float("inf")
    max_observed = 0.0

    for cell_idx, source_path in enumerate(npz_files):
        original = sparse.load_npz(source_path).toarray()
        contact = restore_observed_contacts(
            completed_tensor[cell_idx],
            original,
        )
        positive = contact[contact > 0.0]
        completed_nnz += int(positive.size)
        if positive.size:
            completed_max = max(completed_max, float(positive.max()))
            completed_min_positive = min(
                completed_min_positive,
                float(positive.min()),
            )
        original_contact = clean_contact_matrix(original, str(source_path))
        max_observed = max(max_observed, float(original_contact.max()))
        destination = imputed_dir / f"{cell_type}_cell_{cell_idx:04d}.npz"
        sparse.save_npz(destination, sparse.csr_matrix(contact))

    summary = {
        "cell_type": cell_type,
        "n_cells": int(completed_tensor.shape[0]),
        "n_bins": int(completed_tensor.shape[1]),
        "method": METHOD_NAME,
        "space": "contact_count",
        "observed_contacts_restored": True,
        "completed_contact_nnz": completed_nnz,
        "completed_contact_min_positive": (
            completed_min_positive
            if np.isfinite(completed_min_positive)
            else None
        ),
        "completed_contact_max": completed_max,
        "max_observed_contact": max_observed,
        "source_completed": str(completed_path),
        "output_dir": str(imputed_dir),
    }
    for summary_path in (
        output_dir / "contact_postprocess_summary.json",
        imputed_dir / f"contact_summary_{cell_type}.json",
    ):
        with summary_path.open("w") as handle:
            json.dump(summary, handle, indent=2)

    LOGGER.info(
        "post %s: wrote %d contact NPZ files to %s",
        cell_type,
        completed_tensor.shape[0],
        imputed_dir,
    )
    return imputed_dir


def run_flamingo_completion(
    cell_type: str,
    max_iter: int,
    tol: float,
    mu: float,
    max_mu: float,
    rho: float,
    n_threads: int,
    patience: int,
    min_rel_improvement: float,
    python_bin: str,
    input_root: Path = FLAMINGO_INPUT_ROOT,
    output_root: Path = FLAMINGO_OUTPUT_ROOT,
) -> None:
    """Run the validated serial complex-SVD FLAMINGO completion backend."""
    if not LRTC_SCRIPT.is_file():
        raise FileNotFoundError(f"FLAMINGO completion script missing: {LRTC_SCRIPT}")
    command = [
        python_bin,
        str(LRTC_SCRIPT),
        "--input-root", str(input_root),
        "--input-subdir", INPUT_SUBDIR,
        "--output-root", str(output_root),
        "--datasets", cell_type,
        "--max-iter", str(max_iter),
        "--tol", str(tol),
        "--mu", str(mu),
        "--max-mu", str(max_mu),
        "--rho", str(rho),
        "--n-threads", str(n_threads),
        "--svd-backend", "serial",
        "--selection", "best",
        "--patience", str(patience),
        "--min-rel-improvement", str(min_rel_improvement),
        "--keep-observed",
    ]
    LOGGER.info("FLAMINGO completion: %s", " ".join(command))
    start = time.time()
    subprocess.run(command, check=True)
    LOGGER.info(
        "FLAMINGO completion %s finished in %.1fs",
        cell_type,
        time.time() - start,
    )


def run_cell_type(args: argparse.Namespace) -> None:
    prep_cell_type(
        cell_type=args.cell_type,
        force=args.force_prep,
        input_root=args.input_root,
    )
    run_flamingo_completion(
        cell_type=args.cell_type,
        max_iter=args.max_iter,
        tol=args.tol,
        mu=args.mu,
        max_mu=args.max_mu,
        rho=args.rho,
        n_threads=args.n_threads,
        patience=args.patience,
        min_rel_improvement=args.min_rel_improvement,
        python_bin=args.python_bin,
        input_root=args.input_root,
        output_root=args.output_root,
    )
    post_cell_type(
        cell_type=args.cell_type,
        npy_name=args.npy_name,
        input_root=args.input_root,
        output_root=args.output_root,
        imputed_dir=args.imputed_dir,
    )


def add_output_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", type=Path, default=FLAMINGO_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=FLAMINGO_OUTPUT_ROOT)
    parser.add_argument("--imputed-dir", type=Path, default=IMPUTED_DIR)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prep", help="Prepare contact-space input")
    prep.add_argument("--cell-type", required=True, choices=CELL_TYPES)
    prep.add_argument("--force", action="store_true")
    prep.add_argument("--input-root", type=Path, default=FLAMINGO_INPUT_ROOT)

    post = subparsers.add_parser("post", help="Write completed contact NPZ files")
    post.add_argument("--cell-type", required=True, choices=CELL_TYPES)
    post.add_argument("--npy-name", default="completed_tensor.npy")
    add_output_paths(post)

    run = subparsers.add_parser("run", help="Prepare, complete, and postprocess")
    run.add_argument("--cell-type", required=True, choices=CELL_TYPES)
    run.add_argument("--force-prep", action="store_true")
    run.add_argument("--npy-name", default="completed_tensor.npy")
    run.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    run.add_argument("--tol", type=float, default=DEFAULT_TOL)
    run.add_argument("--mu", type=float, default=DEFAULT_MU)
    run.add_argument("--max-mu", type=float, default=DEFAULT_MAX_MU)
    run.add_argument("--rho", type=float, default=DEFAULT_RHO)
    run.add_argument("--n-threads", type=int, default=DEFAULT_N_THREADS)
    run.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    run.add_argument(
        "--min-rel-improvement",
        type=float,
        default=DEFAULT_MIN_REL_IMPROVEMENT,
    )
    run.add_argument("--python-bin", default=DEFAULT_PYTHON)
    add_output_paths(run)

    subparsers.add_parser("list", help="List supported cell types")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "list":
        for task_id, cell_type in enumerate(CELL_TYPES):
            print(f"task_id={task_id} -> {cell_type}")
    elif args.command == "prep":
        prep_cell_type(
            cell_type=args.cell_type,
            force=args.force,
            input_root=args.input_root,
        )
    elif args.command == "post":
        post_cell_type(
            cell_type=args.cell_type,
            npy_name=args.npy_name,
            input_root=args.input_root,
            output_root=args.output_root,
            imputed_dir=args.imputed_dir,
        )
    elif args.command == "run":
        run_cell_type(args)


if __name__ == "__main__":
    main(sys.argv[1:])
