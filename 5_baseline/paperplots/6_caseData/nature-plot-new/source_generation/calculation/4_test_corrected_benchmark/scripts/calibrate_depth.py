#!/usr/bin/env python3
"""Create fixed-depth and raw-depth-matched sparse derivatives without overwriting inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
    }


def _validate_matrix(matrix: sparse.spmatrix, label: str) -> sparse.csr_matrix:
    matrix = matrix.tocsr().astype(np.float64)
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"{label} contains non-finite values")
    if matrix.data.size and matrix.data.min() < 0:
        raise ValueError(f"{label} contains negative values")
    return matrix


def calibrate_rows(
    matrix: sparse.spmatrix,
    target_depths: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    matrix = _validate_matrix(matrix, "source matrix")
    target_depths = np.asarray(target_depths, dtype=np.float64)
    if target_depths.ndim != 1 or target_depths.size != matrix.shape[0]:
        raise ValueError("target depths must have one value per source row")
    if not np.isfinite(target_depths).all() or (target_depths < 0).any():
        raise ValueError("target depths must be finite and non-negative")

    source_depths = np.asarray(matrix.sum(axis=1)).ravel()
    impossible = (source_depths == 0) & (target_depths > 0)
    if impossible.any():
        raise ValueError(
            f"cannot assign a positive target to {int(impossible.sum())} zero-depth source rows"
        )
    scales = np.zeros_like(source_depths)
    nonzero = source_depths > 0
    scales[nonzero] = target_depths[nonzero] / source_depths[nonzero]
    calibrated = matrix.multiply(scales[:, np.newaxis]).tocsr()
    calibrated.eliminate_zeros()
    return calibrated, scales


def calibrate_files(
    source_path: str | Path,
    raw_path: str | Path,
    output_dir: str | Path,
    *,
    fixed_depth: float = 85.0,
    force: bool = False,
) -> dict[str, Path]:
    source_path = Path(source_path).resolve()
    raw_path = Path(raw_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fixed": output_dir / "denoise_recon_inv_depth85_fixed.npz",
        "raw_matched": output_dir / "denoise_recon_inv_depth_matched_raw.npz",
        "metadata": output_dir / "depth_calibration_metadata.json",
    }
    if not force:
        existing = [path for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")
    if source_path in outputs.values() or raw_path in outputs.values():
        raise ValueError("output paths must differ from source paths")

    source = _validate_matrix(sparse.load_npz(source_path), "source matrix")
    raw = _validate_matrix(sparse.load_npz(raw_path), "raw matrix")
    if source.shape != raw.shape:
        raise ValueError(f"shape mismatch: source={source.shape}, raw={raw.shape}")

    source_depths = np.asarray(source.sum(axis=1)).ravel()
    raw_depths = np.asarray(raw.sum(axis=1)).ravel()
    fixed_targets = np.full(source.shape[0], float(fixed_depth), dtype=np.float64)
    fixed, fixed_scales = calibrate_rows(source, fixed_targets)
    raw_matched, raw_scales = calibrate_rows(source, raw_depths)
    fixed = fixed.astype(np.float32)
    raw_matched = raw_matched.astype(np.float32)

    sparse.save_npz(outputs["fixed"], fixed, compressed=True)
    sparse.save_npz(outputs["raw_matched"], raw_matched, compressed=True)

    fixed_after = np.asarray(fixed.sum(axis=1)).ravel()
    raw_after = np.asarray(raw_matched.sum(axis=1)).ravel()
    metadata = {
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "raw": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "fixed_depth": float(fixed_depth),
        "shape": list(source.shape),
        "source_nnz": int(source.nnz),
        "source_depth": _summary(source_depths),
        "raw_depth": _summary(raw_depths),
        "outputs": {
            "fixed": {
                "mode": "fixed_depth",
                "path": str(outputs["fixed"]),
                "sha256": _sha256(outputs["fixed"]),
                "nnz": int(fixed.nnz),
                "scale_factor": _summary(fixed_scales),
                "depth_after": _summary(fixed_after),
                "max_abs_depth_error": float(np.max(np.abs(fixed_after - fixed_targets))),
            },
            "raw_matched": {
                "mode": "raw_depth_matched",
                "path": str(outputs["raw_matched"]),
                "sha256": _sha256(outputs["raw_matched"]),
                "nnz": int(raw_matched.nnz),
                "scale_factor": _summary(raw_scales),
                "depth_after": _summary(raw_after),
                "max_abs_depth_error": float(np.max(np.abs(raw_after - raw_depths))),
            },
        },
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2) + "\n")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fixed-depth", type=float, default=85.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    outputs = calibrate_files(
        args.source,
        args.raw,
        args.output_dir,
        fixed_depth=args.fixed_depth,
        force=args.force,
    )
    for label, path in outputs.items():
        print(f"{label}\t{path}")


if __name__ == "__main__":
    main()
