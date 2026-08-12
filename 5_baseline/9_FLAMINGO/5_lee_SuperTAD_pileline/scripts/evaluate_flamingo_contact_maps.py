#!/usr/bin/env python3
"""Validate final Lee FLAMINGO contact-space output before downstream use."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import sparse
from scipy.stats import pearsonr, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR))
import config as cfg  # noqa: E402


METHOD_NAME = "FLAMINGO_fixed_contact"
METRIC_FIELDS = [
    "cell_type",
    "rho_distance",
    "pcc_49x49",
    "pcc_8x8",
    "finite",
    "symmetric",
    "nonnegative",
    "observed_restored",
    "max_contact",
    "n_sampled",
    "n_total",
    "seed",
    "gate_pass",
]
PROFILE_FIELDS = ["cell_type", "distance_bins", "mean_contact"]


def _square_matrix(matrix: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{label} must be square, got {array.shape}")
    return array


def distance_profile(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean contact at each nonzero genomic offset."""
    array = _square_matrix(matrix, "contact map")
    distances = np.arange(1, array.shape[0], dtype=np.int64)
    means = np.empty(distances.size, dtype=np.float64)
    for index, distance in enumerate(distances):
        values = np.diag(array, k=int(distance))
        finite = values[np.isfinite(values)]
        means[index] = float(finite.mean()) if finite.size else np.nan
    return distances, means


def distance_profile_spearman(matrix: np.ndarray) -> float:
    distances, means = distance_profile(matrix)
    valid = np.isfinite(means)
    if valid.sum() < 2 or np.ptp(means[valid]) == 0.0:
        return float("nan")
    return float(spearmanr(distances[valid], means[valid])[0])


def upper_triangle_pcc(first: np.ndarray, second: np.ndarray) -> float:
    first_array = _square_matrix(first, "first matrix")
    second_array = _square_matrix(second, "second matrix")
    if first_array.shape != second_array.shape:
        raise ValueError(
            f"matrix shape mismatch: {first_array.shape} != {second_array.shape}"
        )
    upper = np.triu_indices(first_array.shape[0], k=1)
    first_values = first_array[upper]
    second_values = second_array[upper]
    if not np.all(np.isfinite(first_values)) or not np.all(np.isfinite(second_values)):
        return float("nan")
    if np.ptp(first_values) == 0.0 or np.ptp(second_values) == 0.0:
        return float("nan")
    return float(pearsonr(first_values, second_values)[0])


def full_matrix_pcc(first: np.ndarray, second: np.ndarray) -> float:
    """Match the flattened PCC used by the Lee trial workflow."""
    first_array = _square_matrix(first, "first matrix").copy()
    second_array = _square_matrix(second, "second matrix").copy()
    if first_array.shape != second_array.shape:
        raise ValueError(
            f"matrix shape mismatch: {first_array.shape} != {second_array.shape}"
        )
    np.fill_diagonal(first_array, 0.0)
    np.fill_diagonal(second_array, 0.0)
    first_values = first_array.ravel()
    second_values = second_array.ravel()
    if not np.all(np.isfinite(first_values)) or not np.all(np.isfinite(second_values)):
        return float("nan")
    if np.ptp(first_values) == 0.0 or np.ptp(second_values) == 0.0:
        return float("nan")
    return float(pearsonr(first_values, second_values)[0])


def observed_contacts_restored(
    predicted: np.ndarray,
    observed: np.ndarray,
    atol: float = 1e-10,
) -> bool:
    predicted_array = _square_matrix(predicted, "predicted matrix")
    observed_array = _square_matrix(observed, "observed matrix")
    if predicted_array.shape != observed_array.shape:
        return False
    observed_clean = observed_array.copy()
    observed_clean[~np.isfinite(observed_clean)] = 0.0
    observed_clean[observed_clean < 0.0] = 0.0
    observed_clean = np.maximum(observed_clean, observed_clean.T)
    mask = observed_clean > 0.0
    return bool(
        np.allclose(
            predicted_array[mask],
            observed_clean[mask],
            rtol=0.0,
            atol=atol,
        )
    )


def hard_gate_passes(metrics: dict[str, Any]) -> bool:
    rho_distance = float(metrics["rho_distance"])
    pcc_49x49 = float(metrics["pcc_49x49"])
    pcc_8x8 = float(metrics["pcc_8x8"])
    return bool(
        np.isfinite(rho_distance)
        and np.isfinite(pcc_49x49)
        and np.isfinite(pcc_8x8)
        and rho_distance < 0.0
        and pcc_49x49 > 0.0
        and pcc_8x8 > 0.0
        and metrics["finite"]
        and metrics["symmetric"]
        and metrics["nonnegative"]
        and metrics["observed_restored"]
    )


def evaluate_cell_type(
    cell_type: str,
    imputed_dir: Path,
    observed_dir: Path,
    target_path: Path,
    n_sample: int = 30,
    seed: int = 42,
    subregion: tuple[int, int] = cfg.PDGFRA_SUB_BINS,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    imputed_dir = Path(imputed_dir)
    observed_dir = Path(observed_dir)
    target_path = Path(target_path)
    predicted_files = sorted(imputed_dir.glob(f"{cell_type}_cell_*.npz"))
    if not predicted_files:
        raise FileNotFoundError(
            f"No predicted contact maps for {cell_type} in {imputed_dir}"
        )

    n_total = len(predicted_files)
    n_selected = min(int(n_sample), n_total)
    sampled_indices = np.random.RandomState(seed).choice(
        n_total,
        size=n_selected,
        replace=False,
    )
    pseudo_bulk: np.ndarray | None = None
    all_finite = True
    all_symmetric = True
    all_nonnegative = True
    all_observed_restored = True
    sampled_files: list[str] = []

    for sampled_index in sampled_indices:
        predicted_path = predicted_files[int(sampled_index)]
        observed_path = observed_dir / predicted_path.name
        if not observed_path.is_file():
            raise FileNotFoundError(f"Missing observed map: {observed_path}")
        predicted = sparse.load_npz(predicted_path).toarray().astype(np.float64)
        observed = sparse.load_npz(observed_path).toarray().astype(np.float64)
        predicted = _square_matrix(predicted, str(predicted_path))
        if pseudo_bulk is None:
            pseudo_bulk = np.zeros_like(predicted)
        if predicted.shape != pseudo_bulk.shape:
            raise ValueError(
                f"Predicted shape mismatch: {predicted_path} has {predicted.shape}, "
                f"expected {pseudo_bulk.shape}"
            )
        pseudo_bulk += predicted
        all_finite = all_finite and bool(np.all(np.isfinite(predicted)))
        all_symmetric = all_symmetric and bool(np.allclose(predicted, predicted.T))
        all_nonnegative = all_nonnegative and bool(np.all(predicted >= 0.0))
        all_observed_restored = all_observed_restored and observed_contacts_restored(
            predicted,
            observed,
        )
        sampled_files.append(str(predicted_path))

    if pseudo_bulk is None:
        raise RuntimeError(f"No cells sampled for {cell_type}")
    target = sparse.load_npz(target_path).toarray().astype(np.float64)
    if target.shape != pseudo_bulk.shape:
        raise ValueError(f"Target shape mismatch: {target.shape} != {pseudo_bulk.shape}")

    start, end = subregion
    if start < 0 or end > pseudo_bulk.shape[0] or start >= end:
        raise ValueError(f"Invalid subregion {subregion} for {pseudo_bulk.shape}")
    distances, means = distance_profile(pseudo_bulk)
    metrics: dict[str, Any] = {
        "cell_type": cell_type,
        "rho_distance": distance_profile_spearman(pseudo_bulk),
        "pcc_49x49": full_matrix_pcc(pseudo_bulk, target),
        "pcc_8x8": full_matrix_pcc(
            pseudo_bulk[start:end, start:end],
            target[start:end, start:end],
        ),
        "finite": all_finite and bool(np.all(np.isfinite(pseudo_bulk))),
        "symmetric": all_symmetric and bool(np.allclose(pseudo_bulk, pseudo_bulk.T)),
        "nonnegative": all_nonnegative and bool(np.all(pseudo_bulk >= 0.0)),
        "observed_restored": all_observed_restored,
        "max_contact": float(np.max(pseudo_bulk)),
        "n_sampled": n_selected,
        "n_total": n_total,
        "seed": int(seed),
    }
    metrics["gate_pass"] = hard_gate_passes(metrics)

    profile_rows = [
        {
            "cell_type": cell_type,
            "distance_bins": int(distance),
            "mean_contact": float(mean),
        }
        for distance, mean in zip(distances, means, strict=True)
    ]
    provenance = {
        "method": METHOD_NAME,
        "space": "contact_count",
        "cell_type": cell_type,
        "seed": int(seed),
        "n_sampled": n_selected,
        "n_total": n_total,
        "sampled_indices": sampled_indices.astype(int).tolist(),
        "sampled_files": sampled_files,
        "imputed_dir": str(imputed_dir.resolve()),
        "observed_dir": str(observed_dir.resolve()),
        "target_path": str(target_path.resolve()),
        "subregion": [int(start), int(end)],
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return metrics, profile_rows, provenance


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_outputs(
    metric_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    provenance_records: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "contact_metrics.csv", metric_rows, METRIC_FIELDS)
    _write_csv(output_dir / "distance_profiles.csv", profile_rows, PROFILE_FIELDS)
    with (output_dir / "provenance.json").open("w") as handle:
        json.dump(provenance_records, handle, indent=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell-types",
        nargs="+",
        choices=cfg.CELL_TYPES,
        default=list(cfg.CELL_TYPES),
    )
    parser.add_argument(
        "--imputed-dir",
        type=Path,
        default=PIPELINE_DIR / "imputed" / METHOD_NAME,
    )
    parser.add_argument(
        "--observed-dir",
        type=Path,
        default=Path(cfg.PER_CELL_NPZ_DIR),
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path(cfg.TARGET_DIR),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "test_outputs/contact_space_metrics",
    )
    parser.add_argument("--n-sample", type=int, default=cfg.N_SAMPLE)
    parser.add_argument("--seed", type=int, default=cfg.BASE_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metric_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    provenance_records: list[dict[str, Any]] = []
    for cell_type in args.cell_types:
        metrics, profiles, provenance = evaluate_cell_type(
            cell_type=cell_type,
            imputed_dir=args.imputed_dir,
            observed_dir=args.observed_dir,
            target_path=args.target_dir / f"{cell_type}_target.npz",
            n_sample=args.n_sample,
            seed=args.seed,
        )
        metric_rows.append(metrics)
        profile_rows.extend(profiles)
        provenance_records.append(provenance)
        print(
            f"{cell_type}: rho={metrics['rho_distance']:.4f} "
            f"PCC49={metrics['pcc_49x49']:.4f} "
            f"PCC8={metrics['pcc_8x8']:.4f} "
            f"gate_pass={metrics['gate_pass']}"
        )
    write_evaluation_outputs(
        metric_rows=metric_rows,
        profile_rows=profile_rows,
        provenance_records=provenance_records,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
