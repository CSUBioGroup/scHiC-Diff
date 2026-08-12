#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz


QUANTILES = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)


def quantile_summary(values, max_points=1_000_000):
    values = np.asarray(values).reshape(-1)
    if values.size == 0:
        return {"count": 0, "quantiles": {}}
    sampled = values
    if values.size > max_points:
        rng = np.random.default_rng(10)
        sampled = values[rng.integers(0, values.size, size=max_points)]
    quantiles = np.quantile(sampled.astype(np.float64, copy=False), QUANTILES)
    return {
        "count": int(values.size),
        "sampled_count": int(sampled.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles": {
            str(q): float(value) for q, value in zip(QUANTILES, quantiles)
        },
    }


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def sparse_stats(matrix):
    matrix = matrix.tocsr()
    rows, cols = matrix.shape
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    row_nnz = np.diff(matrix.indptr)
    col_sums = np.asarray(matrix.sum(axis=0)).ravel()
    total = rows * cols
    return {
        "shape": [int(rows), int(cols)],
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / total),
        "zero_fraction": float(1.0 - matrix.nnz / total),
        "sum": float(matrix.sum()),
        "mean_including_zeros": float(matrix.sum() / total),
        "stored_values": quantile_summary(matrix.data),
        "row_sums": quantile_summary(row_sums),
        "row_nnz": quantile_summary(row_nnz),
        "column_sums": quantile_summary(col_sums),
    }


def metrics_against_raw(prediction, raw):
    prediction = prediction.tocsr().astype(np.float64)
    raw = raw.tocsr().astype(np.float64)
    if prediction.shape != raw.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} vs {raw.shape}")

    rows, cols = prediction.shape
    n = rows * cols
    sum_x = float(prediction.sum())
    sum_y = float(raw.sum())
    sum_x2 = float(prediction.multiply(prediction).sum())
    sum_y2 = float(raw.multiply(raw).sum())
    sum_xy = float(prediction.multiply(raw).sum())
    covariance = sum_xy - sum_x * sum_y / n
    variance_x = sum_x2 - sum_x * sum_x / n
    variance_y = sum_y2 - sum_y * sum_y / n
    denominator = np.sqrt(max(variance_x, 0.0) * max(variance_y, 0.0))
    global_pearson = float(covariance / denominator) if denominator > 0 else None
    squared_error = max(sum_x2 + sum_y2 - 2.0 * sum_xy, 0.0)

    row_x = np.asarray(prediction.sum(axis=1)).ravel()
    row_y = np.asarray(raw.sum(axis=1)).ravel()
    row_x2 = np.asarray(prediction.multiply(prediction).sum(axis=1)).ravel()
    row_y2 = np.asarray(raw.multiply(raw).sum(axis=1)).ravel()
    row_xy = np.asarray(prediction.multiply(raw).sum(axis=1)).ravel()
    row_covariance = row_xy - row_x * row_y / cols
    row_variance_x = row_x2 - row_x * row_x / cols
    row_variance_y = row_y2 - row_y * row_y / cols
    row_denominator = np.sqrt(
        np.maximum(row_variance_x, 0.0) * np.maximum(row_variance_y, 0.0)
    )
    valid = row_denominator > 0
    row_correlations = row_covariance[valid] / row_denominator[valid]

    raw_coo = raw.tocoo()
    prediction_at_observed = prediction[raw_coo.row, raw_coo.col].A1
    observed_mass = float(prediction_at_observed.sum())
    total_mass = float(prediction.sum())
    return {
        "global_pearson_including_zeros": global_pearson,
        "global_rmse_including_zeros": float(np.sqrt(squared_error / n)),
        "row_pearson_including_zeros": quantile_summary(row_correlations),
        "row_sum_pearson": safe_corr(row_x, row_y),
        "prediction_at_raw_nonzero": quantile_summary(prediction_at_observed),
        "raw_nonzero_values": quantile_summary(raw_coo.data),
        "pearson_at_raw_nonzero": safe_corr(prediction_at_observed, raw_coo.data),
        "prediction_mass_at_raw_nonzero_fraction": (
            float(observed_mass / total_mass) if total_mass > 0 else None
        ),
    }


def inverse_transform_check(reconstruction, inverse):
    expected = reconstruction.tocsr().astype(np.float64).copy()
    expected.data = np.expm1(expected.data)
    difference = expected - inverse.tocsr().astype(np.float64)
    if difference.nnz == 0:
        max_abs = 0.0
    else:
        max_abs = float(np.max(np.abs(difference.data)))
    return {
        "difference_nnz": int(difference.nnz),
        "max_abs_difference": max_abs,
    }


def normalized_target_depth(target):
    counts = target.tocsr().astype(np.float64).copy()
    counts.data = np.expm1(counts.data)
    return quantile_summary(np.asarray(counts.sum(axis=1)).ravel())


def analyze_case(result_dir):
    matrices = {
        name: load_npz(result_dir / f"{name}.npz").tocsr()
        for name in ("denoise_recon", "denoise_recon_inv", "denoise_target", "raw_x")
    }
    return {
        "result_dir": str(result_dir),
        "matrices": {name: sparse_stats(matrix) for name, matrix in matrices.items()},
        "prediction_vs_raw": metrics_against_raw(
            matrices["denoise_recon_inv"], matrices["raw_x"]
        ),
        "inverse_transform": inverse_transform_check(
            matrices["denoise_recon"], matrices["denoise_recon_inv"]
        ),
        "target_depth_after_expm1": normalized_target_depth(
            matrices["denoise_target"]
        ),
    }


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="ascii")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    case_dirs = {
        "eval_full": args.base / "results/full/eval_v2",
        "eval_smoke": args.base / "results/smoke/eval_v2",
        "ramani_smoke": args.base / "results/smoke/ramani",
    }
    payload = {"cases": {}}
    for name, result_dir in case_dirs.items():
        required = result_dir / "denoise_recon_inv.npz"
        if required.is_file():
            payload["cases"][name] = analyze_case(result_dir)

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write(args.output, text)
    print(text, end="")


if __name__ == "__main__":
    main()
