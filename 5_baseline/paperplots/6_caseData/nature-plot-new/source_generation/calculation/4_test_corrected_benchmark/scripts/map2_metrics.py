"""Distance-normalized Map2 metrics and matched negative controls."""

from __future__ import annotations

import numpy as np


def _validate_square(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")
    return matrix


def apply_transform(matrix: np.ndarray, mode: str) -> np.ndarray:
    matrix = _validate_square(matrix)
    if mode == "none":
        return matrix.copy()
    if mode == "log1p_all":
        if np.any(matrix < 0):
            raise ValueError("log1p_all cannot be applied to negative values")
        return np.log1p(matrix)
    raise ValueError(f"unknown transform mode: {mode}")


def expected_by_distance(matrix: np.ndarray) -> np.ndarray:
    matrix = _validate_square(matrix)
    expected = np.empty(matrix.shape[0], dtype=float)
    for distance in range(matrix.shape[0]):
        values = np.diagonal(matrix, offset=distance)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"distance {distance} has no finite values")
        expected[distance] = float(finite.mean())
    return expected


def observed_over_expected(
    matrix: np.ndarray,
    expected: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    matrix = _validate_square(matrix)
    expected = np.asarray(expected, dtype=float)
    if expected.shape != (matrix.shape[0],):
        raise ValueError(f"expected must have shape {(matrix.shape[0],)}, got {expected.shape}")
    if not np.isfinite(expected).all():
        raise ValueError("expected contains non-finite values")
    distances = np.abs(np.subtract.outer(np.arange(matrix.shape[0]), np.arange(matrix.shape[0])))
    denominators = np.maximum(expected[distances], epsilon)
    return matrix / denominators


def extract_corrected_metrics(
    oe_matrix: np.ndarray,
    bin1: int,
    bin2: int,
    center_size: int = 5,
    outer_size: int = 21,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    oe_matrix = _validate_square(oe_matrix)
    if center_size <= 0 or center_size % 2 == 0:
        raise ValueError("center_size must be a positive odd integer")
    if outer_size <= center_size or outer_size % 2 == 0:
        raise ValueError("outer_size must be an odd integer larger than center_size")

    outer_radius = outer_size // 2
    r0, r1 = bin1 - outer_radius, bin1 + outer_radius + 1
    c0, c1 = bin2 - outer_radius, bin2 + outer_radius + 1
    if r0 < 0 or c0 < 0 or r1 > oe_matrix.shape[0] or c1 > oe_matrix.shape[1]:
        raise ValueError(
            f"{outer_size}x{outer_size} neighborhood does not fit around ({bin1}, {bin2})"
        )
    outer = oe_matrix[r0:r1, c0:c1]

    center_index = outer_size // 2
    center_radius = center_size // 2
    center_start = center_index - center_radius
    center_stop = center_index + center_radius + 1
    center = outer[center_start:center_stop, center_start:center_stop]
    lower_left = outer[outer_size - center_size : outer_size, 0:center_size]

    donut_mask = np.ones((outer_size, outer_size), dtype=bool)
    donut_mask[center_start:center_stop, center_start:center_stop] = False
    donut_mask[center_index, :] = False
    donut_mask[:, center_index] = False
    donut = outer[donut_mask]

    center_signal = float(center.mean())
    lower_left_signal = float(lower_left.mean())
    donut_signal = float(donut.mean())

    lower_denominator = max(lower_left_signal, epsilon)
    donut_denominator = max(donut_signal, epsilon)
    return {
        "center_signal": center_signal,
        "lower_left_signal": lower_left_signal,
        "donut_signal": donut_signal,
        "lower_left_ratio": center_signal / lower_denominator,
        "donut_ratio": center_signal / donut_denominator,
        "lower_left_log2_enrichment": float(
            np.log2((center_signal + epsilon) / lower_denominator)
        ),
        "donut_log2_enrichment": float(
            np.log2((center_signal + epsilon) / donut_denominator)
        ),
    }


def generate_distance_matched_controls(
    n_bins: int,
    bin1: int,
    bin2: int,
    exclusion_bins: int = 2,
    limit: int = 100,
    seed: int = 42,
    outer_radius: int = 10,
) -> list[tuple[int, int]]:
    if not (0 <= bin1 < n_bins and 0 <= bin2 < n_bins):
        raise ValueError("positive loop bins are outside the matrix")
    separation = abs(bin2 - bin1)
    if separation == 0:
        raise ValueError("positive loop anchors must have non-zero separation")
    first_positive, second_positive = sorted((bin1, bin2))
    eligible = []
    if outer_radius < 0:
        raise ValueError("outer_radius must be non-negative")
    for first in range(outer_radius, n_bins - separation - outer_radius):
        second = first + separation
        if (first, second) == (first_positive, second_positive):
            continue
        if abs(first - first_positive) <= exclusion_bins:
            continue
        if abs(second - second_positive) <= exclusion_bins:
            continue
        eligible.append((first, second))
    if len(eligible) <= limit:
        return eligible
    rng = np.random.RandomState(seed)
    chosen = np.sort(rng.choice(len(eligible), size=limit, replace=False))
    return [eligible[int(idx)] for idx in chosen]


def empirical_control_stats(
    positive: float,
    controls: np.ndarray,
    epsilon: float = 1e-12,
) -> dict[str, float | int]:
    controls = np.asarray(controls, dtype=float)
    if controls.ndim != 1 or controls.size == 0:
        raise ValueError("controls must be a non-empty one-dimensional array")
    if not np.isfinite(controls).all() or not np.isfinite(positive):
        raise ValueError("positive and control values must be finite")
    median = float(np.median(controls))
    mad = float(np.median(np.abs(controls - median)))
    return {
        "control_count": int(controls.size),
        "control_median": median,
        "control_mad": mad,
        "empirical_p_upper": float((1 + np.count_nonzero(controls >= positive)) / (1 + controls.size)),
        "percentile": float(100.0 * np.count_nonzero(controls < positive) / controls.size),
        "robust_effect_mad": float((positive - median) / max(mad, epsilon)),
    }
