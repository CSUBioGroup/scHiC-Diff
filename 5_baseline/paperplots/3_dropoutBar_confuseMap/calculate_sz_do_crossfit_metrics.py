#!/usr/bin/env python3
"""Calculate five-fold cell-wise OOF SZ/DO metrics for HiCImputeData."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PLOT_ROOT = Path(__file__).resolve().parent
METRIC_ROOT = PLOT_ROOT.parent / "1_pccAndMae_all"
if str(METRIC_ROOT) not in sys.path:
    sys.path.insert(0, str(METRIC_ROOT))

from imputation_metric_config import (  # noqa: E402
    METHOD_ORDER,
    assert_expected_shape,
    dataset_descriptors,
    load_sparse_triangle_features,
    method_config,
    resolve_imputed_path,
)


OUTPUT_ROOT = PLOT_ROOT / "1_HiCImputedData"
METRICS_OUTPUT = OUTPUT_ROOT / "HiCImputeData_SZ_DO_5fold_OOF_metrics.tsv"
THRESHOLDS_OUTPUT = OUTPUT_ROOT / "HiCImputeData_SZ_DO_5fold_thresholds.tsv"
FOLD_METRICS_OUTPUT = OUTPUT_ROOT / "HiCImputeData_SZ_DO_5fold_fold_metrics.tsv"
CELL_FOLDS_OUTPUT = OUTPUT_ROOT / "HiCImputeData_SZ_DO_cell_folds.tsv"
CELL_TYPES = ("T1", "T2", "T3")
DEPTHS = ("1k", "2k", "4k", "7k")
N_FOLDS = 5
FOLD_SEED = 20260713


def safe_div(numerator: float, denominator: float) -> float:
    return float("nan") if denominator == 0 else numerator / denominator


def summarize_counts(counts: dict[str, int]) -> dict[str, float]:
    tp, fn, fp, tn = (counts[key] for key in ("TP", "FN", "FP", "TN"))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = (
        float("nan")
        if not np.isfinite(precision + recall) or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    mcc_denominator = np.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    )
    mcc = (
        float("nan")
        if mcc_denominator == 0
        else (tp * tn - fp * fn) / mcc_denominator
    )
    balanced_accuracy = (
        float("nan")
        if not np.isfinite(recall + specificity)
        else (recall + specificity) / 2
    )
    return {
        "accuracy": safe_div(tp + tn, tp + fn + fp + tn),
        "precision_sz": precision,
        "recall_sz": recall,
        "specificity_do": specificity,
        "f1_sz": f1,
        "mcc": mcc,
        "balanced_accuracy": balanced_accuracy,
        "true_sz_pred_sz": recall,
        "true_sz_pred_do": safe_div(fn, tp + fn),
        "true_do_pred_sz": safe_div(fp, fp + tn),
        "true_do_pred_do": specificity,
    }


def candidate_vectors(
    gt: np.ndarray,
    observed: np.ndarray,
    prediction: np.ndarray,
    cell_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gt_subset = gt[cell_mask]
    observed_subset = observed[cell_mask]
    prediction_subset = prediction[cell_mask]
    candidate = observed_subset == 0
    values = prediction_subset[candidate].astype(np.float64, copy=False)
    labels = (gt_subset[candidate] == 0).astype(np.int8, copy=False)
    if values.size == 0:
        raise ValueError("No observed-zero candidates in selected cells")
    if not np.isfinite(values).all():
        raise ValueError("Prediction contains non-finite observed-zero values")
    if np.unique(labels).size != 2:
        raise ValueError("Threshold calibration requires both SZ and DO labels")
    return values, labels


def counts_from_vectors(
    values: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    predicted_sz = values < threshold
    true_sz = labels == 1
    true_do = ~true_sz
    counts = {
        "TP": int(np.sum(predicted_sz & true_sz)),
        "FN": int(np.sum(~predicted_sz & true_sz)),
        "FP": int(np.sum(predicted_sz & true_do)),
        "TN": int(np.sum(~predicted_sz & true_do)),
    }
    counts["candidate_count"] = int(values.size)
    counts["true_sz_count"] = int(true_sz.sum())
    counts["true_do_count"] = int(true_do.sum())
    counts["predicted_sz_count"] = int(predicted_sz.sum())
    counts["predicted_do_count"] = int((~predicted_sz).sum())
    if sum(counts[key] for key in ("TP", "FN", "FP", "TN")) != values.size:
        raise AssertionError("Confusion counts do not cover all candidate values")
    return counts


def select_threshold_max_mcc(
    values: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, dict[str, int], dict[str, float]]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_labels = labels[order].astype(np.int64, copy=False)
    group_ends = np.flatnonzero(
        np.r_[sorted_values[1:] != sorted_values[:-1], True]
    )

    cumulative_tp = np.cumsum(sorted_labels, dtype=np.int64)[group_ends]
    predicted_positive = group_ends.astype(np.int64) + 1
    cumulative_fp = predicted_positive - cumulative_tp
    total_positive = int(sorted_labels.sum())
    total_negative = int(sorted_labels.size - total_positive)

    tp = cumulative_tp.astype(np.float64)
    fp = cumulative_fp.astype(np.float64)
    fn = total_positive - tp
    tn = total_negative - fp
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.full(tp.shape, np.nan, dtype=np.float64)
    valid = denominator > 0
    mcc[valid] = (tp[valid] * tn[valid] - fp[valid] * fn[valid]) / denominator[valid]
    if not np.isfinite(mcc).any():
        raise ValueError("No non-degenerate MCC threshold candidate")

    recall = tp / total_positive
    specificity = tn / total_negative
    balanced_accuracy = (recall + specificity) / 2
    max_mcc = float(np.nanmax(mcc))
    tied = np.flatnonzero(np.isclose(mcc, max_mcc, rtol=0, atol=1e-12))
    max_balanced = float(np.max(balanced_accuracy[tied]))
    tied = tied[
        np.isclose(balanced_accuracy[tied], max_balanced, rtol=0, atol=1e-12)
    ]
    chosen = int(tied[0])
    boundary_value = float(sorted_values[group_ends[chosen]])
    threshold = float(np.nextafter(boundary_value, np.inf))
    counts = counts_from_vectors(values, labels, threshold)
    metrics = summarize_counts(counts)
    if not np.isclose(metrics["mcc"], max_mcc, rtol=0, atol=1e-10):
        raise AssertionError(
            f"Selected threshold MCC mismatch: {metrics['mcc']} vs {max_mcc}"
        )
    return threshold, counts, metrics


def load_prediction(descriptor: dict, method: str) -> tuple[np.ndarray, Path]:
    config = method_config(method, "HiCImputeData")
    if config["loader_kind"] != "sparse_triangle":
        raise ValueError(f"Unsupported loader for {method}: {config['loader_kind']}")
    path = resolve_imputed_path("HiCImputeData", descriptor["name"], method)
    prediction = load_sparse_triangle_features(
        path,
        n_beads=descriptor["n_beads"],
        feature_order=config.get("feature_order", "tril"),
        target_order="tril",
    )
    assert_expected_shape(
        prediction, descriptor["expected_shape"], f"prediction {method}"
    )
    return prediction, path


def build_cell_folds(
    n_cells_by_type: dict[str, int],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    assignments = {}
    rows = []
    for ctype_index, ctype in enumerate(CELL_TYPES):
        n_cells = n_cells_by_type[ctype]
        rng = np.random.default_rng(FOLD_SEED + 1009 * ctype_index)
        permutation = rng.permutation(n_cells)
        fold_ids = np.empty(n_cells, dtype=np.int8)
        for fold, indices in enumerate(np.array_split(permutation, N_FOLDS)):
            fold_ids[indices] = fold
        assignments[ctype] = fold_ids
        for cell_index, fold in enumerate(fold_ids):
            rows.append(
                {
                    "ctype": ctype,
                    "cell_index_zero_based": cell_index,
                    "cell_index_one_based": cell_index + 1,
                    "fold": int(fold),
                    "seed": FOLD_SEED,
                }
            )
        fold_sizes = np.bincount(fold_ids, minlength=N_FOLDS)
        if fold_sizes.max() - fold_sizes.min() > 1:
            raise AssertionError(f"Unbalanced folds for {ctype}: {fold_sizes}")
    return assignments, pd.DataFrame(rows)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)
    print(f"Wrote {len(frame)} rows to {path}")


def main() -> None:
    descriptors = {}
    base_data = {}
    n_cells_by_type = {}
    for descriptor in dataset_descriptors("HiCImputeData"):
        dataset = descriptor["name"]
        _, ctype, depth = dataset.split("_")
        gt = load_sparse_triangle_features(descriptor["gt_path"])
        observed = load_sparse_triangle_features(descriptor["observed_path"])
        assert_expected_shape(gt, descriptor["expected_shape"], f"GT {dataset}")
        assert_expected_shape(
            observed, descriptor["expected_shape"], f"observed {dataset}"
        )
        descriptors[(ctype, depth)] = descriptor
        base_data[(ctype, depth)] = (gt, observed)
        previous = n_cells_by_type.setdefault(ctype, gt.shape[0])
        if previous != gt.shape[0]:
            raise ValueError(f"Inconsistent cell count for {ctype}")

    fold_assignments, cell_folds = build_cell_folds(n_cells_by_type)
    threshold_rows = []
    fold_metric_rows = []
    aggregate_counts: dict[tuple[str, str, str], defaultdict[str, int]] = {}
    threshold_lookup: dict[tuple[str, str], list[float]] = defaultdict(list)
    prediction_paths = {}

    for method in METHOD_ORDER:
        for depth in DEPTHS:
            method_data = {}
            for ctype in CELL_TYPES:
                descriptor = descriptors[(ctype, depth)]
                gt, observed = base_data[(ctype, depth)]
                prediction, path = load_prediction(descriptor, method)
                method_data[ctype] = (gt, observed, prediction)
                prediction_paths[(method, ctype, depth)] = path

            for fold in range(N_FOLDS):
                calibration_values = []
                calibration_labels = []
                calibration_cells = 0
                for ctype in CELL_TYPES:
                    gt, observed, prediction = method_data[ctype]
                    calibration_mask = fold_assignments[ctype] != fold
                    test_mask = fold_assignments[ctype] == fold
                    if np.any(calibration_mask & test_mask):
                        raise AssertionError("Calibration/test cell overlap")
                    calibration_cells += int(calibration_mask.sum())
                    values, labels = candidate_vectors(
                        gt, observed, prediction, calibration_mask
                    )
                    calibration_values.append(values)
                    calibration_labels.append(labels)

                calibration_values_array = np.concatenate(calibration_values)
                calibration_labels_array = np.concatenate(calibration_labels)
                threshold, calibration_counts, calibration_metrics = (
                    select_threshold_max_mcc(
                        calibration_values_array, calibration_labels_array
                    )
                )
                threshold_lookup[(method, depth)].append(threshold)
                threshold_rows.append(
                    {
                        "method": method,
                        "cdepth": depth,
                        "fold": fold,
                        "threshold": threshold,
                        "selection_metric": "MCC",
                        "selection_scope": "calibration_cells_observed_zero",
                        "shared_cell_types": "T1,T2,T3",
                        "calibration_cell_count": calibration_cells,
                        **{
                            f"calibration_{key}": value
                            for key, value in calibration_counts.items()
                        },
                        **{
                            f"calibration_{key}": value
                            for key, value in calibration_metrics.items()
                        },
                    }
                )

                for ctype in CELL_TYPES:
                    gt, observed, prediction = method_data[ctype]
                    test_mask = fold_assignments[ctype] == fold
                    values, labels = candidate_vectors(
                        gt, observed, prediction, test_mask
                    )
                    counts = counts_from_vectors(values, labels, threshold)
                    metrics = summarize_counts(counts)
                    fold_metric_rows.append(
                        {
                            "method": method,
                            "ctype": ctype,
                            "cdepth": depth,
                            "data_name": f"K562_{ctype}_{depth}",
                            "fold": fold,
                            "threshold": threshold,
                            "test_cell_count": int(test_mask.sum()),
                            "evaluation_scope": "observed_zero",
                            **counts,
                            **metrics,
                        }
                    )
                    key = (method, ctype, depth)
                    aggregate = aggregate_counts.setdefault(key, defaultdict(int))
                    for count_name in (
                        "TP",
                        "FN",
                        "FP",
                        "TN",
                        "candidate_count",
                        "true_sz_count",
                        "true_do_count",
                        "predicted_sz_count",
                        "predicted_do_count",
                    ):
                        aggregate[count_name] += counts[count_name]
            print(f"Completed {method} depth={depth}", flush=True)

    metric_rows = []
    for method in METHOD_ORDER:
        for ctype in CELL_TYPES:
            for depth in DEPTHS:
                key = (method, ctype, depth)
                counts = dict(aggregate_counts[key])
                expected_candidates = int(
                    np.sum(base_data[(ctype, depth)][1] == 0)
                )
                if counts["candidate_count"] != expected_candidates:
                    raise AssertionError(
                        f"OOF coverage mismatch for {key}: "
                        f"{counts['candidate_count']} vs {expected_candidates}"
                    )
                thresholds = np.asarray(
                    threshold_lookup[(method, depth)], dtype=np.float64
                )
                if thresholds.size != N_FOLDS:
                    raise AssertionError(f"Expected {N_FOLDS} thresholds for {key}")
                metric_rows.append(
                    {
                        "method": method,
                        "data_name": f"K562_{ctype}_{depth}",
                        "ctype": ctype,
                        "cdepth": depth,
                        "positive_class": "SZ",
                        "evaluation_scope": "observed_zero",
                        "threshold_mode": "5fold_cellwise_crossfit_method_depth",
                        "threshold_selection": "max_MCC",
                        "crossfit_folds": N_FOLDS,
                        "fold_seed": FOLD_SEED,
                        "threshold_mean": float(thresholds.mean()),
                        "threshold_sd": float(thresholds.std()),
                        "threshold_min": float(thresholds.min()),
                        "threshold_max": float(thresholds.max()),
                        **counts,
                        **summarize_counts(counts),
                        "prediction_path": str(prediction_paths[key]),
                    }
                )

    metrics = pd.DataFrame(metric_rows)
    thresholds = pd.DataFrame(threshold_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    if len(metrics) != len(METHOD_ORDER) * len(CELL_TYPES) * len(DEPTHS):
        raise AssertionError(f"Unexpected metric row count: {len(metrics)}")
    if len(thresholds) != len(METHOD_ORDER) * len(DEPTHS) * N_FOLDS:
        raise AssertionError(f"Unexpected threshold row count: {len(thresholds)}")
    expected_fold_rows = len(METHOD_ORDER) * len(CELL_TYPES) * len(DEPTHS) * N_FOLDS
    if len(fold_metrics) != expected_fold_rows:
        raise AssertionError(f"Unexpected fold metric row count: {len(fold_metrics)}")

    write_table(cell_folds, CELL_FOLDS_OUTPUT)
    write_table(thresholds, THRESHOLDS_OUTPUT)
    write_table(fold_metrics, FOLD_METRICS_OUTPUT)
    write_table(metrics, METRICS_OUTPUT)


if __name__ == "__main__":
    main()
