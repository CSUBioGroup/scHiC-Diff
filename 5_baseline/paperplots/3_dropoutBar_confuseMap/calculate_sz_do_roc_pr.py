#!/usr/bin/env python3
"""Compute full OOF ROC/PR threshold scans for HiCImputeData SZ/DO labels.

The imputation methods emit continuous contact values.  For this diagnostic,
smaller values are treated as stronger evidence for a structural zero (SZ), so
the ranking score is ``-imputed_contact``.  ROC and PR quantities are computed
at every distinct imputed value on observed-zero contacts from the five
cell-wise out-of-fold partitions.  The summary reports ROC-AUC, linearly
interpolated trapezoidal PR-AUC, and average precision (AP) separately.  The
existing cross-fitted MCC operating point is read from the authoritative OOF
metric table; it is not used to calculate any curve summary.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import sys
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
OOF_METRICS_INPUT = OUTPUT_ROOT / "HiCImputeData_SZ_DO_5fold_OOF_metrics.tsv"
FULL_SCAN_OUTPUT = OUTPUT_ROOT / "HiCImputeData_SZ_DO_5fold_OOF_threshold_scan.tsv.gz"
PLOT_POINTS_OUTPUT = OUTPUT_ROOT / "HiCImputeData_SZ_DO_5fold_OOF_curve_plot_points.tsv"
SUMMARY_OUTPUT = OUTPUT_ROOT / "HiCImputeData_SZ_DO_5fold_OOF_ROC_PR_AUC.tsv"

CELL_TYPES = ("T1", "T2", "T3")
DEPTHS = ("1k", "2k", "4k", "7k")
N_FOLDS = 5
FOLD_SEED = 20260713
DEFAULT_MAX_PLOT_POINTS = 2500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--oof-metrics", type=Path, default=OOF_METRICS_INPUT)
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=DEFAULT_MAX_PLOT_POINTS,
        help="Maximum rendering points per curve; the compressed scan remains exact.",
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


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
        raise ValueError("Each ROC/PR condition requires both SZ and DO labels")
    return values, labels


def build_cell_folds(n_cells_by_type: dict[str, int]) -> dict[str, np.ndarray]:
    assignments: dict[str, np.ndarray] = {}
    for ctype_index, ctype in enumerate(CELL_TYPES):
        n_cells = n_cells_by_type[ctype]
        rng = np.random.default_rng(FOLD_SEED + 1009 * ctype_index)
        permutation = rng.permutation(n_cells)
        fold_ids = np.empty(n_cells, dtype=np.int8)
        for fold, indices in enumerate(np.array_split(permutation, N_FOLDS)):
            fold_ids[indices] = fold
        fold_sizes = np.bincount(fold_ids, minlength=N_FOLDS)
        if fold_sizes.max() - fold_sizes.min() > 1:
            raise AssertionError(f"Unbalanced folds for {ctype}: {fold_sizes}")
        assignments[ctype] = fold_ids
    return assignments


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


def collect_oof_vectors(
    gt: np.ndarray,
    observed: np.ndarray,
    prediction: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values_by_fold = []
    labels_by_fold = []
    for fold in range(N_FOLDS):
        test_mask = fold_ids == fold
        if not test_mask.any():
            raise AssertionError(f"Fold {fold} has no held-out cells")
        values, labels = candidate_vectors(gt, observed, prediction, test_mask)
        values_by_fold.append(values)
        labels_by_fold.append(labels)

    values = np.concatenate(values_by_fold)
    labels = np.concatenate(labels_by_fold)
    if values.size != int(np.sum(observed == 0)):
        raise AssertionError("OOF score collection does not cover all observed-zero contacts")
    if np.unique(labels).size != 2:
        raise AssertionError("OOF score collection lost a class")
    return values, labels


def exact_threshold_scan(values: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray | float | int]:
    """Return all threshold operating points for ``prediction < threshold``."""
    if values.ndim != 1 or labels.ndim != 1 or values.size != labels.size:
        raise ValueError("values and labels must be equally sized one-dimensional arrays")
    if values.size == 0 or np.unique(labels).size != 2:
        raise ValueError("ROC/PR requires non-empty binary labels")

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_labels = labels[order].astype(np.int64, copy=False)
    group_ends = np.flatnonzero(
        np.r_[sorted_values[1:] != sorted_values[:-1], True]
    )

    total_positive = int(sorted_labels.sum())
    total_negative = int(sorted_labels.size - total_positive)
    cumulative_tp = np.cumsum(sorted_labels, dtype=np.int64)[group_ends]
    predicted_positive = group_ends.astype(np.int64) + 1
    cumulative_fp = predicted_positive - cumulative_tp

    recall = cumulative_tp.astype(np.float64) / total_positive
    fpr = cumulative_fp.astype(np.float64) / total_negative
    precision = cumulative_tp.astype(np.float64) / predicted_positive
    threshold = np.nextafter(sorted_values[group_ends], np.inf)

    # The initial point corresponds to no position being called structural zero.
    recall = np.r_[0.0, recall]
    fpr = np.r_[0.0, fpr]
    precision = np.r_[1.0, precision]
    threshold = np.r_[-np.inf, threshold]

    if not (np.all(np.diff(recall) >= 0) and np.all(np.diff(fpr) >= 0)):
        raise AssertionError("Threshold scan is not monotonic")
    if not (np.isclose(recall[-1], 1.0) and np.isclose(fpr[-1], 1.0)):
        raise AssertionError("Threshold scan does not terminate at all-positive calls")

    roc_auc = float(np.trapz(recall, fpr))
    pr_auc_linear_interpolation = float(np.trapz(precision, recall))
    average_precision = float(np.sum(np.diff(recall) * precision[1:]))
    if not (
        0.0 <= roc_auc <= 1.0
        and 0.0 <= pr_auc_linear_interpolation <= 1.0
        and 0.0 <= average_precision <= 1.0
    ):
        raise AssertionError("AUC is outside [0, 1]")

    return {
        "threshold": threshold,
        "tpr_sz": recall,
        "fpr_do": fpr,
        "precision_sz": precision,
        "recall_sz": recall,
        "roc_auc": roc_auc,
        "pr_auc_linear_interpolation": pr_auc_linear_interpolation,
        "average_precision": average_precision,
        "n_unique_scores": int(group_ends.size),
        "candidate_count": int(values.size),
        "true_sz_count": total_positive,
        "true_do_count": total_negative,
    }


def plot_sample_indices(scan: dict[str, np.ndarray | float | int], maximum: int) -> np.ndarray:
    if maximum < 5:
        raise ValueError("max_plot_points must be at least 5")
    n_points = len(scan["threshold"])
    if n_points <= maximum:
        return np.arange(n_points, dtype=np.int64)

    points_per_strategy = max(1, (maximum - 2) // 3)
    index_points = np.linspace(0, n_points - 1, points_per_strategy, dtype=np.int64)
    fpr_points = np.searchsorted(
        scan["fpr_do"], np.linspace(0.0, 1.0, points_per_strategy), side="left"
    )
    recall_points = np.searchsorted(
        scan["recall_sz"], np.linspace(0.0, 1.0, points_per_strategy), side="left"
    )
    return np.unique(
        np.clip(
            np.r_[0, n_points - 1, index_points, fpr_points, recall_points],
            0,
            n_points - 1,
        )
    )


def load_oof_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t")
    required = {
        "method",
        "data_name",
        "ctype",
        "cdepth",
        "positive_class",
        "evaluation_scope",
        "threshold_mode",
        "threshold_selection",
        "crossfit_folds",
        "threshold_mean",
        "threshold_sd",
        "threshold_min",
        "threshold_max",
        "TP",
        "FN",
        "FP",
        "TN",
        "precision_sz",
        "recall_sz",
        "specificity_do",
        "f1_sz",
        "mcc",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing OOF metric columns: {missing}")
    expected = {
        (method, f"K562_{ctype}_{depth}")
        for method in METHOD_ORDER
        for ctype in CELL_TYPES
        for depth in DEPTHS
    }
    keys = list(zip(frame["method"], frame["data_name"]))
    if pd.Series(keys).duplicated().any() or set(keys) != expected:
        raise ValueError("Expected exactly 7 methods x 12 HiCImputeData OOF rows")
    if set(frame["positive_class"]) != {"SZ"}:
        raise ValueError("Expected SZ as the positive class")
    if set(frame["evaluation_scope"]) != {"observed_zero"}:
        raise ValueError("Expected observed-zero evaluation")
    if set(frame["threshold_mode"]) != {"5fold_cellwise_crossfit_method_depth"}:
        raise ValueError("Unexpected threshold mode")
    if set(frame["threshold_selection"]) != {"max_MCC"}:
        raise ValueError("Expected max-MCC threshold selection")
    if set(frame["crossfit_folds"]) != {N_FOLDS}:
        raise ValueError("Expected five OOF folds")
    return frame.set_index(["method", "data_name"], verify_integrity=True)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)
    print(f"Wrote {len(frame)} rows to {path}")


def run_self_tests() -> None:
    perfect = exact_threshold_scan(
        np.asarray([0.1, 0.2, 0.8, 0.9]), np.asarray([1, 1, 0, 0])
    )
    if not (
        np.isclose(perfect["roc_auc"], 1.0)
        and np.isclose(perfect["pr_auc_linear_interpolation"], 1.0)
        and np.isclose(perfect["average_precision"], 1.0)
    ):
        raise AssertionError("Perfect ranking self-test failed")

    tied = exact_threshold_scan(
        np.asarray([0.5, 0.5, 0.5, 0.5]), np.asarray([1, 0, 1, 0])
    )
    if not (
        np.isclose(tied["roc_auc"], 0.5)
        and np.isclose(tied["pr_auc_linear_interpolation"], 0.75)
        and np.isclose(tied["average_precision"], 0.5)
    ):
        raise AssertionError("Tied-score self-test failed")
    print("ROC/PR self-tests passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_tests()
        return
    if args.max_plot_points < 5:
        raise ValueError("--max-plot-points must be at least 5")

    metrics = load_oof_metrics(args.oof_metrics)
    print(f"Validated {len(metrics)} OOF metric rows from {args.oof_metrics}")
    if args.validate_only:
        return

    output_root = args.output_root
    full_scan_output = output_root / FULL_SCAN_OUTPUT.name
    plot_points_output = output_root / PLOT_POINTS_OUTPUT.name
    summary_output = output_root / SUMMARY_OUTPUT.name
    output_root.mkdir(parents=True, exist_ok=True)

    descriptors: dict[tuple[str, str], dict] = {}
    base_data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    n_cells_by_type: dict[str, int] = {}
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

    fold_assignments = build_cell_folds(n_cells_by_type)
    summary_rows = []
    plot_point_rows = []
    temporary_scan = full_scan_output.with_suffix(full_scan_output.suffix + ".tmp")
    scan_header = [
        "method",
        "ctype",
        "cdepth",
        "data_name",
        "point_index",
        "raw_threshold",
        "prediction_rule",
        "tpr_sz",
        "fpr_do",
        "precision_sz",
        "recall_sz",
        "candidate_count",
        "true_sz_count",
        "true_do_count",
    ]

    with gzip.open(temporary_scan, "wt", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(scan_header)
        for method in METHOD_ORDER:
            for ctype in CELL_TYPES:
                for depth in DEPTHS:
                    descriptor = descriptors[(ctype, depth)]
                    gt, observed = base_data[(ctype, depth)]
                    prediction, prediction_path = load_prediction(descriptor, method)
                    values, labels = collect_oof_vectors(
                        gt, observed, prediction, fold_assignments[ctype]
                    )
                    scan = exact_threshold_scan(values, labels)
                    data_name = f"K562_{ctype}_{depth}"
                    oof = metrics.loc[(method, data_name)]

                    writer.writerows(
                        (
                            method,
                            ctype,
                            depth,
                            data_name,
                            point_index,
                            float(scan["threshold"][point_index]),
                            "imputed_contact < threshold => predicted_SZ",
                            float(scan["tpr_sz"][point_index]),
                            float(scan["fpr_do"][point_index]),
                            float(scan["precision_sz"][point_index]),
                            float(scan["recall_sz"][point_index]),
                            int(scan["candidate_count"]),
                            int(scan["true_sz_count"]),
                            int(scan["true_do_count"]),
                        )
                        for point_index in range(len(scan["threshold"]))
                    )

                    for point_index in plot_sample_indices(scan, args.max_plot_points):
                        plot_point_rows.append(
                            {
                                "method": method,
                                "ctype": ctype,
                                "cdepth": depth,
                                "data_name": data_name,
                                "point_index": int(point_index),
                                "raw_threshold": float(scan["threshold"][point_index]),
                                "tpr_sz": float(scan["tpr_sz"][point_index]),
                                "fpr_do": float(scan["fpr_do"][point_index]),
                                "precision_sz": float(scan["precision_sz"][point_index]),
                                "recall_sz": float(scan["recall_sz"][point_index]),
                                "full_scan_point_count": len(scan["threshold"]),
                            }
                        )

                    summary_rows.append(
                        {
                            "method": method,
                            "ctype": ctype,
                            "cdepth": depth,
                            "data_name": data_name,
                            "positive_class": "SZ",
                            "evaluation_scope": "observed_zero",
                            "score_orientation": "negative_imputed_contact",
                            "prediction_rule": "imputed_contact < threshold => predicted_SZ",
                            "curve_protocol": "5fold_cellwise_OOF_all_unique_prediction_thresholds",
                            "full_scan_path": str(full_scan_output),
                            "candidate_count": int(scan["candidate_count"]),
                            "true_sz_count": int(scan["true_sz_count"]),
                            "true_do_count": int(scan["true_do_count"]),
                            "prediction_min": float(values.min()),
                            "prediction_median": float(np.median(values)),
                            "prediction_max": float(values.max()),
                            "n_unique_scores": int(scan["n_unique_scores"]),
                            "full_scan_point_count": len(scan["threshold"]),
                            "roc_auc": float(scan["roc_auc"]),
                            "pr_auc_linear_interpolation": float(
                                scan["pr_auc_linear_interpolation"]
                            ),
                            "average_precision": float(scan["average_precision"]),
                            "crossfit_threshold_selection": "max_MCC_on_other_four_folds",
                            "crossfit_threshold_mean": float(oof["threshold_mean"]),
                            "crossfit_threshold_sd": float(oof["threshold_sd"]),
                            "crossfit_threshold_min": float(oof["threshold_min"]),
                            "crossfit_threshold_max": float(oof["threshold_max"]),
                            "crossfit_operating_tpr_sz": float(oof["recall_sz"]),
                            "crossfit_operating_fpr_do": float(
                                1.0 - oof["specificity_do"]
                            ),
                            "crossfit_operating_precision_sz": float(oof["precision_sz"]),
                            "crossfit_operating_f1_sz": float(oof["f1_sz"]),
                            "crossfit_operating_mcc": float(oof["mcc"]),
                            "prediction_path": str(prediction_path),
                        }
                    )
                    print(
                        f"Completed ROC/PR scan for {method} {data_name}: "
                        f"ROC-AUC={scan['roc_auc']:.4f}, "
                        f"PR-AUC={scan['pr_auc_linear_interpolation']:.4f}, "
                        f"AP={scan['average_precision']:.4f}",
                        flush=True,
                    )

    temporary_scan.replace(full_scan_output)
    summary = pd.DataFrame(summary_rows)
    plot_points = pd.DataFrame(plot_point_rows)
    expected_conditions = len(METHOD_ORDER) * len(CELL_TYPES) * len(DEPTHS)
    if len(summary) != expected_conditions:
        raise AssertionError(f"Expected {expected_conditions} AUC rows, got {len(summary)}")
    if plot_points.empty:
        raise AssertionError("No rendering points were produced")
    write_table(summary, summary_output)
    write_table(plot_points, plot_points_output)
    print(f"Wrote full threshold scan to {full_scan_output}")


if __name__ == "__main__":
    main()
