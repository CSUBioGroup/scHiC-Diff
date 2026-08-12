#!/usr/bin/env python3
"""Calculate PCC, MAE, and SCC with all/observed/heldout breakdowns.

The task manifest and all GT, observed, and imputed-data paths are generated
from ``imputation_metric_config.py``. Heatmap path tables are intentionally
separate and are not inputs to metric calculation.

For every (method, dataset) pair it computes per-cell PCC, MAE, and SCC for
three feature subsets: all features, observed positions, and heldout nonzero
GT positions.

Usage
-----
    python calculate_imputation_metrics.py prepare-manifest
    python calculate_imputation_metrics.py run-task --task-id <N>
    python calculate_imputation_metrics.py aggregate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from imputation_metric_config import (
    DATASET_FAMILIES,
    FAMILY_ORDER,
    METHOD_ORDER,
    assert_expected_shape,
    dataset_descriptors,
    load_h5ad_gt_and_observed,
    load_sparse_triangle_features,
    load_tensor_flamingo_triu_features,
    method_config,
    resolve_imputed_path,
)

SCRIPT_DIR = Path(__file__).resolve().parent
FAMILY_DIRS = {
    "HiCImputeData": SCRIPT_DIR / "1_HiCImputedData",
    "FLAMINGOData": SCRIPT_DIR / "2_FLAMINGOData",
}
METRIC_MANIFEST_PATH = SCRIPT_DIR / "imputation_metric_tasks.csv"
FLAMINGO_CSV = FAMILY_DIRS["FLAMINGOData"] / "FLAMINGOData_PCC_MAE_SCC_metrics.csv"
HIC_CSV = FAMILY_DIRS["HiCImputeData"] / "HiCImputeData_PCC_MAE_SCC_metrics.csv"


def ensure_output_dirs() -> None:
    for family_dir in FAMILY_DIRS.values():
        (family_dir / "per_cell_metrics").mkdir(parents=True, exist_ok=True)


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    try:
        rho, _ = spearmanr(a, b)
        return float(rho)
    except Exception:
        return float("nan")


def safe_mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def summarize(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)
    nan_count = int(np.isnan(arr).sum())
    if arr.size == 0 or nan_count == arr.size:
        return float("nan"), float("nan"), nan_count
    return float(np.nanmean(arr)), float(np.nanstd(arr)), nan_count


def parse_hicimpute_name(data_name: str) -> tuple[str, str]:
    _, ctype, cdepth = data_name.split("_")
    return ctype, cdepth


def load_truth_observed(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    family_name = row["dataset_family"]
    expected_shape = (int(row["n_cells"]), int(row["n_features"]))
    if family_name == "FLAMINGOData":
        gt, observed = load_h5ad_gt_and_observed(Path(row["gt_path"]))
    else:
        gt = load_sparse_triangle_features(Path(row["gt_path"]))
        observed = load_sparse_triangle_features(Path(row["observed_path"]))
    assert_expected_shape(gt, expected_shape, f"{family_name} GT {row['dataset']}")
    assert_expected_shape(observed, expected_shape, f"{family_name} observed {row['dataset']}")
    return gt, observed


def load_prediction(row: dict[str, Any]) -> np.ndarray:
    expected_shape = (int(row["n_cells"]), int(row["n_features"]))
    loader_kind = row["loader_kind"]
    feature_order = row.get("feature_order", "tril")
    n_beads = int(row["n_beads"])
    if loader_kind == "tensor_tril_encoded_triu":
        pred = load_tensor_flamingo_triu_features(Path(row["imputed_path"]), n_beads)
    elif loader_kind == "sparse_triangle":
        target_order = "triu" if row["dataset_family"] == "FLAMINGOData" else "tril"
        pred = load_sparse_triangle_features(
            Path(row["imputed_path"]),
            n_beads=n_beads,
            feature_order=feature_order,
            target_order=target_order,
        )
    else:
        raise ValueError(f"Unsupported loader kind: {loader_kind}")
    assert_expected_shape(pred, expected_shape, f"prediction {row['method']} {row['dataset']}")
    return pred


def compute_payload(row: dict[str, Any], gt: np.ndarray, observed: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """Compute per-cell PCC/MAE/SCC for all/obs/held subsets.

    Both data families are evaluated directly on their stored raw values.  In
    particular, this deliberately preserves negative imputed values instead
    of clipping them before PCC, MAE, or SCC calculation.

    Subsets:
      all       = the complete feature vector, including zeros
      observed  = observed > 0
      heldout   = GT > 0 and observed <= 0
    """
    pcc_all: list[float] = []
    pcc_obs: list[float] = []
    pcc_held: list[float] = []
    mae_all: list[float] = []
    mae_obs: list[float] = []
    mae_held: list[float] = []
    scc_all: list[float] = []
    scc_obs: list[float] = []
    scc_held: list[float] = []

    for cell_idx in range(gt.shape[0]):
        gt_raw = np.asarray(gt[cell_idx], dtype=np.float64)
        obs_raw = np.asarray(observed[cell_idx], dtype=np.float64)
        pred_raw = np.asarray(pred[cell_idx], dtype=np.float64)

        gt_use = gt_raw
        pred_use = pred_raw

        gt_mask = gt_raw > 0
        obs_mask = obs_raw > 0
        held_mask = gt_mask & ~obs_mask

        pcc_all.append(safe_pearson(pred_use, gt_use))
        pcc_obs.append(safe_pearson(pred_use[obs_mask], gt_use[obs_mask]))
        pcc_held.append(safe_pearson(pred_use[held_mask], gt_use[held_mask]))

        mae_all.append(safe_mae(pred_use, gt_use))
        mae_obs.append(safe_mae(pred_use[obs_mask], gt_use[obs_mask]))
        mae_held.append(safe_mae(pred_use[held_mask], gt_use[held_mask]))

        scc_all.append(safe_spearman(pred_use, gt_use))
        scc_obs.append(safe_spearman(pred_use[obs_mask], gt_use[obs_mask]))
        scc_held.append(safe_spearman(pred_use[held_mask], gt_use[held_mask]))

    payload: dict[str, Any] = {
        "dataset_family": row["dataset_family"],
        "dataset": row["dataset"],
        "method": row["method"],
        "n_cells": int(row["n_cells"]),
        "n_features": int(row["n_features"]),
        "transform": "raw",
        "pcc_all": pcc_all,
        "pcc_obs": pcc_obs,
        "pcc_held": pcc_held,
        "mae_all": mae_all,
        "mae_obs": mae_obs,
        "mae_held": mae_held,
        "scc_all": scc_all,
        "scc_obs": scc_obs,
        "scc_held": scc_held,
    }

    if row["dataset_family"] == "HiCImputeData":
        ctype, cdepth = parse_hicimpute_name(row["dataset"])
        payload["data_name"] = row["dataset"]
        payload["ctype"] = ctype
        payload["cdepth"] = cdepth

    for key in ("pcc_all", "pcc_obs", "pcc_held", "mae_all", "mae_obs", "mae_held", "scc_all", "scc_obs", "scc_held"):
        payload[key + "_nan_count"] = int(np.isnan(np.asarray(payload[key], dtype=np.float64)).sum())

    return payload


def build_manifest_rows(
    dataset_family: str | None = None,
    dataset: str | None = None,
    method: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_id = 0
    for family_name in FAMILY_ORDER:
        if dataset_family and family_name != dataset_family:
            continue
        for descriptor in dataset_descriptors(family_name):
            if dataset and descriptor["name"] != dataset:
                continue
            for method_name in METHOD_ORDER:
                if method and method_name != method:
                    continue
                config = method_config(method_name, family_name)
                imputed_path = resolve_imputed_path(family_name, descriptor["name"], method_name)
                row: dict[str, Any] = {
                    "task_id": task_id,
                    "dataset_family": family_name,
                    "dataset": descriptor["name"],
                    "method": method_name,
                    "loader_kind": config["loader_kind"],
                    "feature_order": config.get("feature_order", "tril"),
                    "gt_path": str(descriptor["gt_path"]),
                    "observed_path": str(descriptor["observed_path"]),
                    "imputed_path": str(imputed_path),
                    "n_cells": descriptor["expected_shape"][0],
                    "n_features": descriptor["expected_shape"][1],
                    "n_beads": descriptor["n_beads"],
                }
                if family_name == "HiCImputeData":
                    ctype, cdepth = parse_hicimpute_name(descriptor["name"])
                    row["data_name"] = descriptor["name"]
                    row["ctype"] = ctype
                    row["cdepth"] = cdepth
                rows.append(row)
                task_id += 1
    return rows


def task_row_path(dataset_family: str, method: str, dataset: str) -> Path:
    return FAMILY_DIRS[dataset_family] / "per_cell_metrics" / f"{method}__{dataset}.json"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    def json_ready(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: json_ready(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_ready(v) for v in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, float):
            return None if not np.isfinite(value) else value
        return value

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(json_ready(payload), fh, indent=2, allow_nan=False)
    tmp.replace(path)


METRIC_KEYS = ("pcc", "mae", "scc")
SUBSET_KEYS = ("all", "obs", "held")


def method_rank(method: str) -> int:
    return METHOD_ORDER.index(method)


def dataset_rank(family_name: str, dataset: str) -> int:
    ordered = [d["name"] for d in DATASET_FAMILIES[family_name]]
    return ordered.index(dataset)


def command_prepare_manifest(args: argparse.Namespace) -> None:
    ensure_output_dirs()
    rows = build_manifest_rows(args.dataset_family, args.dataset, args.method)
    if not rows:
        raise ValueError("No tasks matched the requested manifest filters")
    missing: list[str] = []
    for row in rows:
        for key in ("gt_path", "observed_path", "imputed_path"):
            if not Path(row[key]).exists():
                missing.append(f"task_id={row['task_id']} missing {key}: {row[key]}")
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"Manifest path validation failed:\n{preview}")
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Prepared {len(rows)} metric tasks -> {args.output_csv}")


def command_run_task(args: argparse.Namespace) -> None:
    ensure_output_dirs()
    manifest = pd.read_csv(args.manifest)
    if args.task_id < 0 or args.task_id >= len(manifest):
        raise IndexError(f"task_id {args.task_id} out of bounds (manifest has {len(manifest)} rows)")
    row = manifest.iloc[args.task_id].to_dict()
    gt, observed = load_truth_observed(row)
    pred = load_prediction(row)
    payload = compute_payload(row, gt, observed, pred)
    output_path = task_row_path(row["dataset_family"], row["method"], row["dataset"])
    write_json_atomic(output_path, payload)
    print(f"Completed task_id={args.task_id} method={row['method']} dataset={row['dataset']}")
    print(f"Wrote -> {output_path}")


def load_expected_payloads(
    manifest_path: Path,
    dataset_family: str | None = None,
) -> list[dict[str, Any]]:
    manifest = pd.read_csv(manifest_path)
    if dataset_family is not None:
        manifest = manifest[manifest["dataset_family"] == dataset_family]
    payloads: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in manifest.to_dict(orient="records"):
        path = task_row_path(row["dataset_family"], row["method"], row["dataset"])
        if not path.exists():
            missing.append(str(path))
            continue
        with path.open("r", encoding="utf-8") as fh:
            payloads.append(json.load(fh))
    if missing:
        raise FileNotFoundError(f"Missing task-row JSON files:\n" + "\n".join(missing[:20]))
    return payloads


def command_aggregate(args: argparse.Namespace) -> None:
    ensure_output_dirs()
    payloads = load_expected_payloads(Path(args.manifest), args.dataset_family)

    flamingo_rows: list[dict[str, Any]] = []
    hic_rows: list[dict[str, Any]] = []

    for payload in payloads:
        row: dict[str, Any] = {
            "method": payload["method"],
            "n_cells": payload["n_cells"],
            "n_features": payload["n_features"],
        }
        for mk in METRIC_KEYS:
            for sk in SUBSET_KEYS:
                key = f"{mk}_{sk}"
                mean, std, _ = summarize(payload[key])
                row[f"{key}_mean"] = mean
                row[f"{key}_std"] = std

        if payload["dataset_family"] == "FLAMINGOData":
            row["dataset"] = payload["dataset"]
            # A payload without transform metadata is an old, unverifiable result.
            row["transform"] = payload.get("transform", "legacy-unknown")
            flamingo_rows.append(row)
        else:
            row["data_name"] = payload["data_name"]
            row["ctype"] = payload["ctype"]
            row["cdepth"] = payload["cdepth"]
            row["transform"] = payload.get("transform", "raw")
            hic_rows.append(row)

    flamingo_rows.sort(key=lambda r: (method_rank(r["method"]), dataset_rank("FLAMINGOData", r["dataset"])))
    hic_rows.sort(key=lambda r: (method_rank(r["method"]), dataset_rank("HiCImputeData", r["data_name"])))

    flamingo_cols = (
        ["method", "dataset", "n_cells", "n_features", "transform"]
        + [f"{mk}_{sk}_{stat}" for mk in METRIC_KEYS for sk in SUBSET_KEYS for stat in ("mean", "std")]
    )
    hic_cols = (
        ["method", "data_name", "ctype", "cdepth", "transform"]
        + [f"{mk}_{sk}_{stat}" for mk in METRIC_KEYS for sk in SUBSET_KEYS for stat in ("mean", "std")]
    )

    flamingo_df = pd.DataFrame(flamingo_rows, columns=flamingo_cols)
    hic_df = pd.DataFrame(hic_rows, columns=hic_cols)

    if args.dataset_family in (None, "FLAMINGOData"):
        if len(flamingo_df) == 0:
            raise ValueError("No FLAMINGOData payloads selected for aggregation")
        non_raw = flamingo_df[flamingo_df["transform"] != "raw"]
        if not non_raw.empty:
            raise ValueError("FLAMINGOData aggregation found non-raw payloads; rerun all 49 tasks first")
        flamingo_df.to_csv(FLAMINGO_CSV, index=False)
        print(f"Aggregated {len(flamingo_df)} FLAMINGOData rows -> {FLAMINGO_CSV}")

    if args.dataset_family in (None, "HiCImputeData"):
        if len(hic_df) == 0:
            raise ValueError("No HiCImputeData payloads selected for aggregation")
        hic_df.to_csv(HIC_CSV, index=False)
        print(f"Aggregated {len(hic_df)} HiCImputeData rows -> {HIC_CSV}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate imputation PCC/MAE/SCC metrics")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare-manifest")
    p_prepare.add_argument("--output-csv", type=Path, default=METRIC_MANIFEST_PATH)
    p_prepare.add_argument("--dataset-family", choices=FAMILY_ORDER)
    p_prepare.add_argument("--dataset")
    p_prepare.add_argument("--method", choices=METHOD_ORDER)
    p_prepare.set_defaults(func=command_prepare_manifest)

    p_run = sub.add_parser("run-task")
    p_run.add_argument("--manifest", type=Path, default=METRIC_MANIFEST_PATH)
    p_run.add_argument("--task-id", type=int, required=True)
    p_run.set_defaults(func=command_run_task)

    p_agg = sub.add_parser("aggregate")
    p_agg.add_argument("--manifest", type=Path, default=METRIC_MANIFEST_PATH)
    p_agg.add_argument("--dataset-family", choices=FAMILY_ORDER)
    p_agg.set_defaults(func=command_aggregate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
