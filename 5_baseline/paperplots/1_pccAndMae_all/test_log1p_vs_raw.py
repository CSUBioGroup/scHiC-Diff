#!/usr/bin/env python3
"""Quick test: log1p vs raw transform impact on FLAMINGOData metrics."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from imputation_metric_config import (
    DATASET_FAMILIES,
    METHOD_ORDER,
    method_config,
    resolve_imputed_path,
    load_h5ad_gt_and_observed,
    load_sparse_triangle_features,
    load_tensor_flamingo_triu_features,
)


def safe_pearson(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def safe_mae(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0: return float("nan")
    return float(np.mean(np.abs(a - b)))

def safe_spearman(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or np.nanstd(a) == 0 or np.nanstd(b) == 0: return float("nan")
    try: return float(spearmanr(a, b)[0])
    except: return float("nan")


def compute_one(descriptor, method_name, use_log):
    config = method_config(method_name, "FLAMINGOData")
    gt, observed = load_h5ad_gt_and_observed(descriptor["gt_path"])
    n_beads = descriptor["n_beads"]
    imputed_path = resolve_imputed_path("FLAMINGOData", descriptor["name"], method_name)
    if config["loader_kind"] == "tensor_tril_encoded_triu":
        pred = load_tensor_flamingo_triu_features(imputed_path, n_beads)
    else:
        pred = load_sparse_triangle_features(
            imputed_path, n_beads=n_beads,
            feature_order=config.get("feature_order", "tril"),
            target_order="triu",
        )

    results = {k: [] for k in
               ["pcc_all","pcc_obs","pcc_held",
                "mae_all","mae_obs","mae_held",
                "scc_all","scc_obs","scc_held"]}
    for cell_idx in range(gt.shape[0]):
        gt_raw = np.asarray(gt[cell_idx], dtype=np.float64)
        obs_raw = np.asarray(observed[cell_idx], dtype=np.float64)
        pred_raw = np.asarray(pred[cell_idx], dtype=np.float64)
        if use_log:
            gt_use = np.log1p(np.maximum(gt_raw, 0.0))
            pred_use = np.log1p(np.maximum(pred_raw, 0.0))
        else:
            gt_use = gt_raw
            pred_use = pred_raw
        gt_mask = gt_raw > 0
        obs_mask = obs_raw > 0
        held_mask = gt_mask & ~obs_mask
        results["pcc_all"].append(safe_pearson(pred_use, gt_use))
        results["pcc_obs"].append(safe_pearson(pred_use[obs_mask], gt_use[obs_mask]))
        results["pcc_held"].append(safe_pearson(pred_use[held_mask], gt_use[held_mask]))
        results["mae_all"].append(safe_mae(pred_use, gt_use))
        results["mae_obs"].append(safe_mae(pred_use[obs_mask], gt_use[obs_mask]))
        results["mae_held"].append(safe_mae(pred_use[held_mask], gt_use[held_mask]))
        results["scc_all"].append(safe_spearman(pred_use, gt_use))
        results["scc_obs"].append(safe_spearman(pred_use[obs_mask], gt_use[obs_mask]))
        results["scc_held"].append(safe_spearman(pred_use[held_mask], gt_use[held_mask]))
    return {k: float(np.nanmean(v)) for k, v in results.items()}


# Test on 1 representative dataset × all 7 methods
test_datasets = [
    d for d in DATASET_FAMILIES["FLAMINGOData"]
    if d["name"] == "v3_hybrid_W0p7_500cells_level0"
]

test_methods = list(METHOD_ORDER)

ds_short = {
    "v3_hybrid_W0p7_500cells_level0": "W0.7",
}

metrics = ["pcc_all","pcc_obs","pcc_held",
           "mae_all","mae_obs","mae_held",
           "scc_all","scc_obs","scc_held"]

print(f"{'Method':20s} {'Dataset':12s} {'Metric':10s} {'log1p':>12s} {'raw':>12s} {'diff':>12s} {'changed?'}")
print("-" * 85)

for method_name in test_methods:
    for desc in test_datasets:
        log_res = compute_one(desc, method_name, use_log=True)
        raw_res = compute_one(desc, method_name, use_log=False)
        for metric in metrics:
            log_v = log_res[metric]
            raw_v = raw_res[metric]
            diff = raw_v - log_v
            is_corr = metric.startswith("pcc") or metric.startswith("scc")
            changed = abs(diff) > 1e-10
            if changed:
                tag = "⚠️ changed" if is_corr else "→ scale change (expected)"
            else:
                tag = "✅ same"
            print(f"{method_name:20s} {ds_short[desc['name']]:12s} {metric:10s} {log_v:12.6f} {raw_v:12.6f} {diff:12.6f} {tag}")
    print()