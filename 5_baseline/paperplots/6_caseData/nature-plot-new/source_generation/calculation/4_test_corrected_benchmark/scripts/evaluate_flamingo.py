#!/usr/bin/env python3
"""Run the isolated FLAMINGO loop, APA, and held-out support evaluation.

Existing six-condition outputs are read for matched comparisons but never
modified. New files are written under ``results_flamingo`` and the dedicated
``results_diagnostics/*flamingo*`` directories.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from adapters import load_csr_npz, validate_method
from heldout_apa_diagnostics import run_heldout_diagnostic
from heldout_raw_support_sensitivity import run_analysis
from run_apa import run_all_apa
from run_benchmark import (
    load_resolved_config,
    run_loop_stage,
    run_subsets,
    run_validation,
    sha256_file,
)
from summarize_results import summarize_loops


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
WORKSPACE = BENCHMARK_DIR.parent
FLAMINGO_CONFIG = BENCHMARK_DIR / "configs/benchmark_flamingo.json"
BASE_CONFIG = BENCHMARK_DIR / "configs/benchmark.json"
FLAMINGO_INPUT = WORKSPACE / "imputedData/FLAMINGO/chrom_npz/chr1.npz"
FLAMINGO_RESULTS = BENCHMARK_DIR / "results_flamingo"
HELDOUT_FLAMINGO = BENCHMARK_DIR / "results_diagnostics/heldout_flamingo"
SUPPORT_WITH_FLAMINGO = (
    BENCHMARK_DIR / "results_diagnostics/heldout_raw_support_sensitivity_with_flamingo"
)
SEEDS = [42, 43, 44]


def _row_sums_without_diagonal(matrix, *, include_diagonal: bool, n_bins: int = 100) -> np.ndarray:
    if include_diagonal:
        rows, columns = np.triu_indices(n_bins)
        off_diagonal_columns = np.flatnonzero(rows < columns)
        matrix = matrix[:, off_diagonal_columns]
    return np.asarray(matrix.sum(axis=1)).reshape(-1)


def run_preflight() -> dict[str, object]:
    """Validate layout and record the strongest available row-order evidence."""
    base = load_resolved_config(BASE_CONFIG)
    raw = next(method for method in base["methods"] if method["slug"] == "raw")
    validation = validate_method(
        FLAMINGO_INPUT,
        include_diagonal=False,
        expected_cells=7466,
        n_bins=100,
    )
    flamingo = load_csr_npz(FLAMINGO_INPUT)
    raw_matrix = load_csr_npz(raw["input_npz"])
    flamingo_sums = _row_sums_without_diagonal(flamingo, include_diagonal=False)
    raw_sums = _row_sums_without_diagonal(raw_matrix, include_diagonal=True)
    pearson = stats.pearsonr(raw_sums, flamingo_sums)
    spearman = stats.spearmanr(raw_sums, flamingo_sums)
    payload: dict[str, object] = {
        **validation,
        "sha256": sha256_file(FLAMINGO_INPUT),
        "cellnames_present": False,
        "exact_cell_name_verification": False,
        "row_order_assumption": (
            "FLAMINGO preserves the canonical 7,466-cell input order; archive has no cellnames"
        ),
        "row_order_support": {
            "raw_vs_flamingo_row_sum_pearson_r": float(pearson.statistic),
            "raw_vs_flamingo_row_sum_pearson_p": float(pearson.pvalue),
            "raw_vs_flamingo_row_sum_spearman_rho": float(spearman.statistic),
            "raw_vs_flamingo_row_sum_spearman_p": float(spearman.pvalue),
        },
        "raw_offdiagonal_row_sum_quantiles": np.quantile(
            raw_sums, [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "flamingo_row_sum_quantiles": np.quantile(
            flamingo_sums, [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
    }
    manifest_dir = FLAMINGO_RESULTS / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "flamingo_preflight.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


def run_main_loops(force: bool = False) -> pd.DataFrame:
    config = load_resolved_config(FLAMINGO_CONFIG)
    run_validation(config)
    run_subsets(config)
    runs = run_loop_stage(config, force=force)
    expected_counts = {10: 3, 100: 3, 200: 3, 476: 1}
    observed = runs.groupby("cell_count").size().to_dict()
    if observed != expected_counts:
        raise ValueError(f"unexpected FLAMINGO loop-run design: {observed} != {expected_counts}")
    summary_dir = FLAMINGO_RESULTS / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summarize_loops(runs).to_csv(summary_dir / "loop_summary.csv", index=False)
    return runs


def run_standard_apa(force: bool = False) -> pd.DataFrame:
    config = load_resolved_config(FLAMINGO_CONFIG)
    manifest = run_all_apa(config, force=force)
    if not manifest["status"].isin(["completed", "no_eligible_loops"]).all():
        raise RuntimeError("one or more standard FLAMINGO APA jobs did not finish")
    return manifest


def _copy_if_needed(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or sha256_file(destination) != sha256_file(source):
        shutil.copy2(source, destination)


def seed_heldout_reuse() -> None:
    """Reuse identical raw references and scHiC-Diff loop calls for each split."""
    base_root = BENCHMARK_DIR / "results_diagnostics/heldout_raw_reference"
    for seed in SEEDS:
        source_seed = base_root / f"seed{seed}"
        target_seed = HELDOUT_FLAMINGO / f"seed{seed}"
        for name in ["raw_reference_sum.hic", "raw_reference_sum.npz", "chrom.sizes"]:
            _copy_if_needed(
                source_seed / "reference" / name,
                target_seed / "reference" / name,
            )
        source_loop_dir = source_seed / "loops/schicdiff"
        for source in source_loop_dir.glob("loops.*"):
            if source.is_file():
                _copy_if_needed(source, target_seed / "loops/schicdiff" / source.name)
        run_log = source_loop_dir / "run.log"
        if run_log.is_file():
            _copy_if_needed(run_log, target_seed / "loops/schicdiff/run.log")


def _heldout_config() -> dict:
    config = load_resolved_config(BASE_CONFIG)
    if any(method["slug"] == "flamingo" for method in config["methods"]):
        raise ValueError("base config unexpectedly already contains FLAMINGO")
    config["methods"] = [
        *config["methods"],
        {
            "name": "FLAMINGO",
            "slug": "flamingo",
            "role": "imputation",
            "input_npz": FLAMINGO_INPUT.resolve(),
            "include_diagonal": False,
        },
    ]
    return config


def run_heldout_apa() -> pd.DataFrame:
    seed_heldout_reuse()
    return run_heldout_diagnostic(
        _heldout_config(),
        output_root=HELDOUT_FLAMINGO,
        seeds=SEEDS,
        method_slugs=["schicdiff", "flamingo"],
        top_n_values=[10, 20, 50, 100, 200],
    )


def _support_methods() -> tuple[list[dict], dict[int, list[Path]]]:
    diagnostics = BENCHMARK_DIR / "results_diagnostics"
    definitions = [
        (
            "baseline_schicdiff",
            "scHiC-Diff",
            diagnostics / "heldout_raw_reference",
            "schicdiff",
        ),
        ("flamingo", "FLAMINGO", diagnostics / "heldout_flamingo", "flamingo"),
        ("scvi3d", "scVI-3D", diagnostics / "heldout_raw_reference", "scvi3d"),
        (
            "schicluster",
            "scHiCluster",
            diagnostics / "heldout_comparator_methods/scHiCluster",
            "schicdiff",
        ),
        (
            "higashi_nbr0",
            "Higashi-0",
            diagnostics / "heldout_comparator_methods/Higashi_nbr0",
            "schicdiff",
        ),
        (
            "higashi_nbr5",
            "Higashi-5",
            diagnostics / "heldout_comparator_methods/Higashi_nbr5",
            "schicdiff",
        ),
    ]
    methods = [
        {
            "method": slug,
            "method_name": label,
            "loop_paths": {
                seed: root / f"seed{seed}/loops/{loop_slug}/loops.loop.bedpe"
                for seed in SEEDS
            },
        }
        for slug, label, root, loop_slug in definitions
    ]
    reference_root = diagnostics / "heldout_raw_reference"
    references = {
        seed: [reference_root / f"seed{seed}/reference/raw_reference_sum.npz"]
        for seed in SEEDS
    }
    return methods, references


def run_support_analysis() -> dict[str, pd.DataFrame]:
    methods, references = _support_methods()
    return run_analysis(
        methods=methods,
        reference_paths=references,
        output_root=SUPPORT_WITH_FLAMINGO,
        seeds=SEEDS,
        top_n_values=[10, 20, 50, 100, 200],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["preflight", "loops", "apa", "heldout", "support", "all"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = ["preflight", "loops", "apa", "heldout", "support"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"[{stage}]", flush=True)
        if stage == "preflight":
            result = run_preflight()
            print(json.dumps(result["row_order_support"], indent=2), flush=True)
        elif stage == "loops":
            result = run_main_loops(force=args.force)
            print(result[["cell_count", "seed", "loop_count", "summit_count"]].to_string(index=False), flush=True)
        elif stage == "apa":
            result = run_standard_apa(force=args.force)
            print(result[["min_distance_bins", "set_label", "written_count", "effective_count", "P2LL"]].to_string(index=False), flush=True)
        elif stage == "heldout":
            result = run_heldout_apa()
            print(result["status"].value_counts().to_string(), flush=True)
        elif stage == "support":
            result = run_support_analysis()
            print(f"per_loop={len(result['per_loop'])}; topn={len(result['topn_summary'])}", flush=True)


if __name__ == "__main__":
    main()

