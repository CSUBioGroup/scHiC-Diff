#!/usr/bin/env python3
"""Repeated held-out raw-reference APA diagnostics for selected methods."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from adapters import load_csr_npz, standardize_feature_matrix
from build_group_matrices import aggregate_standard_vectors
from map2_metrics import expected_by_distance, extract_corrected_metrics, observed_over_expected
from prepare_apa import BEDPE_COLUMNS, filter_and_rank_loops, load_loop_bedpe, prepare_apa_set
from run_apa import run_apa_job
from run_benchmark import import_legacy_loop_caller, load_resolved_config, vector_to_symmetric_matrix


def make_complementary_split(indices: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split an even, unique index set into deterministic caller/reference halves."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("indices must be a non-empty one-dimensional array")
    if indices.size % 2:
        raise ValueError("held-out split requires an even number of indices")
    if np.unique(indices).size != indices.size:
        raise ValueError("indices must be unique")
    shuffled = np.random.RandomState(seed).permutation(indices)
    midpoint = indices.size // 2
    return np.sort(shuffled[:midpoint]), np.sort(shuffled[midpoint:])


def resolve_heldout_indices(
    *,
    raw_rows: int,
    method_rows: dict[str, int],
    local_row_indexing: bool,
    configured_indices: np.ndarray,
) -> np.ndarray:
    """Choose either legacy-local rows or the benchmark's canonical cell indices."""
    if local_row_indexing:
        mismatched = {
            name: rows for name, rows in method_rows.items() if int(rows) != int(raw_rows)
        }
        if mismatched:
            details = ", ".join(f"{name}={rows}" for name, rows in sorted(mismatched.items()))
            raise ValueError(
                "local row indexing requires every caller matrix to have the same row count "
                f"as the raw reference ({raw_rows}); mismatched row count: {details}"
            )
        if raw_rows <= 0 or raw_rows % 2:
            raise ValueError("local row indexing requires a positive, even raw-reference row count")
        return np.arange(raw_rows, dtype=np.int64)
    configured_indices = np.asarray(configured_indices, dtype=np.int64)
    if configured_indices.size == 0:
        raise ValueError("configured held-out indices are empty")
    if configured_indices.max() >= raw_rows:
        raise ValueError(
            "configured canonical indices exceed raw-reference rows; use --local-row-indexing "
            "for a compact legacy archive"
        )
    mismatched = {
        name: rows for name, rows in method_rows.items() if int(rows) != int(raw_rows)
    }
    if mismatched:
        details = ", ".join(f"{name}={rows}" for name, rows in sorted(mismatched.items()))
        raise ValueError(f"caller matrices do not match raw-reference row count {raw_rows}: {details}")
    return configured_indices


def _separation_bins(frame: pd.DataFrame, resolution: int) -> pd.Series:
    return ((frame["start2"] - frame["start1"]).abs() // resolution).astype(int)


def select_exact_distance_matched(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    resolution: int,
    min_distance_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep equal top-scored counts at every exact anchor separation."""
    if resolution <= 0 or min_distance_bins < 0:
        raise ValueError("resolution must be positive and minimum distance non-negative")
    first = first[BEDPE_COLUMNS].copy()
    second = second[BEDPE_COLUMNS].copy()
    first["distance_bins"] = _separation_bins(first, resolution)
    second["distance_bins"] = _separation_bins(second, resolution)
    first = first[first["distance_bins"] >= min_distance_bins]
    second = second[second["distance_bins"] >= min_distance_bins]
    first_parts = []
    second_parts = []
    for distance in sorted(set(first["distance_bins"]) & set(second["distance_bins"])):
        first_band = first[first["distance_bins"] == distance].sort_values(
            "score", ascending=False, kind="mergesort"
        )
        second_band = second[second["distance_bins"] == distance].sort_values(
            "score", ascending=False, kind="mergesort"
        )
        count = min(len(first_band), len(second_band))
        if count:
            first_parts.append(first_band.head(count))
            second_parts.append(second_band.head(count))
    empty = pd.DataFrame(columns=BEDPE_COLUMNS)
    first_matched = pd.concat(first_parts, ignore_index=True)[BEDPE_COLUMNS] if first_parts else empty
    second_matched = pd.concat(second_parts, ignore_index=True)[BEDPE_COLUMNS] if second_parts else empty
    return first_matched, second_matched


def aggregate_loop_metrics(
    raw_matrix: np.ndarray,
    loops: pd.DataFrame,
    *,
    resolution: int,
    center_size: int = 5,
    outer_size: int = 21,
) -> dict[str, float | int]:
    """Average raw and diagonal O/E APA geometry across a fixed loop set."""
    raw_matrix = np.asarray(raw_matrix, dtype=float)
    expected = expected_by_distance(raw_matrix)
    oe_matrix = observed_over_expected(raw_matrix, expected)
    raw_rows = []
    oe_rows = []
    for loop in loops.itertuples(index=False):
        bin1 = int(loop.start1) // resolution
        bin2 = int(loop.start2) // resolution
        try:
            raw_rows.append(
                extract_corrected_metrics(
                    raw_matrix, bin1, bin2, center_size=center_size, outer_size=outer_size
                )
            )
            oe_rows.append(
                extract_corrected_metrics(
                    oe_matrix, bin1, bin2, center_size=center_size, outer_size=outer_size
                )
            )
        except ValueError:
            continue
    if not raw_rows:
        return {"loop_count": 0, "raw_center": np.nan, "raw_lower_left": np.nan,
                "raw_donut": np.nan, "raw_p2ll": np.nan, "raw_p2donut": np.nan,
                "oe_center": np.nan, "oe_lower_left": np.nan, "oe_donut": np.nan,
                "oe_p2ll": np.nan, "oe_p2donut": np.nan}
    raw = pd.DataFrame(raw_rows).mean(numeric_only=True)
    oe = pd.DataFrame(oe_rows).mean(numeric_only=True)
    return {
        "loop_count": len(raw_rows),
        "raw_center": float(raw["center_signal"]),
        "raw_lower_left": float(raw["lower_left_signal"]),
        "raw_donut": float(raw["donut_signal"]),
        "raw_p2ll": float(raw["lower_left_ratio"]),
        "raw_p2donut": float(raw["donut_ratio"]),
        "oe_center": float(oe["center_signal"]),
        "oe_lower_left": float(oe["lower_left_signal"]),
        "oe_donut": float(oe["donut_signal"]),
        "oe_p2ll": float(oe["lower_left_ratio"]),
        "oe_p2donut": float(oe["donut_ratio"]),
    }


def _write_single_vector_csr(vector: np.ndarray, path: Path) -> None:
    matrix = sparse.csr_matrix(np.asarray(vector, dtype=float).reshape(1, -1))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.asarray(matrix.shape),
    )


def _load_hic_converter():
    script = Path(__file__).resolve().parents[2] / "3_experiment_apa/scripts/selected_cells_to_hic.py"
    specification = importlib.util.spec_from_file_location("heldout_hic_converter", script)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot import local .hic converter: {script}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _method_by_slug(config: dict, slugs: list[str]) -> dict[str, dict]:
    available = {method["slug"]: method for method in config["methods"]}
    missing = sorted(set(slugs).difference(available))
    if missing:
        raise ValueError(f"methods absent from config: {missing}")
    return {slug: available[slug] for slug in slugs}


def replace_method_input(
    config: dict,
    slug: str,
    input_npz: str | Path,
    label: str,
) -> dict:
    """Return a config copy with one method redirected to a new NPZ result."""
    input_npz = Path(input_npz).resolve()
    if not input_npz.is_file():
        raise FileNotFoundError(input_npz)
    updated = dict(config)
    updated_methods = []
    found = False
    for method in config["methods"]:
        replacement = dict(method)
        if replacement["slug"] == slug:
            replacement["input_npz"] = input_npz
            replacement["name"] = label
            found = True
        updated_methods.append(replacement)
    if not found:
        raise ValueError(f"method absent from config: {slug}")
    updated["methods"] = updated_methods
    return updated


def run_heldout_diagnostic(
    config: dict,
    *,
    output_root: str | Path,
    seeds: list[int],
    method_slugs: list[str],
    top_n_values: list[int] = [10, 20, 50, 100, 200],
    local_row_indexing: bool = False,
) -> pd.DataFrame:
    """Run repeated caller/reference splits and write auditable APA records."""
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    methods = _method_by_slug(config, method_slugs)
    raw_method = _method_by_slug(config, ["raw"])["raw"]
    raw_source = load_csr_npz(raw_method["input_npz"])
    sources = {slug: load_csr_npz(method["input_npz"]) for slug, method in methods.items()}
    configured_early = np.load(config["output_root"] / "subsets/earlyNeurons_476cells_full.npy")
    all_early = resolve_heldout_indices(
        raw_rows=raw_source.shape[0],
        method_rows={slug: source.shape[0] for slug, source in sources.items()},
        local_row_indexing=local_row_indexing,
        configured_indices=configured_early,
    )
    caller = import_legacy_loop_caller(config["legacy_loop_script"])
    converter = _load_hic_converter()
    apa_config = dict(config["apa"])
    local_jar = Path.home() / "Downloads/juicer_tools.2.20.00.jar"
    if local_jar.is_file():
        apa_config["juicer_jar"] = local_jar
    loop_config = config["loop_calling"]
    records: list[dict[str, object]] = []

    for seed in seeds:
        caller_indices, reference_indices = make_complementary_split(all_early, seed)
        split_dir = output_root / f"seed{seed}"
        reference_vector = aggregate_standard_vectors(
            raw_source,
            reference_indices,
            include_diagonal=raw_method["include_diagonal"],
            n_bins=config["n_bins"],
        )
        reference_matrix = vector_to_symmetric_matrix(reference_vector, config["n_bins"])
        reference_csr = split_dir / "reference" / "raw_reference_sum.npz"
        _write_single_vector_csr(reference_vector * len(reference_indices), reference_csr)
        reference_hic = split_dir / "reference" / "raw_reference_sum.hic"
        if not reference_hic.exists():
            converter.convert_single(
                reference_csr,
                reference_hic,
                juicer_jar=apa_config["juicer_jar"],
                resolution=int(apa_config["resolution"]),
                chrom="chr1",
                chrom_size=int(config["n_bins"]) * int(apa_config["resolution"]),
            )
        split_apa = dict(apa_config)
        split_apa["reference_hic"] = reference_hic
        loop_frames: dict[str, pd.DataFrame] = {}
        for slug, method in methods.items():
            loop_dir = split_dir / "loops" / slug
            prefix = loop_dir / "loops"
            loop_path = prefix.with_suffix(".loop.bedpe")
            if not loop_path.exists():
                selected = standardize_feature_matrix(
                    sources[slug][caller_indices],
                    include_diagonal=method["include_diagonal"],
                    n_bins=config["n_bins"],
                ).toarray()
                loop_dir.mkdir(parents=True, exist_ok=True)
                with (loop_dir / "run.log").open("w") as handle:
                    with redirect_stdout(handle), redirect_stderr(handle):
                        caller.call_loops(
                            selected,
                            output_prefix=prefix,
                            resolution=int(loop_config["resolution"]),
                            n_bins=int(loop_config["n_bins"]),
                            min_dist=int(loop_config["min_dist"]),
                            max_dist=int(loop_config["max_dist"]),
                            cap=float(loop_config["cap"]),
                            pad=int(loop_config["pad"]),
                            gap=int(loop_config["gap"]),
                            fdr_thres=float(loop_config["fdr"]),
                            dist_thres=int(loop_config["dist_thres"]),
                            size_thres=int(loop_config["size_thres"]),
                        )
            loop_frames[slug] = load_loop_bedpe(loop_path)

        for minimum_bins in [4, 30]:
            for slug, method in methods.items():
                for top_n in top_n_values:
                    branch = split_dir / "apa" / f"min{minimum_bins}bins" / slug
                    record = prepare_apa_set(
                        loop_frames[slug],
                        branch / "bedpe" / f"top{top_n}.bedpe",
                        resolution=int(apa_config["resolution"]),
                        min_distance_bins=minimum_bins,
                        top_n=top_n,
                    )
                    record.update(
                        {
                            "seed": seed,
                            "reference_cells": len(reference_indices),
                            "caller_cells": len(caller_indices),
                            "method": slug,
                            "method_name": method["name"],
                            "comparison_kind": "rank_prefix",
                            "set_label": f"top{top_n}",
                            "apa_dir": str((branch / "runs" / f"top{top_n}").resolve()),
                            "figure_path": str((branch / "figures" / f"top{top_n}.png").resolve()),
                        }
                    )
                    result = run_apa_job(record, split_apa)
                    selected = load_loop_bedpe(result["bedpe_path"])
                    result.update(
                        aggregate_loop_metrics(
                            reference_matrix, selected, resolution=int(apa_config["resolution"])
                        )
                    )
                    records.append(result)

            first_slug, second_slug = method_slugs
            first_matched, second_matched = select_exact_distance_matched(
                loop_frames[first_slug],
                loop_frames[second_slug],
                resolution=int(apa_config["resolution"]),
                min_distance_bins=minimum_bins,
            )
            for slug, selected in [(first_slug, first_matched), (second_slug, second_matched)]:
                method = methods[slug]
                branch = split_dir / "apa" / f"min{minimum_bins}bins" / slug
                record = prepare_apa_set(
                    selected,
                    branch / "bedpe" / "exact_distance_matched_all.bedpe",
                    resolution=int(apa_config["resolution"]),
                    min_distance_bins=minimum_bins,
                    top_n=None,
                )
                record.update(
                    {
                        "seed": seed,
                        "reference_cells": len(reference_indices),
                        "caller_cells": len(caller_indices),
                        "method": slug,
                        "method_name": method["name"],
                        "comparison_kind": "exact_distance_matched",
                        "set_label": "all",
                        "apa_dir": str((branch / "runs" / "exact_distance_matched_all").resolve()),
                        "figure_path": str((branch / "figures" / "exact_distance_matched_all.png").resolve()),
                    }
                )
                result = run_apa_job(record, split_apa)
                result.update(
                    aggregate_loop_metrics(
                        reference_matrix,
                        load_loop_bedpe(result["bedpe_path"]),
                        resolution=int(apa_config["resolution"]),
                    )
                )
                records.append(result)

    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output_root / "apa_per_split.csv", index=False)
    numeric = [
        "P2LL", "P2M", "ZscoreLL", "raw_center", "raw_lower_left", "raw_donut",
        "raw_p2ll", "raw_p2donut", "oe_center", "oe_lower_left", "oe_donut",
        "oe_p2ll", "oe_p2donut", "effective_count", "loop_count",
    ]
    summary = frame.groupby(
        ["method", "method_name", "comparison_kind", "min_distance_bins", "set_label"],
        dropna=False,
    )[numeric].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(part for part in column if part) for column in summary.columns]
    summary.to_csv(output_root / "apa_summary.csv", index=False)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="4_test_corrected_benchmark/configs/benchmark.json")
    parser.add_argument(
        "--output-root",
        default="4_test_corrected_benchmark/results_diagnostics/heldout_raw_reference",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--schicdiff-input-npz",
        help="Replace the baseline scHiC-Diff input with one counts-like Ramani result",
    )
    parser.add_argument(
        "--schicdiff-label",
        help="Label written to the manifest when --schicdiff-input-npz is used",
    )
    parser.add_argument(
        "--raw-input-npz",
        help="Replace the raw reference input, for example with a compact legacy raw archive",
    )
    parser.add_argument(
        "--comparison-input-npz",
        help="Replace the scVI-3D comparison input with another counts-like matrix",
    )
    parser.add_argument(
        "--comparison-label",
        help="Label written when --comparison-input-npz is used",
    )
    parser.add_argument(
        "--comparison-include-diagonal",
        action="store_true",
        help="Declare that the replacement comparison matrix has 5,050 diagonal-including features",
    )
    parser.add_argument(
        "--local-row-indexing",
        action="store_true",
        help="Split compact, aligned archives by their local row indices instead of canonical cell IDs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_resolved_config(args.config)
    if args.raw_input_npz:
        config = replace_method_input(config, "raw", args.raw_input_npz, "Raw")
    if args.schicdiff_input_npz:
        config = replace_method_input(
            config,
            "schicdiff",
            args.schicdiff_input_npz,
            args.schicdiff_label or Path(args.schicdiff_input_npz).parent.name,
        )
    if args.comparison_input_npz:
        config = replace_method_input(
            config,
            "scvi3d",
            args.comparison_input_npz,
            args.comparison_label or Path(args.comparison_input_npz).parent.name,
        )
        if args.comparison_include_diagonal:
            config["methods"] = [
                {**method, "include_diagonal": True}
                if method["slug"] == "scvi3d" else method
                for method in config["methods"]
            ]
    result = run_heldout_diagnostic(
        config,
        output_root=args.output_root,
        seeds=args.seeds,
        method_slugs=["schicdiff", "scvi3d"],
        local_row_indexing=args.local_row_indexing,
    )
    print(result["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
