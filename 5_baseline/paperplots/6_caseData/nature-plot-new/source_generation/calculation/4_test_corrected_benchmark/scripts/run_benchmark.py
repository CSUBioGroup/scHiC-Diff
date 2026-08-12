#!/usr/bin/env python3
"""Run validation, repeated loop calling, and corrected Map2 analysis."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from adapters import (
    load_canonical_names,
    load_csr_npz,
    load_npz_cellnames,
    standardize_feature_matrix,
    validate_method,
    validate_named_archive,
)
from build_group_matrices import (
    aggregate_standard_vectors,
    build_group_indices,
    write_subset_manifest,
)
from map2_metrics import (
    apply_transform,
    empirical_control_stats,
    expected_by_distance,
    extract_corrected_metrics,
    generate_distance_matched_controls,
    observed_over_expected,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _resolve(base: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_resolved_config(config_path: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent
    benchmark = _read_json(config_path)
    methods_path = _resolve(config_dir, benchmark["methods_config"])
    loci_path = _resolve(config_dir, benchmark["loci_config"])
    methods_config = _read_json(methods_path)
    loci_config = _read_json(loci_path)

    methods = []
    for method in methods_config["methods"]:
        resolved = dict(method)
        resolved["input_npz"] = _resolve(methods_path.parent, method["input_npz"])
        methods.append(resolved)

    apa = dict(benchmark["apa"])
    apa["reference_hic"] = _resolve(config_dir, apa["reference_hic"])
    apa["java_bin"] = _resolve(config_dir, apa["java_bin"])
    apa["juicer_jar"] = _resolve(config_dir, apa["juicer_jar"])

    resolved = dict(benchmark)
    resolved.update(
        {
            "config_path": config_path,
            "config_dir": config_dir,
            "methods_config_path": methods_path,
            "loci_config_path": loci_path,
            "methods": methods,
            "loci": loci_config["loci"],
            "canonical_h5ad": _resolve(methods_path.parent, methods_config["canonical_h5ad"]),
            "canonical_named_npz": _resolve(
                methods_path.parent, methods_config["canonical_named_npz"]
            ),
            "early_neurons_npz": _resolve(
                methods_path.parent, methods_config["early_neurons_npz"]
            ),
            "expected_cells": int(methods_config["expected_cells"]),
            "n_bins": int(methods_config["n_bins"]),
            "legacy_loop_script": _resolve(config_dir, benchmark["legacy_loop_script"]),
            "output_root": _resolve(config_dir, benchmark["output_root"]),
            "apa": apa,
        }
    )
    return resolved


def import_legacy_loop_caller(script_path: str | Path):
    script_path = Path(script_path).resolve()
    specification = importlib.util.spec_from_file_location("corrected_benchmark_legacy_loop", script_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot import legacy loop caller: {script_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def make_run_id(method_slug: str, group: str, cell_count: int, seed: int) -> str:
    return f"{method_slug}__{group}__{cell_count}cells__seed{seed}"


def loop_run_dir(
    output_root: str | Path,
    method_slug: str,
    group: str,
    cell_count: int,
    seed: int,
) -> Path:
    return Path(output_root) / "loops" / method_slug / group / f"{cell_count}cells_seed{seed}"


def write_run_status(directory: str | Path, status: str, **fields) -> Path:
    run_dir = Path(directory)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, **fields}
    status_path = run_dir / "run_status.json"
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return status_path


def is_completed_run(
    run_dir: str | Path,
    subset_sha256: str,
    input_sha256: str,
) -> bool:
    status_path = Path(run_dir) / "run_status.json"
    if not status_path.exists():
        return False
    try:
        payload = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "completed"
        and payload.get("subset_sha256") == subset_sha256
        and payload.get("input_sha256") == input_sha256
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def vector_to_symmetric_matrix(vector: np.ndarray, n_bins: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    expected = n_bins * (n_bins + 1) // 2
    if vector.shape != (expected,):
        raise ValueError(f"standard vector must have shape {(expected,)}, got {vector.shape}")
    matrix = np.zeros((n_bins, n_bins), dtype=float)
    upper = np.triu_indices(n_bins)
    matrix[upper] = vector
    return matrix + matrix.T - np.diag(np.diag(matrix))


def _canonical_groups(config: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    canonical = load_canonical_names(config["canonical_h5ad"])
    validate_named_archive(canonical, config["canonical_named_npz"])
    early_names = load_npz_cellnames(config["early_neurons_npz"])
    return canonical, build_group_indices(canonical, early_names)


def run_validation(config: dict) -> pd.DataFrame:
    _, groups = _canonical_groups(config)
    records = []
    for method in config["methods"]:
        record = validate_method(
            method["input_npz"],
            include_diagonal=method["include_diagonal"],
            expected_cells=config["expected_cells"],
            n_bins=config["n_bins"],
        )
        record.update({"slug": method["slug"], "name": method["name"], "role": method["role"]})
        records.append(record)
    output_dir = config["output_root"] / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output_dir / "adapter_validation.csv", index=False)
    (output_dir / "group_sizes.json").write_text(
        json.dumps({name: int(len(indices)) for name, indices in groups.items()}, indent=2) + "\n"
    )
    return frame


def run_subsets(config: dict) -> pd.DataFrame:
    _, groups = _canonical_groups(config)
    return write_subset_manifest(
        groups=groups,
        counts=[int(value) for value in config["cell_counts"]],
        seeds=[int(value) for value in config["seeds"]],
        output_dir=config["output_root"] / "subsets",
    )


def _load_unique_subsets(config: dict, smoke: bool = False) -> pd.DataFrame:
    manifest_path = config["output_root"] / "subsets" / "subset_manifest.csv"
    if not manifest_path.exists():
        run_subsets(config)
    manifest = pd.read_csv(manifest_path)
    configured_groups = [str(value) for value in config.get("cell_groups", [])]
    if configured_groups:
        missing_groups = sorted(set(configured_groups).difference(manifest["group"].unique()))
        if missing_groups:
            raise ValueError(f"configured cell groups are absent from subset manifest: {missing_groups}")
        manifest = manifest.loc[manifest["group"].isin(configured_groups)].copy()
    manifest = manifest.drop_duplicates(["group", "cell_count", "index_sha256"]).copy()
    if smoke:
        manifest = manifest[
            (manifest["group"] == "earlyNeurons")
            & (manifest["seed"] == 42)
            & (manifest["cell_count"].isin([3, 10]))
        ].copy()
    return manifest.sort_values(["group", "cell_count", "seed"]).reset_index(drop=True)


def run_map2_stage(config: dict, smoke: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    subsets = _load_unique_subsets(config, smoke=smoke)
    locus = config["loci"][0]
    loop = locus["known_loops"][0]
    bin1, bin2 = int(loop["bin1"]), int(loop["bin2"])
    map_config = config["map2"]
    metric_records: list[dict[str, object]] = []
    control_records: list[dict[str, object]] = []

    for method in config["methods"]:
        source = load_csr_npz(method["input_npz"])
        for subset in subsets.itertuples(index=False):
            indices = np.load(subset.subset_path)
            standard_vector = aggregate_standard_vectors(
                source,
                indices=indices,
                include_diagonal=method["include_diagonal"],
                n_bins=config["n_bins"],
            )
            base_matrix = vector_to_symmetric_matrix(standard_vector, config["n_bins"])
            controls = generate_distance_matched_controls(
                n_bins=config["n_bins"],
                bin1=bin1,
                bin2=bin2,
                exclusion_bins=int(map_config["control_exclusion_bins"]),
                limit=int(map_config["control_limit"]),
                seed=int(subset.seed),
                outer_radius=int(map_config["outer_size"]) // 2,
            )
            for transform in config["transform_modes"]:
                transformed = apply_transform(base_matrix, transform)
                expected = expected_by_distance(transformed)
                oe_matrix = observed_over_expected(
                    transformed,
                    expected,
                    epsilon=float(map_config["epsilon"]),
                )
                positive_metrics = extract_corrected_metrics(
                    oe_matrix,
                    bin1=bin1,
                    bin2=bin2,
                    center_size=int(map_config["center_size"]),
                    outer_size=int(map_config["outer_size"]),
                    epsilon=float(map_config["epsilon"]),
                )
                control_metrics = [
                    extract_corrected_metrics(
                        oe_matrix,
                        bin1=control_bin1,
                        bin2=control_bin2,
                        center_size=int(map_config["center_size"]),
                        outer_size=int(map_config["outer_size"]),
                        epsilon=float(map_config["epsilon"]),
                    )
                    for control_bin1, control_bin2 in controls
                ]
                for background in config["background_modes"]:
                    ratio_key = f"{background}_ratio"
                    signal_key = f"{background}_signal"
                    log_key = f"{background}_log2_enrichment"
                    control_values = np.asarray([entry[ratio_key] for entry in control_metrics])
                    stats = empirical_control_stats(positive_metrics[ratio_key], control_values)
                    base_record = {
                        "method": method["slug"],
                        "method_name": method["name"],
                        "group": subset.group,
                        "cell_count": int(subset.cell_count),
                        "seed": int(subset.seed),
                        "subset_sha256": subset.index_sha256,
                        "transform": transform,
                        "background": background,
                        "loop_id": loop["loop_id"],
                        "bin1": bin1,
                        "bin2": bin2,
                        "center_signal": positive_metrics["center_signal"],
                        "background_signal": positive_metrics[signal_key],
                        "ratio": positive_metrics[ratio_key],
                        "log2_enrichment": positive_metrics[log_key],
                        **stats,
                    }
                    metric_records.append(base_record)
                    for (control_bin1, control_bin2), value in zip(controls, control_values):
                        control_records.append(
                            {
                                "method": method["slug"],
                                "group": subset.group,
                                "cell_count": int(subset.cell_count),
                                "seed": int(subset.seed),
                                "transform": transform,
                                "background": background,
                                "control_bin1": control_bin1,
                                "control_bin2": control_bin2,
                                "ratio": float(value),
                            }
                        )
        del source

    root = config["output_root"] / ("smoke/map2" if smoke else "map2")
    root.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame.from_records(metric_records)
    controls_frame = pd.DataFrame.from_records(control_records)
    metrics_frame.to_csv(root / "map2_metrics.csv", index=False)
    controls_frame.to_csv(root / "map2_controls.csv", index=False)
    return metrics_frame, controls_frame


def _count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def run_loop_stage(config: dict, smoke: bool = False, force: bool = False) -> pd.DataFrame:
    caller = import_legacy_loop_caller(config["legacy_loop_script"])
    subsets = _load_unique_subsets(config, smoke=smoke)
    output_root = config["output_root"] / "smoke" if smoke else config["output_root"]
    loop_config = config["loop_calling"]
    records: list[dict[str, object]] = []

    for method in config["methods"]:
        source_hash = sha256_file(method["input_npz"])
        source = load_csr_npz(method["input_npz"])
        for subset in subsets.itertuples(index=False):
            run_dir = loop_run_dir(
                output_root,
                method["slug"],
                subset.group,
                int(subset.cell_count),
                int(subset.seed),
            )
            if not force and is_completed_run(run_dir, subset.index_sha256, source_hash):
                records.append(json.loads((run_dir / "run_status.json").read_text()))
                continue
            indices = np.load(subset.subset_path)
            selected = standardize_feature_matrix(
                source[indices],
                include_diagonal=method["include_diagonal"],
                n_bins=config["n_bins"],
            ).toarray()
            run_dir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            try:
                with (run_dir / "run.log").open("w") as log_handle:
                    with redirect_stdout(log_handle), redirect_stderr(log_handle):
                        candidates, filtered = caller.call_loops(
                            selected,
                            output_prefix=run_dir / "loops",
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
                record = {
                    "status": "completed",
                    "run_id": make_run_id(
                        method["slug"], subset.group, int(subset.cell_count), int(subset.seed)
                    ),
                    "method": method["slug"],
                    "group": subset.group,
                    "cell_count": int(subset.cell_count),
                    "seed": int(subset.seed),
                    "subset_sha256": subset.index_sha256,
                    "input_sha256": source_hash,
                    "candidate_count": int(len(candidates)),
                    "loop_count": int(len(filtered)),
                    "summit_count": _count_nonempty_lines(run_dir / "loops.loop_summit.bedpe"),
                    "elapsed_seconds": round(time.time() - started, 6),
                    "run_dir": str(run_dir.resolve()),
                }
                write_run_status(run_dir, **record)
                records.append(record)
            except Exception as error:
                write_run_status(
                    run_dir,
                    status="failed",
                    method=method["slug"],
                    group=subset.group,
                    cell_count=int(subset.cell_count),
                    seed=int(subset.seed),
                    subset_sha256=subset.index_sha256,
                    input_sha256=source_hash,
                    error=f"{type(error).__name__}: {error}",
                )
                raise
        del source

    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(manifest_root / "loop_runs.csv", index=False)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=["validate", "subsets", "map2", "loops", "smoke", "all"],
        required=True,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_resolved_config(args.config)
    if args.stage == "validate":
        run_validation(config)
    elif args.stage == "subsets":
        run_subsets(config)
    elif args.stage == "map2":
        run_map2_stage(config)
    elif args.stage == "loops":
        run_loop_stage(config, force=args.force)
    elif args.stage == "smoke":
        run_validation(config)
        run_subsets(config)
        run_loop_stage(config, smoke=True, force=args.force)
        run_map2_stage(config, smoke=True)
    elif args.stage == "all":
        run_validation(config)
        run_subsets(config)
        run_loop_stage(config, force=args.force)
        run_map2_stage(config)


if __name__ == "__main__":
    main()
