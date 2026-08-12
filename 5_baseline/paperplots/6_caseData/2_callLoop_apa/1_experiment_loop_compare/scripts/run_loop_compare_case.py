#!/usr/bin/env python3
"""Run the loop-comparison case from a JSON config."""

import argparse
import json
from pathlib import Path

import numpy as np

from call_loops_from_npz import run_case
from plot_comparison_grid import plot_grid


def load_config(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    return config_path, config


def resolve_path(config_dir, raw_path):
    raw = Path(raw_path)
    if raw.is_absolute():
        return raw
    return (config_dir / raw).resolve()


def read_total_cells(npz_path):
    payload = np.load(npz_path, allow_pickle=True)
    return int(payload["shape"][0])


def build_shared_subsets(reference_npz, cell_counts, seed, subset_dir):
    subset_dir.mkdir(parents=True, exist_ok=True)
    total_cells = read_total_cells(reference_npz)
    subset_files = {}
    for count in cell_counts:
        if count >= total_cells:
            indices = np.arange(total_cells, dtype=int)
        else:
            rng = np.random.RandomState(seed)
            indices = np.sort(rng.choice(total_cells, count, replace=False).astype(int))
        subset_path = subset_dir / f"selected_indices_{count}cells.npy"
        np.save(subset_path, indices)
        subset_files[count] = subset_path
    return subset_files


def main():
    parser = argparse.ArgumentParser(description="Run a loop-comparison case")
    parser.add_argument("--config", required=True, help="Path to a JSON config")
    args = parser.parse_args()

    config_path, config = load_config(args.config)
    config_dir = config_path.parent

    cell_type = config["cell_type"]
    cell_counts = config["cell_counts"]
    seed = config.get("seed", 42)
    resolution = config.get("resolution", 20000)

    methods = []
    for method in config["methods"]:
        methods.append(
            {
                "name": method["name"],
                "slug": method["slug"],
                "input_npz": resolve_path(config_dir, method["input_npz"]),
                "output_dir": resolve_path(config_dir, method["output_dir"]),
            }
        )

    reference_slug = config.get("reference_method_slug", methods[0]["slug"])
    reference_method = next(method for method in methods if method["slug"] == reference_slug)
    shared_subset_dir = resolve_path(config_dir, config["shared_subset_dir"])
    subset_files = build_shared_subsets(
        reference_npz=reference_method["input_npz"],
        cell_counts=cell_counts,
        seed=seed,
        subset_dir=shared_subset_dir,
    )

    for method in methods:
        print(f"Running method: {method['name']}")
        for count in cell_counts:
            run_case(
                input_npz=method["input_npz"],
                output_dir=method["output_dir"],
                cell_type=cell_type,
                n_cells=count,
                seed=seed,
                selected_indices_file=subset_files[count],
                resolution=resolution,
                n_bins=config.get("n_bins"),
                triu_k=method.get("triu_k", config.get("triu_k")),
                min_dist=config.get("min_dist", 60000),
                max_dist=config.get("max_dist", 2000000),
                cap=config.get("cap", 5),
                pad=config.get("pad", 5),
                gap=config.get("gap", 2),
                fdr=config.get("fdr", 0.05),
                dist_thres=config.get("dist_thres", 40000),
                size_thres=config.get("size_thres", 1),
            )

    plot_methods = [{"name": method["name"], "base_dir": str(method["output_dir"])} for method in methods]
    plots = config["plots"]
    plot_grid(
        methods=plot_methods,
        cell_counts=cell_counts,
        cell_type=cell_type,
        output_file=resolve_path(config_dir, plots["no_loop"]),
        resolution=resolution,
        see_loop=False,
    )

    if "with_loop" in plots:
        plot_grid(
            methods=plot_methods,
            cell_counts=cell_counts,
            cell_type=cell_type,
            output_file=resolve_path(config_dir, plots["with_loop"]),
            resolution=resolution,
            see_loop=True,
        )


if __name__ == "__main__":
    main()
