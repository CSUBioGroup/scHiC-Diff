#!/usr/bin/env python3
"""Prepare, run, reconcile, and plot corrected Juicer APA jobs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = BENCHMARK_DIR / ".mplconfig"
XDG_CACHE_HOME = BENCHMARK_DIR / ".cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
os.environ["XDG_CACHE_HOME"] = str(XDG_CACHE_HOME)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_apa import (
    build_juicer_command,
    load_loop_bedpe,
    prepare_apa_set,
)
from run_benchmark import load_resolved_config, sha256_file


def parse_measures(path: str | Path) -> dict[str, float]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    measures: dict[str, float] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2:
            try:
                measures[fields[0]] = float(fields[1])
            except ValueError:
                continue
    return measures


def count_enhancement_values(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    values = []
    for token in path.read_text().split():
        value = float(token)
        if not np.isfinite(value):
            raise ValueError(f"enhancement file contains non-finite value: {path}")
        values.append(value)
    return len(values)


def reconcile_counts(
    written_count: int,
    parsed_count: int,
    effective_count: int,
) -> dict[str, object]:
    if parsed_count != written_count:
        raise ValueError(
            f"parsed BEDPE count {parsed_count} does not match written count {written_count}"
        )
    if effective_count > parsed_count:
        raise ValueError(
            f"effective APA count {effective_count} exceeds parsed count {parsed_count}"
        )
    return {
        "parsed_count": int(parsed_count),
        "effective_count": int(effective_count),
        "count_status": "exact" if effective_count == parsed_count else "effective_subset",
    }


def _count_bedpe_data_rows(path: Path) -> int:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return max(0, len(lines) - 1)


def _result_directory(apa_dir: Path, resolution: int) -> Path:
    chromosome = apa_dir / str(resolution) / "chr1vchr1"
    if chromosome.exists():
        return chromosome
    genome_wide = apa_dir / str(resolution) / "gw"
    if genome_wide.exists():
        return genome_wide
    raise FileNotFoundError(f"Juicer result directory not found under {apa_dir}")


def _plot_apa(result_dir: Path, output_path: Path, title: str) -> None:
    matrix_path = result_dir / "APA.npy"
    if not matrix_path.exists():
        matrix_path = result_dir / "normedAPA.npy"
    matrix = np.load(matrix_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    image = axis.imshow(matrix, cmap="afmhot_r", origin="lower", interpolation="nearest")
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("APA window bins")
    axis.set_ylabel("APA window bins")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=250)
    plt.close(figure)


def run_apa_job(
    record: dict[str, object],
    apa_config: dict,
    force: bool = False,
) -> dict[str, object]:
    if record.get("status") == "no_eligible_loops" or int(record.get("written_count", 0)) == 0:
        return {
            **record,
            "status": "no_eligible_loops",
            "parsed_count": 0,
            "effective_count": 0,
            "count_status": "no_eligible_loops",
            "P2LL": None,
        }

    bedpe_path = Path(str(record["bedpe_path"]))
    apa_dir = Path(str(record["apa_dir"]))
    figure_path = Path(str(record["figure_path"]))
    status_path = apa_dir / "job_status.json"
    bedpe_sha256 = sha256_file(bedpe_path)
    if status_path.exists() and not force:
        previous = json.loads(status_path.read_text())
        if previous.get("status") == "completed" and previous.get("bedpe_sha256") == bedpe_sha256:
            return previous

    apa_dir.mkdir(parents=True, exist_ok=True)
    command = build_juicer_command(
        java=apa_config["java_bin"],
        java_options=apa_config["java_options"],
        jar=apa_config["juicer_jar"],
        hic=apa_config["reference_hic"],
        bedpe=bedpe_path,
        output=apa_dir,
        resolution=int(apa_config["resolution"]),
        norm=apa_config["normalization"],
        window=int(apa_config["window_bins"]),
        min_distance_bins=int(record["min_distance_bins"]),
    )
    with (apa_dir / "juicer.log").open("w") as log_handle:
        subprocess.run(command, check=True, stdout=log_handle, stderr=subprocess.STDOUT, text=True)

    result_dir = _result_directory(apa_dir, int(apa_config["resolution"]))
    measures = parse_measures(result_dir / "measures.txt")
    effective_count = count_enhancement_values(result_dir / "enhancement.txt")
    parsed_count = _count_bedpe_data_rows(bedpe_path)
    reconciliation = reconcile_counts(
        written_count=int(record["written_count"]),
        parsed_count=parsed_count,
        effective_count=effective_count,
    )
    result = {
        **record,
        **reconciliation,
        "status": "completed",
        "bedpe_sha256": bedpe_sha256,
        "P2LL": measures.get("P2LL"),
        "P2M": measures.get("P2M"),
        "ZscoreLL": measures.get("ZscoreLL"),
        "result_dir": str(result_dir.resolve()),
        "command": command,
    }
    title = (
        f"{record['method_name']} {record['set_label']} | min={record['min_distance_bins']} bins\n"
        f"written={record['written_count']}, effective={effective_count}, P2LL={measures.get('P2LL', float('nan')):.3f}"
    )
    _plot_apa(result_dir, figure_path, title)
    status_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def prepare_jobs(config: dict) -> list[dict[str, object]]:
    apa_root = config["output_root"] / "apa"
    jobs: list[dict[str, object]] = []
    for method in config["methods"]:
        source = (
            config["output_root"]
            / "loops"
            / method["slug"]
            / "earlyNeurons"
            / "476cells_seed42"
            / "loops.loop.bedpe"
        )
        loops = load_loop_bedpe(source)
        for min_distance in config["apa"]["minimum_distance_bins"]:
            branch = apa_root / f"min{min_distance}bins"
            set_specs = [("all", None)] + [
                (f"top{top_n}", int(top_n)) for top_n in config["apa"]["top_n_values"]
            ]
            for set_label, top_n in set_specs:
                bedpe_path = branch / "bedpe" / f"{method['slug']}_{set_label}.bedpe"
                record = prepare_apa_set(
                    loops,
                    output_path=bedpe_path,
                    resolution=int(config["apa"]["resolution"]),
                    min_distance_bins=int(min_distance),
                    top_n=top_n,
                )
                record.update(
                    {
                        "method": method["slug"],
                        "method_name": method["name"],
                        "set_label": set_label,
                        "apa_dir": str((branch / "runs" / f"{method['slug']}_{set_label}").resolve()),
                        "figure_path": str(
                            (branch / "figures" / f"APA_{method['slug']}_{set_label}.png").resolve()
                        ),
                    }
                )
                jobs.append(record)
    apa_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(jobs).to_csv(apa_root / "apa_jobs_prepared.csv", index=False)
    return jobs


def prepare_legacy_comparison_jobs(
    config: dict,
    output_root: str | Path | None = None,
) -> list[dict[str, object]]:
    """Prepare corrected APA sets from the two legacy 476-cell loop collections."""
    workspace = config["config_path"].parents[2]
    sources = [
        {
            "slug": "legacy_schicdiff",
            "name": "Legacy scHiC-Diff loops",
            "path": workspace
            / "1_experiment_loop_compare/results/output_schicdiff/earlyNeurons_476cells/loops.loop.bedpe",
        },
        {
            "slug": "legacy_schicluster",
            "name": "Legacy scHiCluster loops",
            "path": workspace
            / "1_experiment_loop_compare/results/output_schicluster/earlyNeurons_476cells/loops.loop.bedpe",
        },
    ]
    apa_root = (
        Path(output_root)
        if output_root is not None
        else config["output_root"] / "apa_legacy_loops_corrected"
    )
    jobs: list[dict[str, object]] = []
    for source in sources:
        loops = load_loop_bedpe(source["path"])
        for min_distance in config["apa"]["minimum_distance_bins"]:
            branch = apa_root / f"min{min_distance}bins"
            set_specs = [("all", None)] + [
                (f"top{top_n}", int(top_n)) for top_n in config["apa"]["top_n_values"]
            ]
            for set_label, top_n in set_specs:
                record = prepare_apa_set(
                    loops,
                    output_path=branch / "bedpe" / f"{source['slug']}_{set_label}.bedpe",
                    resolution=int(config["apa"]["resolution"]),
                    min_distance_bins=int(min_distance),
                    top_n=top_n,
                )
                record.update(
                    {
                        "method": source["slug"],
                        "method_name": source["name"],
                        "set_label": set_label,
                        "source_kind": "legacy_loop_set",
                        "apa_dir": str((branch / "runs" / f"{source['slug']}_{set_label}").resolve()),
                        "figure_path": str(
                            (branch / "figures" / f"APA_{source['slug']}_{set_label}.png").resolve()
                        ),
                    }
                )
                jobs.append(record)
    apa_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(jobs).to_csv(apa_root / "apa_jobs_prepared.csv", index=False)
    return jobs


def run_all_apa(config: dict, force: bool = False) -> pd.DataFrame:
    jobs = prepare_jobs(config)
    results = []
    for job in jobs:
        try:
            results.append(run_apa_job(job, config["apa"], force=force))
        except Exception as error:
            failed = {**job, "status": "failed", "error": f"{type(error).__name__}: {error}"}
            results.append(failed)
            pd.DataFrame.from_records(results).to_csv(
                config["output_root"] / "apa" / "apa_manifest.partial.csv", index=False
            )
            raise
    frame = pd.DataFrame.from_records(results)
    frame.to_csv(config["output_root"] / "apa" / "apa_manifest.csv", index=False)
    return frame


def run_legacy_comparison(config: dict, force: bool = False) -> pd.DataFrame:
    jobs = prepare_legacy_comparison_jobs(config)
    results = [run_apa_job(job, config["apa"], force=force) for job in jobs]
    frame = pd.DataFrame.from_records(results)
    output = config["output_root"] / "apa_legacy_loops_corrected" / "apa_manifest.csv"
    frame.to_csv(output, index=False)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--legacy-comparison", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_resolved_config(args.config)
    manifest = run_all_apa(config, force=args.force)
    print(manifest["status"].value_counts().to_string())
    if args.legacy_comparison:
        legacy = run_legacy_comparison(config, force=args.force)
        print("legacy comparison")
        print(legacy["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
