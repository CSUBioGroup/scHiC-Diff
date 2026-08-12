#!/usr/bin/env python3
"""Run Juicer APA for one case config and render the final figures."""

import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd

from convert_loops_to_juicer_bedpe import convert_single
from plot_apa_from_directory import plot_apa


def resolve_path(config_dir, raw_path):
    raw = Path(raw_path)
    if raw.is_absolute():
        return raw
    return (config_dir / raw).resolve()


def load_config(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    return config_path, config


def extract_top_n(in_bedpe, out_bedpe, top_n):
    df = pd.read_csv(in_bedpe, sep=r"\s+", header=None, comment="#", engine="python")
    numeric_cols = [idx for idx in df.columns if pd.api.types.is_numeric_dtype(df[idx])]
    if not numeric_cols:
        raise ValueError(f"No numeric score column found in {in_bedpe}")
    score_col = numeric_cols[-1]
    df.sort_values(by=score_col, ascending=False).head(top_n).to_csv(
        out_bedpe,
        sep="\t",
        header=False,
        index=False,
    )


def run_juicer_apa(juicer_jar, reference_hic, bedpe_file, output_dir, resolution, norm, window_bins, java_bin, java_options):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(java_bin),
        *java_options,
        "-jar",
        str(juicer_jar),
        "apa",
        "-r",
        str(resolution),
        "-k",
        str(norm),
        "-u",
        "-w",
        str(window_bins),
        str(reference_hic),
        str(bedpe_file),
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run a Juicer APA case from JSON config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--plot-only", action="store_true", help="Skip Juicer APA and only render plots from existing outputs")
    args = parser.parse_args()

    config_path, config = load_config(args.config)
    config_dir = config_path.parent

    juicer_jar = resolve_path(config_dir, config["juicer_jar"])
    reference_hic = resolve_path(config_dir, config["reference_hic"])
    java_bin = config.get("java_bin", "java")
    java_options = config.get("java_options", ["-Djava.awt.headless=true", "-Xmx8g"])
    resolution = config.get("resolution", 20000)
    norm = config.get("norm", "NONE")
    window_bins = config.get("window_bins", 10)
    output_root = resolve_path(config_dir, config["output_root"])
    bedpe_root = output_root / "bedpe_juicer"
    apa_root = output_root / "apa_runs"
    final_plot_dir = output_root / "final"

    for method in config["methods"]:
        method_name = method["name"]
        slug = method["slug"]
        sample_name = method["sample_name"]
        loop_dir = resolve_path(config_dir, method["loop_result_dir"])
        loop_bedpe = loop_dir / sample_name / "loops.loop.bedpe"
        juicer_bedpe = bedpe_root / f"{slug}_{sample_name}.bedpe"
        convert_single(loop_bedpe, juicer_bedpe, chrom=config.get("chrom", "chr1"))

        apa_jobs = [
            {
                "label": "all",
                "bedpe": juicer_bedpe,
                "apa_dir": apa_root / f"{slug}_{sample_name}",
                "title": method["plot_titles"]["all"],
                "png": final_plot_dir / f"APA_{method['plot_titles']['all']}.png",
            }
        ]

        for top_n in config.get("top_n_values", [50, 100]):
            top_bedpe = bedpe_root / f"{slug}_{sample_name}.top{top_n}.bedpe"
            extract_top_n(juicer_bedpe, top_bedpe, top_n=top_n)
            apa_jobs.append(
                {
                    "label": f"top{top_n}",
                    "bedpe": top_bedpe,
                    "apa_dir": apa_root / f"{slug}_apa_top{top_n}",
                    "title": method["plot_titles"][f"top{top_n}"],
                    "png": final_plot_dir / f"APA_{method['plot_titles'][f'top{top_n}']}.png",
                }
            )

        for job in apa_jobs:
            if not args.plot_only:
                print(f"Running APA for {method_name} [{job['label']}]")
                run_juicer_apa(
                    juicer_jar=juicer_jar,
                    reference_hic=reference_hic,
                    bedpe_file=job["bedpe"],
                    output_dir=job["apa_dir"],
                    resolution=resolution,
                    norm=norm,
                    window_bins=window_bins,
                    java_bin=java_bin,
                    java_options=java_options,
                )
            chr_dir = job["apa_dir"] / str(resolution) / "chr1vchr1"
            if not chr_dir.exists():
                chr_dir = job["apa_dir"] / str(resolution) / "gw"
            plot_apa(
                apa_dir=chr_dir,
                title=job["title"],
                output_png=job["png"],
                resolution=resolution,
                cmap=config.get("cmap", "afmhot_r"),
            )


if __name__ == "__main__":
    main()
