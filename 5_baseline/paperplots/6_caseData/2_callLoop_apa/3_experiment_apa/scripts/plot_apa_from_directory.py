#!/usr/bin/env python3
"""Plot one APA heatmap from a Juicer APA output directory."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_p2ll(measures_file):
    if not measures_file.exists():
        return None
    for line in measures_file.read_text().splitlines():
        if line.strip().startswith("P2LL"):
            try:
                return float(line.split()[1])
            except (ValueError, IndexError):
                return None
    return None


def plot_apa(apa_dir, title, output_png, resolution=20000, cmap="afmhot_r"):
    apa_dir = Path(apa_dir)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    matrix_path = apa_dir / "APA.npy"
    if not matrix_path.exists():
        alt = apa_dir / "normedAPA.npy"
        if alt.exists():
            matrix_path = alt
        else:
            raise FileNotFoundError(f"Neither APA.npy nor normedAPA.npy found under {apa_dir}")

    apa = np.load(matrix_path)
    p2ll = read_p2ll(apa_dir / "measures.txt")

    n_bins = apa.shape[0]
    half_width = (n_bins * resolution) / 2
    extent = (-half_width, half_width, -half_width, half_width)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    image = ax.imshow(
        apa,
        cmap=plt.get_cmap(cmap),
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
    )

    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    start_center = -half_width + (resolution / 2)
    end_center = half_width - (resolution / 2)
    ticks = np.arange(start_center, end_center + 1, resolution)
    tick_labels = []
    for tick in ticks:
        value_k = int(tick / 1000)
        if value_k == 0:
            tick_labels.append("0")
        elif tick % 200000 == 0 or abs(tick) == abs(end_center):
            tick_labels.append(f"{value_k}K")
        else:
            tick_labels.append("")

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=10)
    ax.tick_params(axis="both", which="major", length=4, width=1)

    final_title = title
    if p2ll is not None:
        final_title += f"\nAPA (P2LL)={p2ll:.3f}"
    ax.set_title(final_title, fontsize=14)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(output_png, dpi=400)
    plt.close(fig)
    print(f"Saved APA plot to {output_png}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot one APA heatmap from a Juicer output directory")
    parser.add_argument("--apa-dir", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, default=20000)
    parser.add_argument("--cmap", default="afmhot_r")
    return parser.parse_args()


def main():
    args = parse_args()
    plot_apa(args.apa_dir, args.title, args.output, resolution=args.resolution, cmap=args.cmap)


if __name__ == "__main__":
    main()
