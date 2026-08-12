#!/usr/bin/env python3
"""Plot the loop-comparison heatmap grid."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from scipy import sparse


def infer_n_bins(n_features, triu_k=None):
    """Infer matrix dimension from upper-triangle feature count. Returns (n_bins, triu_k)."""
    if triu_k is not None:
        if triu_k == 0:
            n_bins = int((-1 + np.sqrt(1 + 8 * n_features)) / 2)
            if n_bins * (n_bins + 1) // 2 != n_features:
                raise ValueError(f"n_features={n_features} is not valid triu(k=0)")
            return n_bins, 0
        elif triu_k == 1:
            n_bins = int((1 + np.sqrt(1 + 8 * n_features)) / 2)
            if n_bins * (n_bins - 1) // 2 != n_features:
                raise ValueError(f"n_features={n_features} is not valid triu(k=1)")
            return n_bins, 1
        else:
            raise ValueError(f"triu_k must be 0 or 1, got {triu_k}")
    n_bins_k0 = int((-1 + np.sqrt(1 + 8 * n_features)) / 2)
    if n_bins_k0 * (n_bins_k0 + 1) // 2 == n_features:
        return n_bins_k0, 0
    n_bins_k1 = int((1 + np.sqrt(1 + 8 * n_features)) / 2)
    if n_bins_k1 * (n_bins_k1 - 1) // 2 == n_features:
        return n_bins_k1, 1
    raise ValueError(f"Cannot infer a valid upper-triangle matrix size from n_features={n_features}")


def flat_to_matrix(flat, n_bins, triu_k=0):
    mat = np.zeros((n_bins, n_bins), dtype=np.float32)
    idx = np.triu_indices(n_bins, k=triu_k)
    mat[idx] = flat
    mat = mat + mat.T - np.diag(np.diag(mat))
    return mat


def load_selected_cells_mean_matrix(npz_path):
    payload = np.load(npz_path, allow_pickle=True)
    mat = sparse.csr_matrix(
        (payload["data"], payload["indices"], payload["indptr"]),
        shape=tuple(payload["shape"]),
    )
    dense = mat.toarray()
    avg_flat = dense.mean(axis=0)
    n_bins, triu_k = infer_n_bins(avg_flat.shape[0])
    return flat_to_matrix(avg_flat, n_bins=n_bins, triu_k=triu_k)


def load_loops(csv_path, resolution):
    df = pd.read_csv(csv_path)
    if "x1" not in df.columns or "y1" not in df.columns:
        return None, None
    return df["x1"].values // resolution, df["y1"].values // resolution


def plot_grid(methods, cell_counts, cell_type, output_file, resolution=20000, see_loop=False, vmax_percentile=98):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(methods)
    n_cols = len(cell_counts)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3, n_rows * 3.5),
        gridspec_kw={"hspace": -0.1, "wspace": 0.05},
    )

    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, method in enumerate(methods):
        base_path = Path(method["base_dir"])
        for col_idx, n_cells in enumerate(cell_counts):
            ax = axes[row_idx, col_idx]
            subdir = base_path / f"{cell_type}_{n_cells}cells"
            selected_npz = subdir / "selected_cells.npz"
            loop_csv = subdir / "loops.loop_info.csv"

            if selected_npz.exists():
                matrix = load_selected_cells_mean_matrix(selected_npz)
                vmax = np.percentile(matrix, vmax_percentile)
                if vmax == 0:
                    vmax = 1
                image = ax.imshow(matrix, cmap="Reds", origin="upper", vmin=0, vmax=vmax)

                cax = inset_axes(
                    ax,
                    width="30%",
                    height="5%",
                    loc="lower left",
                    bbox_to_anchor=(0.1, -0.10, 1, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0,
                )
                cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
                cbar.set_ticks([])
                cax.text(-0.1, 0.5, "0", ha="right", va="center", transform=cax.transAxes, fontsize=12)
                cax.text(1.1, 0.5, f"{vmax:.1f}", ha="left", va="center", transform=cax.transAxes, fontsize=12)

                if see_loop and loop_csv.exists():
                    loop_x, loop_y = load_loops(loop_csv, resolution=resolution)
                    if loop_x is not None:
                        ax.scatter(loop_y, loop_x, s=10, facecolors="none", edgecolors="black", linewidths=1.5)
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center")
                ax.set_facecolor("#f0f0f0")

            if col_idx == 0:
                ax.set_ylabel(method["name"], fontsize=14, fontweight="bold")
            if row_idx == 0:
                ax.set_title(f"{n_cells} cells", fontsize=14, fontweight="bold")

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1)

    plt.savefig(output_file, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison grid to {output_file}")


def parse_method_arg(raw):
    if "=" not in raw:
        raise ValueError(f"Method must look like Label=/path/to/output_dir: {raw}")
    name, path = raw.split("=", 1)
    return {"name": name, "base_dir": path}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a loop-comparison grid from selected_cells.npz outputs")
    parser.add_argument("--method", action="append", required=True, help="Repeated Label=/path/to/method_output")
    parser.add_argument("--cell-type", default="earlyNeurons")
    parser.add_argument("--cell-counts", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, default=20000)
    parser.add_argument("--see-loop", action="store_true", help="Overlay called loops")
    return parser.parse_args()


def main():
    args = parse_args()
    methods = [parse_method_arg(item) for item in args.method]
    plot_grid(
        methods=methods,
        cell_counts=args.cell_counts,
        cell_type=args.cell_type,
        output_file=args.output,
        resolution=args.resolution,
        see_loop=args.see_loop,
    )


if __name__ == "__main__":
    main()
