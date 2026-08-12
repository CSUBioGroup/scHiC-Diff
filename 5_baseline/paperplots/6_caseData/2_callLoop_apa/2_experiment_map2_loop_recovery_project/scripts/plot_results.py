"""Publication-ready matplotlib plotting for Map2 loop recovery results."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

from config import CELL_NUMBERS, METHODS


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def plot_metric(
    results: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_pdf: Path,
    title: str,
) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    for method, meta in METHODS.items():
        subset = results[results["method"] == method].sort_values("cell_number")
        if subset.empty:
            continue
        ax.plot(
            subset["cell_number"],
            subset[metric],
            marker=meta["marker"],
            markersize=4.5,
            linewidth=1.5,
            color=meta["color"],
            label=meta["label"],
        )

    ax.set_xscale("log")
    ax.set_xticks(CELL_NUMBERS)
    ax.set_xticklabels([str(n) for n in CELL_NUMBERS])
    ax.set_xlabel("Aggregated cells")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_all(results: pd.DataFrame, output_dir: Path) -> None:
    plot_metric(
        results,
        metric="P2LL",
        ylabel="P2LL score",
        output_pdf=output_dir / "Map2_known_loop_P2LL.pdf",
        title="Map2 known loop recovery",
    )
    plot_metric(
        results,
        metric="log2_enrichment",
        ylabel="Loop center enrichment (log2)",
        output_pdf=output_dir / "Map2_known_loop_enrichment.pdf",
        title="Map2 loop center enrichment",
    )
