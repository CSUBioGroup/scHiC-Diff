"""Prepare corrected, headered Juicer APA loop sets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BEDPE_COLUMNS = ["chrom1", "start1", "end1", "chrom2", "start2", "end2", "name", "score"]


def load_loop_bedpe(path: str | Path, chrom: str = "chr1") -> pd.DataFrame:
    """Load the legacy five-column caller BEDPE or a corrected eight-column BEDPE."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    first_line = path.read_text().splitlines()[0] if path.stat().st_size else ""
    if not first_line:
        return pd.DataFrame(columns=BEDPE_COLUMNS)
    first_fields = first_line.split("\t")
    if first_fields == BEDPE_COLUMNS:
        frame = pd.read_csv(path, sep="\t")
        return frame[BEDPE_COLUMNS].copy()

    raw = pd.read_csv(path, sep=r"\s+", header=None, comment="#", engine="python")
    if raw.shape[1] < 5:
        raise ValueError(f"expected at least five columns in {path}, got {raw.shape[1]}")
    output = pd.DataFrame(
        {
            "chrom1": chrom,
            "start1": raw.iloc[:, 0].astype(int),
            "end1": raw.iloc[:, 1].astype(int),
            "chrom2": chrom,
            "start2": raw.iloc[:, 2].astype(int),
            "end2": raw.iloc[:, 3].astype(int),
            "name": [f"loop_{idx + 1}" for idx in range(len(raw))],
            "score": pd.to_numeric(raw.iloc[:, 4], errors="raise"),
        }
    )
    return output[BEDPE_COLUMNS]


def filter_and_rank_loops(
    frame: pd.DataFrame,
    resolution: int,
    min_distance_bins: int,
    top_n: int | None = None,
) -> pd.DataFrame:
    if resolution <= 0 or min_distance_bins < 0:
        raise ValueError("resolution must be positive and minimum distance non-negative")
    missing = [column for column in BEDPE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"loop table is missing columns: {missing}")
    working = frame[BEDPE_COLUMNS].copy()
    distances = np.abs(working["start2"].to_numpy() - working["start1"].to_numpy())
    working["distance_bins"] = distances / resolution
    working = working.loc[working["distance_bins"] >= min_distance_bins]
    working = working.sort_values("score", ascending=False, kind="mergesort")
    if top_n is not None:
        if top_n <= 0:
            raise ValueError("top_n must be positive when provided")
        working = working.head(top_n)
    return working[BEDPE_COLUMNS].reset_index(drop=True)


def write_juicer_bedpe(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [column for column in BEDPE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"loop table is missing columns: {missing}")
    frame[BEDPE_COLUMNS].to_csv(path, sep="\t", header=True, index=False)
    return path


def prepare_apa_set(
    frame: pd.DataFrame,
    output_path: str | Path,
    resolution: int,
    min_distance_bins: int,
    top_n: int | None,
) -> dict[str, object]:
    eligible = filter_and_rank_loops(
        frame,
        resolution=resolution,
        min_distance_bins=min_distance_bins,
        top_n=None,
    )
    selected = eligible if top_n is None else eligible.head(top_n).reset_index(drop=True)
    output_path = write_juicer_bedpe(selected, output_path)
    return {
        "status": "ready" if len(selected) else "no_eligible_loops",
        "source_count": int(len(frame)),
        "eligible_count": int(len(eligible)),
        "requested_top_n": top_n,
        "written_count": int(len(selected)),
        "min_distance_bins": int(min_distance_bins),
        "bedpe_path": str(output_path.resolve()),
    }


def build_juicer_command(
    java: str | Path,
    java_options: list[str],
    jar: str | Path,
    hic: str | Path,
    bedpe: str | Path,
    output: str | Path,
    resolution: int,
    norm: str,
    window: int,
    min_distance_bins: int,
) -> list[str]:
    return [
        str(java),
        *[str(option) for option in java_options],
        "-jar",
        str(jar),
        "apa",
        "-n",
        str(min_distance_bins),
        "-r",
        str(resolution),
        "-k",
        str(norm),
        "-u",
        "-w",
        str(window),
        str(hic),
        str(bedpe),
        str(output),
    ]
