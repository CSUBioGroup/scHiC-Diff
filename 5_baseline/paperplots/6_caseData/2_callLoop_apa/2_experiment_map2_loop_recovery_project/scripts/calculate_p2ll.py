"""P2LL and center enrichment calculation for a known Map2 loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

from config import EPSILON


@dataclass(frozen=True)
class LoopBin:
    loop_id: int
    chrom1: str
    start1: int
    end1: int
    chrom2: str
    start2: int
    end2: int
    anchor1_midpoint: int
    anchor2_midpoint: int
    bin1: int
    bin2: int


def genomic_to_bin(position: int, region_start: int, resolution: int) -> int:
    return int((position - region_start) // resolution)


def convert_loops_to_bins(
    loop_df: pd.DataFrame,
    region_chrom: str,
    region_start: int,
    region_end: int,
    resolution: int,
    matrix_size: int,
) -> pd.DataFrame:
    records = []
    for idx, row in loop_df.iterrows():
        if row["chrom1"] != region_chrom or row["chrom2"] != region_chrom:
            raise ValueError(
                f"Loop {idx} is outside region chromosome {region_chrom}: "
                f"{row['chrom1']} / {row['chrom2']}"
            )
        mid1 = int((row["start1"] + row["end1"]) // 2)
        mid2 = int((row["start2"] + row["end2"]) // 2)
        if not (region_start <= mid1 < region_end and region_start <= mid2 < region_end):
            raise ValueError(
                f"Loop {idx} midpoint outside region {region_chrom}:{region_start}-{region_end}: "
                f"{mid1}, {mid2}"
            )
        bin1 = genomic_to_bin(mid1, region_start, resolution)
        bin2 = genomic_to_bin(mid2, region_start, resolution)
        if not (0 <= bin1 < matrix_size and 0 <= bin2 < matrix_size):
            raise ValueError(f"Loop {idx} bin outside matrix range 0-{matrix_size - 1}: {bin1}, {bin2}")
        records.append(
            LoopBin(
                loop_id=idx + 1,
                chrom1=row["chrom1"],
                start1=int(row["start1"]),
                end1=int(row["end1"]),
                chrom2=row["chrom2"],
                start2=int(row["start2"]),
                end2=int(row["end2"]),
                anchor1_midpoint=mid1,
                anchor2_midpoint=mid2,
                bin1=bin1,
                bin2=bin2,
            ).__dict__
        )
    return pd.DataFrame.from_records(records)


def maybe_log1p_transform(matrix: np.ndarray, threshold: float, logger: logging.Logger, matrix_name: str) -> tuple[np.ndarray, bool]:
    max_value = float(np.nanmax(matrix))
    if max_value < threshold:
        logger.info("%s appears log-transformed already; max=%.6g < %.6g", matrix_name, max_value, threshold)
        return matrix.copy(), False
    logger.info("%s appears untransformed; applying log1p; max=%.6g >= %.6g", matrix_name, max_value, threshold)
    return np.log1p(matrix), True


def _window_bounds(center: int, radius: int, size: int) -> tuple[int, int]:
    return max(0, center - radius), min(size, center + radius + 1)


def extract_center_signal(matrix: np.ndarray, bin1: int, bin2: int, window_size: int) -> float:
    radius = window_size // 2
    r0, r1 = _window_bounds(bin1, radius, matrix.shape[0])
    c0, c1 = _window_bounds(bin2, radius, matrix.shape[1])
    return float(np.nanmean(matrix[r0:r1, c0:c1]))


def extract_background_signal(
    matrix: np.ndarray,
    bin1: int,
    bin2: int,
    background_window: int,
    center_window: int,
) -> float:
    bg_radius = background_window // 2
    center_radius = center_window // 2
    r0, r1 = _window_bounds(bin1, bg_radius, matrix.shape[0])
    c0, c1 = _window_bounds(bin2, bg_radius, matrix.shape[1])
    window = matrix[r0:r1, c0:c1].copy()

    center_r0 = max(0, bin1 - center_radius) - r0
    center_r1 = min(matrix.shape[0], bin1 + center_radius + 1) - r0
    center_c0 = max(0, bin2 - center_radius) - c0
    center_c1 = min(matrix.shape[1], bin2 + center_radius + 1) - c0
    window[center_r0:center_r1, center_c0:center_c1] = np.nan
    return float(np.nanmean(window))


def calculate_loop_metrics(
    matrix: np.ndarray,
    loop_bins: pd.DataFrame,
    center_window: int,
    background_window: int,
    epsilon: float = EPSILON,
) -> pd.DataFrame:
    records = []
    for _, loop in loop_bins.iterrows():
        bin1 = int(loop["bin1"])
        bin2 = int(loop["bin2"])
        center_signal = extract_center_signal(matrix, bin1, bin2, center_window)
        background_signal = extract_background_signal(matrix, bin1, bin2, background_window, center_window)
        denominator = max(background_signal, epsilon)
        p2ll = center_signal / denominator
        log2_enrichment = float(np.log2((center_signal + epsilon) / denominator))
        records.append(
            {
                "loop_id": int(loop["loop_id"]),
                "bin1": bin1,
                "bin2": bin2,
                "center_signal": center_signal,
                "background_signal": background_signal,
                "P2LL": p2ll,
                "log2_enrichment": log2_enrichment,
            }
        )
    return pd.DataFrame.from_records(records)
