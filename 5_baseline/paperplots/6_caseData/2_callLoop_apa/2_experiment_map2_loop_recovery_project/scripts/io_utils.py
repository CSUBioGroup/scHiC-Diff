"""Input/output helpers for Map2 known-loop recovery analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("map2_loop_recovery")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def read_region(region_file: Path) -> tuple[str, int, int]:
    lines = [line.strip() for line in region_file.read_text().splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"Region file must contain chrom, start, end on separate lines: {region_file}")
    chrom = lines[0]
    start = int(lines[1])
    end = int(lines[2])
    if end <= start:
        raise ValueError(f"Region end must be greater than start: {region_file}")
    return chrom, start, end


def read_loop_bedpe(loop_bedpe: Path) -> pd.DataFrame:
    if not loop_bedpe.exists():
        raise FileNotFoundError(
            f"Known loop BEDPE not found: {loop_bedpe}. "
            "Provide Map2_known_loop.bedpe or an explicit candidate loop file."
        )
    df = pd.read_csv(loop_bedpe, sep=r"\s+", header=None, comment="#")
    if df.shape[1] < 6:
        raise ValueError(f"BEDPE must contain at least six columns: {loop_bedpe}")
    df = df.iloc[:, :6].copy()
    df.columns = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    for col in ["start1", "end1", "start2", "end2"]:
        df[col] = df[col].astype(int)
    return df


def load_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")
    matrix = np.load(path)
    if matrix.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got shape {matrix.shape}: {path}")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {matrix.shape}: {path}")
    return matrix.astype(float, copy=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def ensure_output_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

