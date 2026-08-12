"""Configuration defaults for Map2 known-loop recovery analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


METHODS = {
    "raw": {
        "label": "Raw scHi-C",
        "color": "#4D4D4D",
        "marker": "o",
        "data_subdir": "raw",
        "file_pattern": "Map2_raw_n{n}.npy",
    },
    "scHiCluster": {
        "label": "scHiCluster",
        "color": "#1F77B4",
        "marker": "s",
        "data_subdir": "scHiCluster",
        "file_pattern": "Map2_scHiCluster_n{n}.npy",
    },
    "scHiCDiff": {
        "label": "scHiC-Diff",
        "color": "#D62728",
        "marker": "^",
        "data_subdir": "scHiCDiff",
        "file_pattern": "Map2_scHiCDiff_n{n}.npy",
    },
}

CELL_NUMBERS = [3, 5, 10, 100, 200, 476]

DEFAULT_RESOLUTION = 20_000
DEFAULT_CENTER_WINDOW = 5
DEFAULT_BACKGROUND_WINDOW = 20
LOG_TRANSFORM_THRESHOLD = 20.0
EPSILON = 1e-12


@dataclass(frozen=True)
class AnalysisConfig:
    data_dir: Path
    loop_bedpe: Path
    region_file: Path
    output_dir: Path
    resolution: int = DEFAULT_RESOLUTION
    center_window: int = DEFAULT_CENTER_WINDOW
    background_window: int = DEFAULT_BACKGROUND_WINDOW
    log_transform_threshold: float = LOG_TRANSFORM_THRESHOLD
    epsilon: float = EPSILON

