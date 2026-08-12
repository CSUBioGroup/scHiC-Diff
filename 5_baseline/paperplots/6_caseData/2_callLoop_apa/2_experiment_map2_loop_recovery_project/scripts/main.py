"""Run Map2 known enhancer-promoter loop recovery analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs"
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

from calculate_p2ll import calculate_loop_metrics, convert_loops_to_bins, maybe_log1p_transform
from config import AnalysisConfig, CELL_NUMBERS, METHODS
from io_utils import ensure_output_dirs, load_matrix, read_loop_bedpe, read_region, setup_logger, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing raw/scHiCluster/scHiCDiff matrix folders.")
    parser.add_argument("--loop-bedpe", type=Path, required=True, help="Known Map2 enhancer-promoter loop BEDPE file.")
    parser.add_argument("--region-file", type=Path, required=True, help="Region file with chrom, start, end on separate lines.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Output directory.")
    parser.add_argument("--resolution", type=int, default=20_000, help="Matrix resolution in bp.")
    parser.add_argument("--center-window", type=int, default=5, help="Odd-sized center window, e.g. 5 for 5x5.")
    parser.add_argument("--background-window", type=int, default=20, help="Background window size around loop center.")
    parser.add_argument("--log-transform-threshold", type=float, default=20.0, help="If matrix max is below this value, treat as already log-transformed.")
    return parser.parse_args()


def validate_windows(center_window: int, background_window: int) -> None:
    if center_window <= 0 or center_window % 2 == 0:
        raise ValueError("--center-window must be a positive odd integer")
    if background_window <= center_window:
        raise ValueError("--background-window must be larger than --center-window")


def matrix_path_for(data_dir: Path, method: str, cell_number: int) -> Path:
    meta = METHODS[method]
    return data_dir / meta["data_subdir"] / meta["file_pattern"].format(n=cell_number)


def run_analysis(config: AnalysisConfig) -> pd.DataFrame:
    validate_windows(config.center_window, config.background_window)
    ensure_output_dirs([config.output_dir])
    logger = setup_logger(config.output_dir / "analysis.log")

    logger.info("Starting Map2 known-loop recovery analysis")
    region_chrom, region_start, region_end = read_region(config.region_file)
    logger.info("Region: %s:%d-%d; resolution=%d", region_chrom, region_start, region_end, config.resolution)

    loop_df = read_loop_bedpe(config.loop_bedpe)
    logger.info("Loaded %d known loop(s) from %s", len(loop_df), config.loop_bedpe)

    first_matrix = load_matrix(matrix_path_for(config.data_dir, "raw", CELL_NUMBERS[0]))
    logger.info("Reference matrix shape from raw n=%d: %s", CELL_NUMBERS[0], first_matrix.shape)
    loop_bins = convert_loops_to_bins(
        loop_df,
        region_chrom=region_chrom,
        region_start=region_start,
        region_end=region_end,
        resolution=config.resolution,
        matrix_size=first_matrix.shape[0],
    )
    write_csv(loop_bins, config.output_dir / "loop_bin_pair.csv")
    for _, row in loop_bins.iterrows():
        logger.info("Loop %d bin pair: (%d, %d)", row["loop_id"], row["bin1"], row["bin2"])

    result_frames = []
    for method in METHODS:
        for cell_number in CELL_NUMBERS:
            matrix_path = matrix_path_for(config.data_dir, method, cell_number)
            matrix = load_matrix(matrix_path)
            if matrix.shape != first_matrix.shape:
                raise ValueError(f"Matrix shape mismatch for {matrix_path}: {matrix.shape} != {first_matrix.shape}")
            logger.info("Processing method=%s cells=%d shape=%s", method, cell_number, matrix.shape)
            matrix, log1p_applied = maybe_log1p_transform(
                matrix, config.log_transform_threshold, logger, f"{method} n={cell_number}"
            )
            metrics = calculate_loop_metrics(
                matrix,
                loop_bins,
                center_window=config.center_window,
                background_window=config.background_window,
                epsilon=config.epsilon,
            )
            metrics["method"] = method
            metrics["method_label"] = METHODS[method]["label"]
            metrics["cell_number"] = cell_number
            metrics["log1p_applied"] = log1p_applied
            for _, row in metrics.iterrows():
                logger.info(
                    "method=%s cells=%d loop=%d center=%.6g background=%.6g P2LL=%.6g log2_enrichment=%.6g",
                    method,
                    cell_number,
                    row["loop_id"],
                    row["center_signal"],
                    row["background_signal"],
                    row["P2LL"],
                    row["log2_enrichment"],
                )
            result_frames.append(metrics)

    results = pd.concat(result_frames, ignore_index=True)
    ordered_cols = [
        "method",
        "method_label",
        "cell_number",
        "loop_id",
        "bin1",
        "bin2",
        "P2LL",
        "center_signal",
        "background_signal",
        "log2_enrichment",
        "log1p_applied",
    ]
    results = results[ordered_cols]
    write_csv(results, config.output_dir / "P2LL_results.csv")
    write_csv(results[["method", "method_label", "cell_number", "loop_id", "log2_enrichment", "center_signal", "background_signal"]], config.output_dir / "enrichment_results.csv")
    from plot_results import plot_all

    plot_all(results, config.output_dir)
    logger.info("Analysis complete. Results written to %s", config.output_dir)
    return results


def main() -> None:
    args = parse_args()
    config = AnalysisConfig(
        data_dir=args.data_dir,
        loop_bedpe=args.loop_bedpe,
        region_file=args.region_file,
        output_dir=args.output_dir,
        resolution=args.resolution,
        center_window=args.center_window,
        background_window=args.background_window,
        log_transform_threshold=args.log_transform_threshold,
    )
    run_analysis(config)


if __name__ == "__main__":
    main()
