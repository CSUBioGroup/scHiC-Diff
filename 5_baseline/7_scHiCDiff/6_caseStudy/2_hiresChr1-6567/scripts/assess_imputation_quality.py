#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np

from diagnose_eval_v2_outputs import analyze_case, atomic_write


LOSS_PATTERN = re.compile(
    r"Training: loss_sum:\s*([0-9.eE+-]+)"
)


def parse_training_losses(path):
    losses = [
        float(match.group(1))
        for match in LOSS_PATTERN.finditer(path.read_text(encoding="utf-8"))
    ]
    if len(losses) < 2:
        raise ValueError(f"expected at least two training losses in {path}")
    if not np.isfinite(losses).all():
        raise ValueError(f"training log contains non-finite losses: {path}")
    window = min(5, len(losses) // 2)
    return {
        "count": len(losses),
        "first_window_mean": float(np.mean(losses[:window])),
        "last_window_mean": float(np.mean(losses[-window:])),
        "window_size": window,
    }


def gate(actual, requirement, passed):
    if isinstance(actual, np.generic):
        actual = actual.item()
    return {
        "actual": actual,
        "requirement": requirement,
        "passed": bool(passed),
    }


def assess(result_dir, train_log, args):
    case = analyze_case(result_dir)
    losses = parse_training_losses(train_log)
    matrices = case["matrices"]
    metrics = case["prediction_vs_raw"]
    prediction_depth = matrices["denoise_recon_inv"]["row_sums"]["quantiles"]["0.5"]
    target_depth = case["target_depth_after_expm1"]["quantiles"]["0.5"]
    depth_ratio = prediction_depth / target_depth if target_depth > 0 else None

    expected_shape = [args.expected_rows, args.expected_cols]
    shapes_match = all(
        matrix["shape"] == expected_shape for matrix in matrices.values()
    )
    global_pearson = metrics["global_pearson_including_zeros"]
    median_row_pearson = metrics["row_pearson_including_zeros"]["quantiles"].get(
        "0.5"
    )
    observed_mass = metrics["prediction_mass_at_raw_nonzero_fraction"]
    loss_ratio = losses["last_window_mean"] / losses["first_window_mean"]

    gates = {
        "all_shapes": gate(
            {name: matrix["shape"] for name, matrix in matrices.items()},
            f"all equal {expected_shape}",
            shapes_match,
        ),
        "training_loss_ratio": gate(
            loss_ratio,
            f"< {args.max_loss_ratio}",
            np.isfinite(loss_ratio) and loss_ratio < args.max_loss_ratio,
        ),
        "global_pearson": gate(
            global_pearson,
            f">= {args.min_global_pearson}",
            global_pearson is not None
            and np.isfinite(global_pearson)
            and global_pearson >= args.min_global_pearson,
        ),
        "median_row_pearson": gate(
            median_row_pearson,
            f">= {args.min_median_row_pearson}",
            median_row_pearson is not None
            and np.isfinite(median_row_pearson)
            and median_row_pearson >= args.min_median_row_pearson,
        ),
        "observed_mass_fraction": gate(
            observed_mass,
            f">= {args.min_observed_mass_fraction}",
            observed_mass is not None
            and np.isfinite(observed_mass)
            and observed_mass >= args.min_observed_mass_fraction,
        ),
        "median_depth_ratio": gate(
            depth_ratio,
            f"between {args.min_depth_ratio} and {args.max_depth_ratio}",
            depth_ratio is not None
            and np.isfinite(depth_ratio)
            and args.min_depth_ratio <= depth_ratio <= args.max_depth_ratio,
        ),
    }
    passed = all(item["passed"] for item in gates.values())
    return {
        "status": "passed" if passed else "failed",
        "result_dir": str(result_dir),
        "training_losses": losses,
        "gates": gates,
        "metrics": case,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--expected-rows", type=int, default=7466)
    parser.add_argument("--expected-cols", type=int, default=5050)
    parser.add_argument("--max-loss-ratio", type=float, default=1.0)
    parser.add_argument("--min-global-pearson", type=float, default=0.0061)
    parser.add_argument("--min-median-row-pearson", type=float, default=0.10)
    parser.add_argument(
        "--min-observed-mass-fraction", type=float, default=0.02
    )
    parser.add_argument("--min-depth-ratio", type=float, default=0.5)
    parser.add_argument("--max-depth-ratio", type=float, default=4.0)
    args = parser.parse_args()

    train_log = args.train_log or args.result_dir / "train.log"
    marker = args.result_dir / "quality_passed.flag"
    marker.unlink(missing_ok=True)
    payload = assess(args.result_dir, train_log, args)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write(args.result_dir / "quality.json", text)
    print(text, end="")
    if payload["status"] != "passed":
        raise SystemExit(1)
    atomic_write(marker, "quality passed\n")


if __name__ == "__main__":
    main()
