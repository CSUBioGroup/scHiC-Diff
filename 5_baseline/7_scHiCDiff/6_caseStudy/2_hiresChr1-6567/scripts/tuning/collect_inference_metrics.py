#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnose_eval_v2_outputs import analyze_case, atomic_write


def collect(result_dir, denoise_t, checkpoint, seed):
    case = analyze_case(result_dir)
    correlation = case["prediction_vs_raw"]
    inverse = case["matrices"]["denoise_recon_inv"]
    return {
        "checkpoint": str(checkpoint),
        "denoise_t_sample": denoise_t,
        "seed": seed,
        "global_pearson": correlation["global_pearson_including_zeros"],
        "median_row_pearson": correlation["row_pearson_including_zeros"][
            "quantiles"
        ]["0.5"],
        "observed_mass_fraction": correlation[
            "prediction_mass_at_raw_nonzero_fraction"
        ],
        "median_inverse_depth": inverse["row_sums"]["quantiles"]["0.5"],
        "density": inverse["density"],
        "shape": inverse["shape"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--denoise-t", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    payload = collect(
        args.result_dir, args.denoise_t, args.checkpoint, args.seed
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write(args.result_dir / "inference_metrics.json", text)
    print(text, end="")


if __name__ == "__main__":
    main()
