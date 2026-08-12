#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnose_eval_v2_outputs import atomic_write


ORIGINAL_GLOBAL = 0.45878373928183375
ORIGINAL_MEDIAN_ROW = 0.4985079789669996
EXPECTED_TIMES = (100, 250, 500, 750, 999, 1000)


def load_records(sweep_root):
    records = []
    for denoise_t in EXPECTED_TIMES:
        path = sweep_root / f"t{denoise_t}" / "inference_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="ascii") as handle:
            record = json.load(handle)
        if record["denoise_t_sample"] != denoise_t:
            raise ValueError(f"timestep mismatch in {path}")
        for key in ("global_pearson", "median_row_pearson"):
            if not math.isfinite(record[key]):
                raise ValueError(f"non-finite {key} in {path}")
        records.append(record)
    return records


def summarize(sweep_root):
    records = load_records(sweep_root)
    control = next(
        record for record in records if record["denoise_t_sample"] == 1000
    )
    for record in records:
        record["delta_global_vs_t1000"] = (
            record["global_pearson"] - control["global_pearson"]
        )
        record["delta_median_row_vs_t1000"] = (
            record["median_row_pearson"] - control["median_row_pearson"]
        )
        record["delta_global_vs_original"] = (
            record["global_pearson"] - ORIGINAL_GLOBAL
        )
        record["delta_median_row_vs_original"] = (
            record["median_row_pearson"] - ORIGINAL_MEDIAN_ROW
        )
    return {
        "candidate_count": len(records),
        "original_ramani": {
            "global_pearson": ORIGINAL_GLOBAL,
            "median_row_pearson": ORIGINAL_MEDIAN_ROW,
        },
        "standalone_t1000": control,
        "best_global_pearson": max(
            records, key=lambda record: record["global_pearson"]
        ),
        "best_median_row_pearson": max(
            records, key=lambda record: record["median_row_pearson"]
        ),
        "records": records,
    }


def render_tsv(payload):
    columns = (
        "denoise_t_sample",
        "global_pearson",
        "median_row_pearson",
        "delta_global_vs_t1000",
        "delta_median_row_vs_t1000",
        "delta_global_vs_original",
        "delta_median_row_vs_original",
        "median_inverse_depth",
        "density",
    )
    lines = ["\t".join(columns)]
    for record in payload["records"]:
        lines.append("\t".join(str(record[column]) for column in columns))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", required=True, type=Path)
    args = parser.parse_args()
    payload = summarize(args.sweep_root)
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write(args.sweep_root / "summary.json", json_text)
    atomic_write(args.sweep_root / "summary.tsv", render_tsv(payload))
    print(json_text, end="")


if __name__ == "__main__":
    main()
