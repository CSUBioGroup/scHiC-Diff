#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnose_eval_v2_outputs import atomic_write


EPOCHS = (499, 999, 1499, 1999, 2499, 2999)


def load_records(sweep_root, epoch2999_result_dir):
    records = []
    for epoch in EPOCHS:
        reused = epoch == 2999
        result_dir = (
            epoch2999_result_dir
            if reused
            else sweep_root / f"epoch{epoch}"
        )
        path = result_dir / "inference_metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="ascii") as handle:
            record = json.load(handle)
        if record["denoise_t_sample"] != 750 or record["seed"] != 10:
            raise ValueError(f"effective parameter mismatch in {path}")
        expected_checkpoint = f"epoch={epoch:06d}.ckpt"
        if not record["checkpoint"].endswith(expected_checkpoint):
            raise ValueError(f"checkpoint mismatch in {path}")
        for key in ("global_pearson", "median_row_pearson"):
            if not math.isfinite(record[key]):
                raise ValueError(f"non-finite {key} in {path}")
        record["checkpoint_epoch"] = epoch
        record["result_dir"] = str(result_dir)
        record["reused_existing_result"] = reused
        records.append(record)
    return records


def summarize(sweep_root, epoch2999_result_dir):
    records = load_records(sweep_root, epoch2999_result_dir)
    control = records[-1]
    for record in records:
        record["delta_global_vs_epoch2999"] = (
            record["global_pearson"] - control["global_pearson"]
        )
        record["delta_median_row_vs_epoch2999"] = (
            record["median_row_pearson"] - control["median_row_pearson"]
        )
    return {
        "candidate_count": len(records),
        "fixed_denoise_t_sample": 750,
        "fixed_seed": 10,
        "epoch2999_control": control,
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
        "checkpoint_epoch",
        "global_pearson",
        "median_row_pearson",
        "delta_global_vs_epoch2999",
        "delta_median_row_vs_epoch2999",
        "median_inverse_depth",
        "density",
        "observed_mass_fraction",
        "reused_existing_result",
        "result_dir",
    )
    lines = ["\t".join(columns)]
    for record in payload["records"]:
        lines.append("\t".join(str(record[column]) for column in columns))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", required=True, type=Path)
    parser.add_argument("--epoch2999-result-dir", required=True, type=Path)
    args = parser.parse_args()
    payload = summarize(args.sweep_root, args.epoch2999_result_dir)
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write(args.sweep_root / "summary.json", json_text)
    atomic_write(args.sweep_root / "summary.tsv", render_tsv(payload))
    print(json_text, end="")


if __name__ == "__main__":
    main()
