#!/usr/bin/env python3
"""Resume classic Higashi after the stage 1 embedding checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from run_higashi_one import (
    patch_higashi_epochs,
    patch_higashi_feature_generation,
    patch_higashi_negative_sampling,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-updates", type=int, default=1000)
    parser.add_argument("--eval-updates", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"[{time.ctime()}] run_higashi_resume_after_stage1 start", flush=True)
    patch_higashi_epochs(args.training_updates, args.eval_updates)
    patch_higashi_negative_sampling()
    patch_higashi_feature_generation()

    from higashi.Higashi_wrapper import Higashi

    config = json.loads(args.config.read_text())
    stage1_model = Path(config["temp_dir"]) / "model" / "model.chkpt_stage1_model"
    if not stage1_model.exists():
        raise FileNotFoundError(f"missing stage1 checkpoint: {stage1_model}")

    print(f"[INFO] config={args.config.resolve()}", flush=True)
    print(f"[INFO] using stage1 checkpoint={stage1_model}", flush=True)
    print(f"[INFO] temp_dir={config['temp_dir']}", flush=True)

    higashi = Higashi(str(args.config.resolve()))
    print(f"[{time.ctime()}] prep_model start", flush=True)
    higashi.prep_model()
    print(f"[{time.ctime()}] prep_model done", flush=True)

    print(f"[{time.ctime()}] train_for_imputation_nbr_0 start", flush=True)
    higashi.train_for_imputation_nbr_0()
    print(f"[{time.ctime()}] train_for_imputation_nbr_0 done", flush=True)

    print(f"[{time.ctime()}] impute_no_nbr start", flush=True)
    higashi.impute_no_nbr()
    print(f"[{time.ctime()}] impute_no_nbr done", flush=True)

    if config.get("impute_with_nbr", False):
        print(f"[{time.ctime()}] train_for_imputation_with_nbr start", flush=True)
        higashi.train_for_imputation_with_nbr()
        print(f"[{time.ctime()}] train_for_imputation_with_nbr done", flush=True)
        print(f"[{time.ctime()}] impute_with_nbr start", flush=True)
        higashi.impute_with_nbr()
        print(f"[{time.ctime()}] impute_with_nbr done", flush=True)

    print("[OK] Higashi resume finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
