#!/usr/bin/env python3
"""Run one FLAMINGO dataset through classic Higashi."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path


def safe_generate_feats_one(temp1, temp2, size, length, c, qc_list):
    """Higashi feature generation that tolerates filtered-empty feature blocks."""
    import numpy as np
    from scipy.sparse import hstack
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    if temp1.shape[0] <= 3:
        return np.eye(temp1.shape[0]), c

    results = []
    component_size = size
    for temp in [temp1, temp2]:
        if temp is None or len(temp.data) == 0:
            continue
        mask = np.array(np.sum(temp > 0, axis=0) > min(5, temp.shape[0] - 2)).reshape((-1))
        temp = temp[:, mask]
        if temp.shape[1] <= 2 or len(temp.data) == 0:
            continue

        component_size = min(component_size, temp.shape[-1] - 2)
        mean_, std_ = np.mean(temp.data), np.std(temp.data)
        np.clip(temp.data, a_min=None, a_max=mean_ + 10 * std_, out=temp.data)
        results.append(temp)

    if not results:
        return np.eye(temp1.shape[0]), c

    component_size = max(1, int(component_size))
    if len(results) == 2:
        split = results[0].shape[1]
        temp = hstack(results)
        temp = normalize(temp, norm="l1", axis=1) * length
        temp1, temp2 = temp[:, :split], temp[:, split:]
    else:
        temp1, temp2 = results[0], None

    qc_list = np.asarray(qc_list)
    if len(qc_list) - np.sum(qc_list) > 10:
        model = TruncatedSVD(n_components=component_size, algorithm="randomized", n_iter=2).fit(temp1[qc_list])
        temp1 = model.transform(temp1)
        if temp2 is not None:
            model = TruncatedSVD(n_components=component_size, algorithm="randomized", n_iter=2).fit(temp2[qc_list])
            temp2 = model.transform(temp2)
    else:
        temp1 = TruncatedSVD(n_components=component_size, algorithm="randomized", n_iter=2).fit_transform(temp1)
        if temp2 is not None:
            temp2 = TruncatedSVD(n_components=component_size, algorithm="randomized", n_iter=2).fit_transform(temp2)
    if temp2 is not None:
        temp1 = np.concatenate([temp1, temp2], axis=1)
    return temp1, c


def patch_higashi_feature_generation() -> None:
    module = importlib.import_module("higashi.Process")
    module.generate_feats_one = safe_generate_feats_one
    print("[INFO] patched Higashi generate_feats_one for empty feature blocks", flush=True)


def patch_higashi_epochs(training_updates: int, eval_updates: int) -> None:
    print(f"[{time.ctime()}] importing higashi.Higashi_wrapper", flush=True)
    module = importlib.import_module("higashi.Higashi_wrapper")
    print(f"[{time.ctime()}] imported higashi.Higashi_wrapper", flush=True)
    original = module.Higashi.fetch_info_from_config

    def patched(self):
        original(self)
        self.update_num_per_training_epoch = training_updates
        self.update_num_per_eval_epoch = eval_updates
        print(
            f"[INFO] patched Higashi updates: train={training_updates} eval={eval_updates}",
            flush=True,
        )

    module.Higashi.fetch_info_from_config = patched


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training-updates", type=int, default=100)
    parser.add_argument("--eval-updates", type=int, default=4)
    parser.add_argument("--skip-process", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"[{time.ctime()}] run_higashi_one start", flush=True)
    patch_higashi_epochs(args.training_updates, args.eval_updates)
    patch_higashi_feature_generation()
    print(f"[{time.ctime()}] importing Higashi class", flush=True)
    from higashi.Higashi_wrapper import Higashi
    from higashi.Process import create_matrix, generate_chrom_start_end
    print(f"[{time.ctime()}] imported Higashi class", flush=True)

    config = json.loads(args.config.read_text())
    print(f"[INFO] config={args.config.resolve()}", flush=True)
    print(f"[INFO] temp_dir={config['temp_dir']}", flush=True)
    higashi = Higashi(str(args.config.resolve()))
    if not args.skip_process:
        print(f"[{time.ctime()}] create_matrix start", flush=True)
        generate_chrom_start_end(higashi.config)
        create_matrix(higashi.config)
        print(f"[{time.ctime()}] create_matrix done", flush=True)
    print(f"[{time.ctime()}] prep_model start", flush=True)
    higashi.prep_model()
    print(f"[{time.ctime()}] prep_model done", flush=True)
    print(f"[{time.ctime()}] train_for_embeddings start", flush=True)
    higashi.train_for_embeddings()
    print(f"[{time.ctime()}] train_for_embeddings done", flush=True)
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
    print("[OK] Higashi run finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
