#!/usr/bin/env python3
"""
Local GPU training and imputation runner for K562 datasets.
Directly corresponds to `examples/submit_k562_bs128.sbatch` but runs on local GPUs
without requiring a SLURM cluster.

Features:
  - Supports single dataset or all 12 K562 datasets sequentially or in parallel.
  - Automatically skips datasets whose `denoise_recon_inv.npz` already exists (use --force to overwrite).
  - Matches the exact hyperparameter protocol: bs=128, test_bs=9999, patience=25, min_delta=1e-4.
  - Optional post-training automatic evaluation against 5_baseline/0_gtData.

Usage:
  # Run all 12 datasets sequentially on GPU 0
  python examples/run_k562_bs128_local.py

  # Run a single dataset on GPU 0
  python examples/run_k562_bs128_local.py --dataset K562_T1_1k

  # Run on a specific GPU with evaluation
  python examples/run_k562_bs128_local.py --gpu 1 --evaluate
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ALL_DATASETS = [
    "K562_T1_1k", "K562_T1_2k", "K562_T1_4k", "K562_T1_7k",
    "K562_T2_1k", "K562_T2_2k", "K562_T2_4k", "K562_T2_7k",
    "K562_T3_1k", "K562_T3_2k", "K562_T3_4k", "K562_T3_7k",
]


def resolve_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    for p in [current] + list(current.parents):
        if (p / "main.py").exists() and (p / "schicdiff").is_dir():
            return p
    raise RuntimeError("Could not resolve repository root containing main.py and schicdiff/")


def check_gpu(gpu_id: int):
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ Error: No CUDA-capable GPU detected.", file=sys.stderr)
            print("   main.py enforces accelerator='gpu'. Please run on a GPU node/machine.", file=sys.stderr)
            sys.exit(1)
        count = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= count:
            print(f"❌ Error: Invalid --gpu {gpu_id}. Available device count: {count}", file=sys.stderr)
            sys.exit(1)
        dev_name = torch.cuda.get_device_name(gpu_id)
        print(f"✓ GPU check passed: [{gpu_id}] {dev_name} (Total GPUs: {count})")
    except ImportError:
        print("⚠ Warning: PyTorch not installed in the current environment.", file=sys.stderr)


def train_single_dataset(
    ds: str,
    repo_root: Path,
    input_dir: Path,
    save_root: Path,
    log_dir: Path,
    patience: int = 25,
    ckpt_every: int = 50,
    batch_size: int = 128,
    test_batch_size: int = 9999,
    num_workers: int = 4,
    gpu_id: int = 0,
    force_retrain: bool = False,
) -> tuple[str, str, str]:
    save_path = save_root / f"{ds}_sim"
    h5ad_path = input_dir / f"{ds}_sim.h5ad"
    done_target = save_path / "denoise_recon_inv.npz"

    if not h5ad_path.exists():
        return ds, "FAIL(missing input)", f"File not found: {h5ad_path}"

    if done_target.exists() and not force_retrain:
        sz_kb = done_target.stat().st_size / 1024
        return ds, "SKIP", f"Already exists ({sz_kb:.1f} KB): {done_target.name}"

    save_path.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"local_{ds}.log"

    cmd = [
        sys.executable, "main.py",
        "-t", "True",
        "--base", "configs/recon_masked.yaml",
        "-n", f"{ds}_sim.seed10",
        "-l", str(log_dir),
        "--save_path", str(save_path),
        f"data.params.batch_size={batch_size}",
        f"data.params.test_batch_size={test_batch_size}",
        f"data.params.num_workers={num_workers}",
        f"lightning.callbacks.early_stopping_callback.params.patience={patience}",
        f"lightning.modelcheckpoint.params.every_n_epochs={ckpt_every}",
        "data.params.train.params.dataset=K562",
        f"data.params.train.params.fname={h5ad_path}",
        "data.params.validation.params.dataset=K562",
        f"data.params.validation.params.fname={h5ad_path}",
        "data.params.test.params.dataset=K562",
        f"data.params.test.params.fname={h5ad_path}",
    ]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["NUMEXPR_NUM_THREADS"] = "4"

    t0 = time.time()
    with open(log_file, "w", encoding="utf-8") as fh:
        res = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0

    if res.returncode == 0 and done_target.exists():
        return ds, "DONE", f"Completed in {elapsed:.1f}s (Log: {log_file.name})"
    else:
        return ds, f"FAIL(rc={res.returncode})", f"Check log: {log_file}"


def main():
    parser = argparse.ArgumentParser(
        description="Local GPU runner for K562 scHiC-Diff training and imputation (SLURM-free)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help=f"Dataset to train ('all' for all 12 conditions, or specific like 'K562_T1_1k'). Choices: {ALL_DATASETS + ['all']}",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to run on")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--test-batch-size", type=int, default=9999, help="Test batch size (single pass)")
    parser.add_argument("--patience", type=int, default=25, help="EarlyStopping patience")
    parser.add_argument("--ckpt-every", type=int, default=50, help="Checkpoint saving frequency (every n epochs)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--parallel", type=int, default=1, help="Number of datasets to run concurrently (if GPU memory permits)")
    parser.add_argument("--force", action="store_true", help="Force retrain even if output npz already exists")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation across completed datasets after training")
    parser.add_argument("--save-root", type=str, default=None, help="Custom result save root directory")
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    input_dir = repo_root / "5_baseline/7_scHiCDiff/1_HiCImputeData/input"
    save_root = Path(args.save_root) if args.save_root else repo_root / "results/training_results_v5fast_bs128"
    log_dir = repo_root / "logs/recon_masked_v5fast_bs128"

    print("=" * 75)
    print("scHiC-Diff (schicdiff) Local GPU Training Runner")
    print(f"Project Root:     {repo_root}")
    print(f"Input Directory:  {input_dir}")
    print(f"Results Root:     {save_root}")
    print(f"Logs Directory:   {log_dir}")
    print(f"Selected Dataset: {args.dataset}")
    print(f"Target GPU:       cuda:{args.gpu}")
    print(f"Hyperparameters:  bs={args.batch_size}, patience={args.patience}, ckpt_every={args.ckpt_every}")
    print("=" * 75)

    check_gpu(args.gpu)

    if args.dataset == "all":
        datasets_to_run = ALL_DATASETS
    else:
        if args.dataset not in ALL_DATASETS:
            print(f"❌ Unknown dataset '{args.dataset}'. Must be one of: {ALL_DATASETS}", file=sys.stderr)
            sys.exit(1)
        datasets_to_run = [args.dataset]

    # Pre-check existing
    pending = [ds for ds in datasets_to_run if not (save_root / f"{ds}_sim/denoise_recon_inv.npz").exists()]
    print(f"\nTotal datasets to process: {len(datasets_to_run)} ({len(pending)} pending training)")

    start_time = time.time()
    results = []

    if args.parallel > 1 and len(datasets_to_run) > 1:
        print(f"Executing with ThreadPoolExecutor(max_workers={args.parallel})...")
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = [
                ex.submit(
                    train_single_dataset,
                    ds=ds,
                    repo_root=repo_root,
                    input_dir=input_dir,
                    save_root=save_root,
                    log_dir=log_dir,
                    patience=args.patience,
                    ckpt_every=args.ckpt_every,
                    batch_size=args.batch_size,
                    test_batch_size=args.test_batch_size,
                    num_workers=args.num_workers,
                    gpu_id=args.gpu,
                    force_retrain=args.force,
                )
                for ds in datasets_to_run
            ]
            for f in futures:
                res = f.result()
                results.append(res)
                print(f"[{res[1]:10s}] {res[0]:15s} | {res[2]}")
    else:
        for i, ds in enumerate(datasets_to_run, 1):
            print(f"\n({i}/{len(datasets_to_run)}) Processing {ds}...")
            res = train_single_dataset(
                ds=ds,
                repo_root=repo_root,
                input_dir=input_dir,
                save_root=save_root,
                log_dir=log_dir,
                patience=args.patience,
                ckpt_every=args.ckpt_every,
                batch_size=args.batch_size,
                test_batch_size=args.test_batch_size,
                num_workers=args.num_workers,
                gpu_id=args.gpu,
                force_retrain=args.force,
            )
            results.append(res)
            print(f"[{res[1]:10s}] {res[0]:15s} | {res[2]}")

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print("TRAINING SUMMARY:")
    print("=" * 75)
    for ds, status, info in results:
        print(f"  {ds:15s} : {status:10s} - {info}")
    print(f"\nTotal elapsed time: {total_elapsed:.1f}s")

    # Post evaluation if requested
    if args.evaluate:
        eval_script = repo_root / "examples/evaluate_k562_imputation.py"
        if eval_script.exists():
            print("\n[Auto Evaluation] Launching evaluate_k562_imputation.py...")
            eval_cmd = [sys.executable, str(eval_script), "--save-root", str(save_root)]
            subprocess.run(eval_cmd, cwd=repo_root)
        else:
            print(f"⚠ Evaluation script not found: {eval_script}", file=sys.stderr)


if __name__ == "__main__":
    main()
