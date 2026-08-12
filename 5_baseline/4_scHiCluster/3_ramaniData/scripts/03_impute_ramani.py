#!/usr/bin/env python3
"""Run scHiCluster impute-cell for 626-cell Ramani data across all chromosomes.

For each chromosome and each cell (1..626), calls ``hicluster impute-cell``
with the standard parameters (pad=1, std=1, rp=0.5).  Cells are dispatched
to a process pool for parallelism.  Already-imputed cells are skipped.

Parameters mirror the v3 FLAMINGO pipeline and the existing Ramani results:
  --pad 1 --std 1 --rp 0.5 --tol 0.01 --mode pad1_std1_rp0.5_sqrtvc
  --window_size n_bins --step_size n_bins --output_dist n_bins
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = BASE_DIR / "input" / "schicluster_input"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output" / "1_imputed_hdf5"
DEFAULT_HICLUSTER = Path(
    "/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/hicluster"
)

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
PAD = 1
STD = 1.0
RP = 0.5
TOL = 0.01
RESOLUTION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hicluster", type=Path, default=DEFAULT_HICLUSTER)
    parser.add_argument("--chroms", nargs="*", default=None)
    parser.add_argument("--pad", type=int, default=PAD)
    parser.add_argument("--std", type=float, default=STD)
    parser.add_argument("--rp", type=float, default=RP)
    parser.add_argument("--tol", type=float, default=TOL)
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--output-format", default="hdf5", choices=("npz", "hdf5"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-cells", type=int, default=0)
    return parser.parse_args()


def mode_name(pad: int, std: float, rp: float) -> str:
    return f"pad{pad}_std{std:g}_rp{rp:g}_sqrtvc"


def n_bins_from_chrom_sizes(path: Path) -> int:
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            return int(parts[1]) + 1
    raise ValueError(f"Cannot parse n_bins from {path}")


_W_HICLUSTER: Path = Path()
_W_INPUT_DIR: Path = Path()
_W_OUTPUT_DIR: Path = Path()
_W_CHROM: str = ""
_W_RES: int = 1
_W_CHROM_FILE: Path = Path()
_W_PAD: int = 1
_W_STD: float = 1.0
_W_RP: float = 0.5
_W_TOL: float = 0.01
_W_WINDOW: int = 0
_W_STEP: int = 0
_W_OUT_DIST: int = 0
_W_OUT_FMT: str = "hdf5"
_W_MODE: str = ""
_W_OVERWRITE: bool = False


def _worker_init(kwargs: dict) -> None:
    global _W_HICLUSTER, _W_INPUT_DIR, _W_OUTPUT_DIR, _W_CHROM, _W_RES
    global _W_CHROM_FILE, _W_PAD, _W_STD, _W_RP, _W_TOL, _W_WINDOW, _W_STEP
    global _W_OUT_DIST, _W_OUT_FMT, _W_MODE, _W_OVERWRITE
    for k, v in kwargs.items():
        globals()[f"_W_{k.upper()}"] = v


def _impute_cell(cell_id: int) -> tuple[int, str]:
    out_path = _W_OUTPUT_DIR / f"cell_{cell_id}_{_W_CHROM}_{_W_MODE}.{_W_OUT_FMT}"
    if out_path.exists() and not _W_OVERWRITE:
        return cell_id, "skip"
    cmd = [
        str(_W_HICLUSTER), "impute-cell",
        "--indir", f"{_W_INPUT_DIR}/",
        "--outdir", f"{_W_OUTPUT_DIR}/",
        "--cell", f"cell_{cell_id}",
        "--chrom", _W_CHROM,
        "--res", str(_W_RES),
        "--chrom_file", str(_W_CHROM_FILE),
        "--pad", str(_W_PAD),
        "--std", str(_W_STD),
        "--rp", str(_W_RP),
        "--tol", str(_W_TOL),
        "--window_size", str(_W_WINDOW),
        "--step_size", str(_W_STEP),
        "--output_dist", str(_W_OUT_DIST),
        "--output_format", _W_OUT_FMT,
        "--mode", _W_MODE,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip().splitlines()[-3:] if proc.stderr else []
        return cell_id, f"FAIL: {' | '.join(msg)}"
    if not out_path.exists():
        return cell_id, f"FAIL: no output"
    return cell_id, "ok"


def impute_chrom(chrom: str, args: argparse.Namespace, workers: int,
                 mode: str) -> int:
    input_dir = args.input_root / chrom
    chrom_file = input_dir / f"ramani_{chrom}.chrom.sizes"
    if not input_dir.is_dir():
        print(f"[impute] {chrom}: input dir {input_dir} not found, skipping",
              flush=True)
        return 1
    if not chrom_file.exists():
        print(f"[impute] {chrom}: chrom.sizes {chrom_file} not found, skipping",
              flush=True)
        return 1
    if not args.hicluster.exists():
        raise FileNotFoundError(args.hicluster)

    n_bins = n_bins_from_chrom_sizes(chrom_file)
    output_dir = args.output_root / chrom
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob(f"cell_*_{chrom}.txt"),
                         key=lambda p: int(p.stem.split("_")[1]))
    n_cells = len(input_files)
    if args.max_cells > 0:
        n_cells = min(n_cells, args.max_cells)
    cell_ids = list(range(1, n_cells + 1))
    print(f"[impute] {chrom}: {n_cells} cells, n_bins={n_bins}, "
          f"mode={mode}, workers={workers}", flush=True)

    init_kwargs = {
        "hicluster": args.hicluster,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "chrom": chrom,
        "res": args.resolution,
        "chrom_file": chrom_file,
        "pad": args.pad,
        "std": args.std,
        "rp": args.rp,
        "tol": args.tol,
        "window": n_bins,
        "step": n_bins,
        "out_dist": n_bins,
        "out_fmt": args.output_format,
        "mode": mode,
        "overwrite": args.overwrite,
    }

    ok = skip = 0
    failed: list[tuple[int, str]] = []
    if workers <= 1:
        _worker_init(init_kwargs)
        for cid in cell_ids:
            _, status = _impute_cell(cid)
            if status == "ok": ok += 1
            elif status == "skip": skip += 1
            else: failed.append((cid, status))
            if cid % 50 == 0:
                print(f"  {chrom}: {cid}/{n_cells}", flush=True)
    else:
        with Pool(processes=workers, initializer=_worker_init,
                  initargs=(init_kwargs,)) as pool:
            for cid, status in pool.imap_unordered(_impute_cell, cell_ids,
                    chunksize=max(1, len(cell_ids) // (workers * 4))):
                if status == "ok": ok += 1
                elif status == "skip": skip += 1
                else: failed.append((cid, status))
                if (ok + skip + len(failed)) % 50 == 0:
                    print(f"  {chrom}: {ok+skip+len(failed)}/{n_cells}", flush=True)

    print(f"[impute] {chrom}: ok={ok} skip={skip} fail={len(failed)}", flush=True)
    if failed:
        for cid, msg in failed[:10]:
            print(f"  FAIL cell_{cid}: {msg}", flush=True)
        return 1
    return 0


def main() -> int:
    args = parse_args()
    workers = args.workers
    if workers <= 0:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    chroms = args.chroms or CHROMS
    mode = mode_name(args.pad, args.std, args.rp)
    print(f"[impute] chroms={len(chroms)}, mode={mode}, workers={workers}",
          flush=True)
    rc = 0
    for chrom in chroms:
        if impute_chrom(chrom, args, workers, mode) != 0:
            rc = 1
    print("[impute] all done", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
