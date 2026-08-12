#!/usr/bin/env python3
"""Run scHiCluster imputation for a single v3 paramsweep dataset.

Calls ``hicluster impute-cell`` once per cell.  Cells are dispatched to a
process pool (size = workers) so that many cells run in parallel on a CPU
node.  Already-imputed cells are skipped unless --overwrite.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path


DEFAULT_INPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "4_scHiCluster/2_FLAMINGOData/v3_inputData"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/"
    "4_scHiCluster/2_FLAMINGOData/v3_outputData"
)
DEFAULT_HICLUSTER = Path(
    "/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/hicluster"
)

CHROM = "chr19"
N_BINS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hicluster impute-cell for one v3 paramsweep dataset."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chrom", default=CHROM)
    parser.add_argument("--n-bins", type=int, default=N_BINS)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--rp", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--step-size", type=int, default=500)
    parser.add_argument("--output-format", default="npz", choices=("npz", "hdf5"))
    parser.add_argument("--hicluster", type=Path, default=DEFAULT_HICLUSTER)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel cells; 0 = SLURM_CPUS_PER_TASK or 1")
    parser.add_argument("--max-cells", type=int, default=0)
    return parser.parse_args()


def mode_name(pad: int, std: float, rp: float) -> str:
    return f"pad{pad}_std{std:g}_rp{rp:g}_sqrtvc"


_W_HICLUSTER: Path = Path()
_W_INPUT_DIR: Path = Path()
_W_OUTPUT_DIR: Path = Path()
_W_CHROM: str = CHROM
_W_RES: int = 1
_W_CHROM_FILE: Path = Path()
_W_PAD: int = 1
_W_STD: float = 1.0
_W_RP: float = 0.5
_W_TOL: float = 0.01
_W_WINDOW: int = 500
_W_STEP: int = 500
_W_OUT_DIST: int = 500
_W_OUT_FMT: str = "npz"
_W_MODE: str = "pad1_std1_rp0.5_sqrtvc"
_W_OVERWRITE: bool = False


def _worker_init(kwargs: dict) -> None:
    global _W_HICLUSTER, _W_INPUT_DIR, _W_OUTPUT_DIR, _W_CHROM, _W_RES
    global _W_CHROM_FILE, _W_PAD, _W_STD, _W_RP, _W_TOL, _W_WINDOW, _W_STEP
    global _W_OUT_DIST, _W_OUT_FMT, _W_MODE, _W_OVERWRITE
    _W_HICLUSTER = kwargs["hicluster"]
    _W_INPUT_DIR = kwargs["input_dir"]
    _W_OUTPUT_DIR = kwargs["output_dir"]
    _W_CHROM = kwargs["chrom"]
    _W_RES = kwargs["resolution"]
    _W_CHROM_FILE = kwargs["chrom_file"]
    _W_PAD = kwargs["pad"]
    _W_STD = kwargs["std"]
    _W_RP = kwargs["rp"]
    _W_TOL = kwargs["tol"]
    _W_WINDOW = kwargs["window_size"]
    _W_STEP = kwargs["step_size"]
    _W_OUT_DIST = kwargs["output_dist"]
    _W_OUT_FMT = kwargs["output_format"]
    _W_MODE = kwargs["mode"]
    _W_OVERWRITE = kwargs["overwrite"]


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
        msg = proc.stderr.strip().splitlines()[-5:] if proc.stderr else []
        return cell_id, f"FAIL rc={proc.returncode}: {' | '.join(msg)}"
    if not out_path.exists():
        return cell_id, f"FAIL no output {out_path}"
    return cell_id, "ok"


def main() -> int:
    args = parse_args()
    workers = args.workers
    if workers <= 0:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    input_dir = args.input_root / args.dataset
    chrom_file = input_dir / f"simu_{args.chrom}.chrom.sizes"
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if not chrom_file.exists():
        raise FileNotFoundError(chrom_file)
    if not args.hicluster.exists():
        raise FileNotFoundError(args.hicluster)

    mode = mode_name(args.pad, args.std, args.rp)
    output_dir = args.output_root / "1_imputed_npz" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count input cells.
    input_files = sorted(input_dir.glob(f"cell_*_{args.chrom}.txt"),
                         key=lambda p: int(p.stem.split("_")[1]))
    n_cells = len(input_files)
    if args.max_cells > 0:
        n_cells = min(n_cells, args.max_cells)
    cell_ids = list(range(1, n_cells + 1))
    print(f"[impute] {args.dataset}: {n_cells} cells, mode={mode}, "
          f"workers={workers}, out={output_dir}", flush=True)

    init_kwargs = {
        "hicluster": args.hicluster,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "chrom": args.chrom,
        "resolution": args.resolution,
        "chrom_file": chrom_file,
        "pad": args.pad,
        "std": args.std,
        "rp": args.rp,
        "tol": args.tol,
        "window_size": args.window_size,
        "step_size": args.step_size,
        "output_dist": args.n_bins,
        "output_format": args.output_format,
        "mode": mode,
        "overwrite": args.overwrite,
    }

    ok = 0
    skipped = 0
    failed: list[tuple[int, str]] = []
    if workers <= 1:
        _worker_init(init_kwargs)
        for cid in cell_ids:
            status = _impute_cell(cid)[1]
            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                failed.append((cid, status))
            print(f"  cell_{cid}: {status}", flush=True)
    else:
        with Pool(processes=workers, initializer=_worker_init,
                  initargs=(init_kwargs,)) as pool:
            for cid, status in pool.imap_unordered(_impute_cell, cell_ids,
                                                   chunksize=max(1, len(cell_ids) // (workers * 4))):
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed.append((cid, status))
                print(f"  cell_{cid}: {status}", flush=True)

    print(f"[impute] {args.dataset}: ok={ok} skip={skipped} fail={len(failed)}",
          flush=True)
    if failed:
        for cid, msg in failed[:20]:
            print(f"  FAIL cell_{cid}: {msg}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())