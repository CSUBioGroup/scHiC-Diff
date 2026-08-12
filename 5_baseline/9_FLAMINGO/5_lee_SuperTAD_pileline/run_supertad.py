"""
Stage 3: SuperTAD TAD boundary identification
- Runs SuperTAD on 4 target (shared) + 400 trial matrices (per method)
Usage: python run_supertad.py --method scHiC-Diff
"""
import os
import sys
import json
import subprocess
import argparse
import numpy as np
from scipy.sparse import load_npz

import config as cfg


def matrix_to_txt(matrix, txt_path):
    np.savetxt(txt_path, matrix, fmt="%.6f")


def run_supertad(txt_path, output_dir, name):
    cmd = [cfg.SUPERTAD_BIN, "multi", txt_path,
           "--chrom1", cfg.CHROM, "-r", str(cfg.RESOLUTION),
           "-h", str(cfg.SUPERTAD_HEIGHT)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"  WARNING: SuperTAD failed for {name}: {result.stderr[:200]}")
        return None
    tsv_path = txt_path + ".multi.tsv"
    if not os.path.exists(tsv_path): return None
    out_path = os.path.join(output_dir, f"{name}.tsv")
    os.rename(tsv_path, out_path)
    return out_path


def parse_tad_tsv(tsv_path):
    tads = []
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 8:
                tads.append({"bin1": int(parts[1]) - 1, "bin2": int(parts[5]) - 1})
    return tads


def main():
    parser = argparse.ArgumentParser(description="SuperTAD TAD boundary identification")
    parser.add_argument("--method", required=True, help="Imputation method name")
    args = parser.parse_args()
    method = args.method

    print("=" * 60)
    print(f"Stage 3: SuperTAD (method={method})")
    print("=" * 60)

    if not os.path.exists(cfg.SUPERTAD_BIN):
        print(f"ERROR: SuperTAD binary not found at {cfg.SUPERTAD_BIN}")
        sys.exit(1)

    tmp_dir = os.path.join(cfg.SUPERTAD_DIR, "tmp")
    # Target SuperTAD: shared across methods
    target_out = os.path.join(cfg.SUPERTAD_DIR, "target")
    # Trial SuperTAD: per-method
    method_dir = os.path.join(cfg.SUPERTAD_DIR, method)
    trial_out = os.path.join(method_dir, "trials")
    for d in [tmp_dir, target_out, trial_out]:
        os.makedirs(d, exist_ok=True)

    summary = {"targets": {}, "trials": {}}
    n_runs = 0; n_errors = 0

    # === Target matrices (shared, run once) ===
    print("\n--- Target matrices ---")
    for cell_type in cfg.CELL_TYPES:
        name = f"{cell_type}_target"
        tsv_path = os.path.join(target_out, f"{name}.tsv")
        if os.path.exists(tsv_path):
            tads = parse_tad_tsv(tsv_path)
            summary["targets"][name] = {"n_tads": len(tads), "path": tsv_path}
            print(f"  {name}: {len(tads)} TADs (cached)")
            continue
        npz_path = os.path.join(cfg.TARGET_DIR, f"{cell_type}_target.npz")
        matrix = load_npz(npz_path).toarray()
        txt_path = os.path.join(tmp_dir, f"{name}.txt")
        matrix_to_txt(matrix, txt_path)
        out_path = run_supertad(txt_path, target_out, name)
        if out_path:
            tads = parse_tad_tsv(out_path)
            summary["targets"][name] = {"n_tads": len(tads), "path": out_path}
            print(f"  {name}: {len(tads)} TADs")
            n_runs += 1
        else:
            n_errors += 1
        if os.path.exists(txt_path): os.remove(txt_path)

    # === Trial matrices (per-method, 400 runs) ===
    print(f"\n--- Trial matrices ({method}) ---")
    matrices_dir = os.path.join(cfg.TRIALS_DIR, method, "matrices")
    for cell_type in cfg.CELL_TYPES:
        print(f"  {cell_type}...", end="", flush=True)
        for trial_id in range(cfg.N_TRIALS):
            npz_path = os.path.join(matrices_dir, f"{cell_type}_trial{trial_id:03d}.npz")
            if not os.path.exists(npz_path): n_errors += 1; continue
            matrix = load_npz(npz_path).toarray()
            name = f"{cell_type}_trial{trial_id:03d}"
            txt_path = os.path.join(tmp_dir, f"{name}.txt")
            matrix_to_txt(matrix, txt_path)
            out_path = run_supertad(txt_path, trial_out, name)
            if out_path:
                tads = parse_tad_tsv(out_path)
                summary["trials"][name] = {"n_tads": len(tads)}
                n_runs += 1
            else:
                n_errors += 1
            if os.path.exists(txt_path): os.remove(txt_path)
        print(f" done ({trial_id+1} runs)")

    try: os.rmdir(tmp_dir)
    except OSError: pass

    summary["total_runs"] = n_runs
    summary["total_errors"] = n_errors
    summary_path = os.path.join(method_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"  Total runs: {n_runs}, Errors: {n_errors}")
    print(f"  Summary: {summary_path}")
    print("\nStage 3 complete.")


if __name__ == "__main__":
    main()
