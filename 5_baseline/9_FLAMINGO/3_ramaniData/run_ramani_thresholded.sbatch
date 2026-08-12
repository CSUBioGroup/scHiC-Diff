#!/usr/bin/env bash
#SBATCH -J ramani_thresh
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/3_ramaniData/output/logs/slurm_%x_%A.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/3_ramaniData/output/logs/slurm_%x_%A.err

# =============================================================================
# Thresholded post-processing for Ramani FLAMINGO completed tensors.
# Does NOT re-run FLAMINGO — only post-processes existing completed_tensor.npy
# files to remove the uniform ~1.0 background that homogenizes cells.
# Tries multiple thresholds and generates embeddings for clustering eval.
# =============================================================================

set -euo pipefail

export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/3_ramaniData"
PYTHON="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python"
SCRIPT="${BASE}/scripts/threshold_ramani_flamingo.py"

mkdir -p "${BASE}/output/logs"

echo "============================================================"
echo "Ramani FLAMINGO thresholded post-processing"
echo "  node = $(hostname)"
echo "  start = $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Try multiple thresholds: 0.5, 1.0, 2.0
for THRESH in 0.5 1.0 2.0; do
    echo ""
    echo "=== Threshold ${THRESH} ==="
    "${PYTHON}" "${SCRIPT}" \
        --output-root "${BASE}/output" \
        --input-root "${BASE}/input" \
        --threshold "${THRESH}" \
        --make-embedding --log1p \
        --per-chrom-dim 5 --seed 100
done

echo ""
echo "============================================================"
echo "Done. Outputs:"
echo "  output/chrom_npz_thresh*/  — thresholded chrom NPZ"
echo "  output/ramani_embedding_thresh*.npz — embeddings"
echo "  end = $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"