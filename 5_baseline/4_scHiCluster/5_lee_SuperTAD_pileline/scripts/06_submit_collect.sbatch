#!/usr/bin/env bash
#SBATCH --job-name=lee_collect
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/5_lee_SuperTAD_pileline/scripts/logs/collect_%A.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/5_lee_SuperTAD_pileline/scripts/logs/collect_%A.log
#SBATCH --account=pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00

# Collect per-cell imputed NPZ into per-cell-type feature NPZ
# Submit after all impute array tasks complete:
#   jid=$(sbatch --parsable 05_submit_impute.sbatch)
#   sbatch --dependency=afterok:${jid} 06_submit_collect.sbatch

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/5_lee_SuperTAD_pileline/scripts"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/5_lee_SuperTAD_pileline/result"
OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/5_lee_SuperTAD_pileline/output"
PYTHON="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python"
ENV_DIR="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38"

mkdir -p "${SCRIPT_DIR}/logs" "${OUTPUT_ROOT}"
export PATH="${ENV_DIR}/bin:${PATH}"

echo "=============================================="
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "=============================================="

"${PYTHON}" --version
"${PYTHON}" "${SCRIPT_DIR}/03_collect_lee.py" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --chrom chr4 \
  --n-bins 49 \
  --pad 1 --std 1 --rp 0.5 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --overwrite

echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="