#!/usr/bin/env bash
#SBATCH --job-name=lee_prepare
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/5_lee_SuperTAD_pileline/scripts/logs/prepare_%A.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/5_lee_SuperTAD_pileline/scripts/logs/prepare_%A.log
#SBATCH --account=pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=01:00:00

# Prepare Lee SuperTAD BEDPE -> scHiCluster triplet input
# 4 cell types (Astro/Endo/ODC/OPC), total 2098 cells

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/5_lee_SuperTAD_pileline/scripts"
PYTHON="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python"
ENV_DIR="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38"

mkdir -p "${SCRIPT_DIR}/logs"
export PATH="${ENV_DIR}/bin:${PATH}"

echo "=============================================="
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "=============================================="

"${PYTHON}" --version
"${PYTHON}" "${SCRIPT_DIR}/01_prepare_lee_bedpe.py" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --overwrite

echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="