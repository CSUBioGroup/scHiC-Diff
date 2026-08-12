#!/usr/bin/env bash
#SBATCH --job-name=hicimpute_sz_tril
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_scripts/logs/convert_sz_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_scripts/logs/convert_sz_%A_%a.log
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-6

set -euo pipefail

PYTHON_BIN="/public/home/hpc254701055/micromamba/envs/2_schic-scvi-3d/bin/python"
SCRIPT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_scripts/v3_convert_sz_to_tril_npz.py"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Started at: $(date)"
echo "Task: ${SLURM_ARRAY_TASK_ID}"

"${PYTHON_BIN}" "${SCRIPT}" "${SLURM_ARRAY_TASK_ID}"

echo "Finished at: $(date)"