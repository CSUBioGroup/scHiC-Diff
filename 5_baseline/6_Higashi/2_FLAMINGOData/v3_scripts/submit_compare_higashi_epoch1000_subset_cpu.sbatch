#!/usr/bin/env bash
#SBATCH -J v3_higashi_u1000_cmp
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH --qos=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/2_FLAMINGOData/v3_epoch1000_outputData/metrics/compare_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/2_FLAMINGOData/v3_epoch1000_outputData/metrics/compare_%j.err

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export PYTHONUNBUFFERED=1

ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi"
SCRIPT_DIR="${ROOT}/2_FLAMINGOData/v3_scripts"
OUTPUT_ROOT="${ROOT}/2_FLAMINGOData/v3_epoch1000_outputData"
PY38="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python"

mkdir -p "${OUTPUT_ROOT}/metrics"
echo "Started at: $(date)"
"${PY38}" "${SCRIPT_DIR}/compare_higashi_epoch1000_vs_epoch200.py"
echo "Finished at: $(date)"
