#!/usr/bin/env bash
#SBATCH --job-name=v3_schicluster_impute
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/2_FLAMINGOData/v3_scripts/logs/impute_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/2_FLAMINGOData/v3_scripts/logs/impute_%A_%a.log
#SBATCH --account=pi_limin
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/2_FLAMINGOData/v3_scripts"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/2_FLAMINGOData/v3_inputData"
OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/2_FLAMINGOData/v3_outputData"
PYTHON="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python"
HICLUSTER="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/hicluster"
ENV_DIR="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38"

mkdir -p "${SCRIPT_DIR}/logs" "${OUTPUT_ROOT}"
export PATH="${ENV_DIR}/bin:${PATH}"

DATASETS=(
    "v3_hybrid_W0p5_500cells_level0"
    "v3_hybrid_W0p6_500cells_level0"
    "v3_hybrid_W0p7_500cells_level0_r0p01"
    "v3_hybrid_W0p7_500cells_level0_r0p05"
    "v3_hybrid_W0p8_500cells_level0"
    "v3_hybrid_W0p9_500cells_level0"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
STEM="${DATASETS[${TASK_ID}]}"

echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "Task: ${TASK_ID}  Dataset: ${STEM}"
"${PYTHON}" --version

cd "${SCRIPT_DIR}"
"${PYTHON}" 03_impute_v3_paramsweep.py \
  --dataset "${STEM}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --chrom chr19 \
  --n-bins 500 \
  --resolution 1 \
  --pad 1 \
  --std 1 \
  --rp 0.5 \
  --tol 0.01 \
  --window-size 500 \
  --step-size 500 \
  --output-format npz \
  --hicluster "${HICLUSTER}" \
  --workers "${SLURM_CPUS_PER_TASK}"

echo "Finished at: $(date)"