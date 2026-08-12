#!/usr/bin/env bash
#SBATCH --job-name=v3_schicluster_prepare
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/2_FLAMINGOData/v3_scripts/logs/prepare_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/2_FLAMINGOData/v3_scripts/logs/prepare_%A_%a.log
#SBATCH --account=pi_limin
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --array=0-5

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/2_FLAMINGOData/v3_scripts"
DATA_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/2_FLAMINGOData/v3_inputData"
PYTHON="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python"
ENV_DIR="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38"

mkdir -p "${SCRIPT_DIR}/logs" "${INPUT_ROOT}"
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
"${PYTHON}" 01_prepare_v3_paramsweep_schicluster.py \
  --data-dir "${DATA_DIR}" \
  --input-root "${INPUT_ROOT}" \
  --chrom chr19 \
  --n-bins 500 \
  --layer counts \
  --single-dataset "${STEM}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --overwrite

echo "Finished at: $(date)"