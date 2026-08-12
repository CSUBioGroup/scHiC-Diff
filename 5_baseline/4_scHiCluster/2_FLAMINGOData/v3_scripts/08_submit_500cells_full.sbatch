#!/usr/bin/env bash
#SBATCH --job-name=v3_schicluster_500cells
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/2_FLAMINGOData/v3_scripts/logs/run500cells_%A.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/2_FLAMINGOData/v3_scripts/logs/run500cells_%A.log
#SBATCH --account=pi_limin
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/2_FLAMINGOData/v3_scripts"
# Source data dir for this single dataset.
DATA_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/3_500cells_datasets"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/2_FLAMINGOData/v3_inputData"
OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/2_FLAMINGOData/v3_outputData"
PYTHON="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python"
HICLUSTER="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/hicluster"
ENV_DIR="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38"
STEM="v3_hybrid_W0p7_500cells_level0"
METRICS_CSV="${OUTPUT_ROOT}/metrics/scHiCluster_FLAMINGO_v3_paramsweep_quality_metrics.csv"

mkdir -p "${SCRIPT_DIR}/logs" "${OUTPUT_ROOT}/metrics"
export PATH="${ENV_DIR}/bin:${PATH}"

echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "Dataset: ${STEM}  (from 3_500cells_datasets)"
"${PYTHON}" --version

cd "${SCRIPT_DIR}"

echo "===== STEP 1: PREPARE ====="
"${PYTHON}" 01_prepare_v3_paramsweep_schicluster.py \
  --data-dir "${DATA_DIR}" \
  --input-root "${INPUT_ROOT}" \
  --chrom chr19 \
  --n-bins 500 \
  --layer counts \
  --datasets "${STEM}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --overwrite

echo "===== STEP 2: IMPUTE ====="
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

echo "===== STEP 3: COLLECT ====="
"${PYTHON}" 05_collect_v3_paramsweep.py \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --chrom chr19 \
  --n-bins 500 \
  --pad 1 --std 1 --rp 0.5 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --datasets "${STEM}" \
  --overwrite

echo "===== STEP 4: EVALUATE ====="
"${PYTHON}" 06_evaluate_v3_paramsweep.py \
  --data-dir "${DATA_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --datasets "${STEM}" \
  --output-csv "${METRICS_CSV}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --append

echo "===== ALL DONE ====="
echo "Finished at: $(date)"