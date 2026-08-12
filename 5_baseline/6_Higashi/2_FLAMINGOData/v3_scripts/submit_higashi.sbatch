#!/usr/bin/env bash
#SBATCH --job-name=v3_higashi
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/2_FLAMINGOData/v3_scripts/logs/impute_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/2_FLAMINGOData/v3_scripts/logs/impute_%A_%a.log
#SBATCH --account=pi_limin
#SBATCH --partition=gpu2Q
#SBATCH --qos=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-13

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/2_FLAMINGOData/v3_scripts"
DATA_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/2_FLAMINGOData/v3_inputData"
OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/2_FLAMINGOData/v3_outputData"
PYTHON="/public/home/hpc254701055/micromamba/envs/6_higashi/bin/python"
PY38="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python"
METRICS_CSV="${OUTPUT_ROOT}/metrics/scHiCluster_FLAMINGO_v3_paramsweep_quality_metrics.csv"

mkdir -p "${SCRIPT_DIR}/logs" "${INPUT_ROOT}" "${OUTPUT_ROOT}/npz_lower_tri" "${OUTPUT_ROOT}/metrics"
export PATH="/public/home/hpc254701055/micromamba/envs/6_higashi/bin:${PATH}"

DATASETS=(
    "v3_hybrid_W0p5_500cells_level0"
    "v3_hybrid_W0p6_500cells_level0"
    "v3_hybrid_W0p7_500cells_level0"
    "v3_hybrid_W0p7_500cells_level0_r0p01"
    "v3_hybrid_W0p7_500cells_level0_r0p05"
    "v3_hybrid_W0p8_500cells_level0"
    "v3_hybrid_W0p9_500cells_level0"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
DATASET_INDEX=$(( TASK_ID / 2 ))
NEIGHBOR=$(( (TASK_ID % 2) * 5 ))
STEM="${DATASETS[${DATASET_INDEX}]}"
RECORD_DIR="${INPUT_ROOT}/${STEM}_nbr${NEIGHBOR}"
CONFIG="${RECORD_DIR}/config.JSON"

echo "Started at: $(date)"
echo "Host: $(hostname)  Task: ${TASK_ID}  Dataset: ${STEM}  Neighbor: ${NEIGHBOR}"
"${PYTHON}" --version

# 1) Prepare input
if [[ ! -f "${CONFIG}" ]]; then
  echo "===== STEP 1: PREPARE ====="
  "${PY38}" "${SCRIPT_DIR}/v3_prepare_higashi.py" \
    --data-dir "${DATA_DIR}" --input-root "${INPUT_ROOT}" \
    --datasets "${STEM}" --neighbors "${NEIGHBOR}" --cpu-num 20 --gpu-num 1 --overwrite
else
  echo "[skip] prepare, ${CONFIG} exists"
fi

# 2) Run Higashi train + impute
IMP_HDF5="${RECORD_DIR}/temp/chrFLAMINGO_flamingo_higashi_nbr_${NEIGHBOR}_impute.hdf5"
if [[ ! -f "${IMP_HDF5}" ]]; then
  echo "===== STEP 2: HIGASHI TRAIN+IMPUTE ====="
  "${PYTHON}" "${SCRIPT_DIR}/run_higashi_one.py" \
    --config "${CONFIG}" \
    --training-updates 200 --eval-updates 10
else
  echo "[skip] Higashi, ${IMP_HDF5} exists"
fi

# 3) Collect
echo "===== STEP 3: COLLECT ====="
"${PY38}" "${SCRIPT_DIR}/v3_collect_higashi.py" \
  --input-root "${INPUT_ROOT}" --output-root "${OUTPUT_ROOT}" \
  --datasets "${STEM}" --neighbor "${NEIGHBOR}" --overwrite

# 4) Evaluate
echo "===== STEP 4: EVALUATE ====="
"${PY38}" "${SCRIPT_DIR}/v3_evaluate.py" \
  --method "Higashi ${NEIGHBOR} nbr" \
  --config-tag "${NEIGHBOR}nbr" \
  --data-dir "${DATA_DIR}" \
  --pred-dir "${OUTPUT_ROOT}/npz_lower_tri" \
  --pred-pattern "{stem}_higashi_nbr_${NEIGHBOR}_lower_tri.npz" \
  --output-csv "${METRICS_CSV}" \
  --datasets "${STEM}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --append

echo "Finished at: $(date)"