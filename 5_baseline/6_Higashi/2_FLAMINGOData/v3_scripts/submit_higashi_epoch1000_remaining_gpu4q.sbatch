#!/usr/bin/env bash
#SBATCH -J v3_higashi_u1000_rem
#SBATCH -A pi_limin_r
#SBATCH -p gpu4Q
#SBATCH --qos=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-9%10
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/2_FLAMINGOData/v3_scripts/logs_epoch1000/u1000_remaining_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/2_FLAMINGOData/v3_scripts/logs_epoch1000/u1000_remaining_%A_%a.log

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export PYTHONUNBUFFERED=1

ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi"
SCRIPT_DIR="${ROOT}/2_FLAMINGOData/v3_scripts"
DATA_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets"
INPUT_ROOT="${ROOT}/2_FLAMINGOData/v3_epoch1000_inputData"
OUTPUT_ROOT="${ROOT}/2_FLAMINGOData/v3_epoch1000_outputData"
PYTHON="/public/home/hpc254701055/micromamba/envs/6_higashi/bin/python"
PY38="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python"

mkdir -p "${SCRIPT_DIR}/logs_epoch1000" \
  "${INPUT_ROOT}" \
  "${OUTPUT_ROOT}/npz_lower_tri" \
  "${OUTPUT_ROOT}/metrics/log1p_tasks"
export PATH="/public/home/hpc254701055/micromamba/envs/6_higashi/bin:${PATH}"

STEMS=(
  "v3_hybrid_W0p6_500cells_level0"
  "v3_hybrid_W0p6_500cells_level0"
  "v3_hybrid_W0p7_500cells_level0_r0p01"
  "v3_hybrid_W0p7_500cells_level0_r0p01"
  "v3_hybrid_W0p7_500cells_level0_r0p05"
  "v3_hybrid_W0p7_500cells_level0_r0p05"
  "v3_hybrid_W0p8_500cells_level0"
  "v3_hybrid_W0p8_500cells_level0"
  "v3_hybrid_W0p9_500cells_level0"
  "v3_hybrid_W0p9_500cells_level0"
)
NEIGHBORS=(0 5 0 5 0 5 0 5 0 5)

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
STEM="${STEMS[${TASK_ID}]}"
NEIGHBOR="${NEIGHBORS[${TASK_ID}]}"
RECORD_DIR="${INPUT_ROOT}/${STEM}_nbr${NEIGHBOR}"
CONFIG="${RECORD_DIR}/config.JSON"
IMP_HDF5="${RECORD_DIR}/temp/chrFLAMINGO_flamingo_higashi_nbr_${NEIGHBOR}_impute.hdf5"
OUT_NPZ="${OUTPUT_ROOT}/npz_lower_tri/${STEM}_higashi_nbr_${NEIGHBOR}_lower_tri.npz"
TASK_METRICS="${OUTPUT_ROOT}/metrics/log1p_tasks/${STEM}_higashi_updates1000_nbr_${NEIGHBOR}_log1p_metrics.csv"

echo "Started at: $(date)"
echo "Host: $(hostname)  Task: ${TASK_ID}  Dataset: ${STEM}  Neighbor: ${NEIGHBOR}"
echo "INPUT_ROOT=${INPUT_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
"${PYTHON}" --version
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true

if [[ ! -f "${CONFIG}" ]]; then
  echo "===== STEP 1: PREPARE cpu_num=10 ====="
  "${PY38}" "${SCRIPT_DIR}/v3_prepare_higashi.py" \
    --data-dir "${DATA_DIR}" \
    --input-root "${INPUT_ROOT}" \
    --datasets "${STEM}" \
    --neighbors "${NEIGHBOR}" \
    --cpu-num 10 \
    --gpu-num 1 \
    --overwrite
else
  echo "[skip] prepare, ${CONFIG} exists"
fi

if [[ "${HIGASHI_FORCE_RERUN:-0}" == "1" ]]; then
  echo "[force] removing previous epoch1000 outputs"
  rm -f "${IMP_HDF5}" "${OUT_NPZ}" "${TASK_METRICS}"
fi

if [[ ! -f "${IMP_HDF5}" ]]; then
  echo "===== STEP 2: HIGASHI TRAIN+IMPUTE updates=1000 eval=10 ====="
  "${PYTHON}" "${SCRIPT_DIR}/run_higashi_one.py" \
    --config "${CONFIG}" \
    --training-updates 1000 \
    --eval-updates 10
else
  echo "[skip] Higashi, ${IMP_HDF5} exists"
fi

echo "===== STEP 3: COLLECT ====="
"${PY38}" "${SCRIPT_DIR}/v3_collect_higashi.py" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --datasets "${STEM}" \
  --neighbor "${NEIGHBOR}" \
  --overwrite

echo "===== STEP 4: LOG1P EVALUATE TASK CSV ====="
"${PY38}" "${SCRIPT_DIR}/v3_evaluate.py" \
  --method "Higashi updates1000 ${NEIGHBOR} nbr" \
  --config-tag "updates1000_${NEIGHBOR}nbr" \
  --data-dir "${DATA_DIR}" \
  --pred-dir "${OUTPUT_ROOT}/npz_lower_tri" \
  --pred-pattern "{stem}_higashi_nbr_${NEIGHBOR}_lower_tri.npz" \
  --output-csv "${TASK_METRICS}" \
  --datasets "${STEM}" \
  --workers "${SLURM_CPUS_PER_TASK}"

echo "Finished at: $(date)"
