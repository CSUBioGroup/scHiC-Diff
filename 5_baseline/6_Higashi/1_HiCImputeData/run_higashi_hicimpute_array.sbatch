#!/usr/bin/env bash
#SBATCH -J higashi_hicimp
#SBATCH -A pi_limin_r
#SBATCH -p gpu2Q,gpu4Q,gpu8Q
#SBATCH --qos=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-11%6
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/1_HiCImputeData/logs/%x_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/1_HiCImputeData/logs/%x_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/1_HiCImputeData"
PYTHON_BIN="/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python"
MANIFEST="${BASE_DIR}/manifest.tsv"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

DATASET_ID="$(awk -F '\t' -v idx="$((TASK_ID + 2))" 'NR==idx {print $1}' "${MANIFEST}")"
if [[ -z "${DATASET_ID}" ]]; then
  echo "[ERROR] no dataset for array task ${TASK_ID}" >&2
  exit 1
fi

hostname
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
echo "[INFO] task=${TASK_ID} dataset=${DATASET_ID}"

for NEIGHBOR in 0 5; do
  DATASET_DIR="${DATASET_ID}_nbr${NEIGHBOR}"
  CONFIG="${BASE_DIR}/input/${DATASET_DIR}/config.JSON"
  OUT="${BASE_DIR}/output/npz_lower_tri/${DATASET_ID}_higashi_nbr_${NEIGHBOR}_lower_tri.npz"
  DENSE_OUT="${BASE_DIR}/output/npz_dense/${DATASET_ID}_higashi_nbr_${NEIGHBOR}.npz"
  HDF5_OUT="${BASE_DIR}/input/${DATASET_DIR}/temp/chrHICIMPUTE_hicimpute_higashi_nbr_${NEIGHBOR}_impute.hdf5"

  if [[ -s "${OUT}" && "${HIGASHI_FORCE_RERUN:-0}" != "1" ]]; then
    echo "[SKIP] existing output: ${OUT}"
    continue
  fi
  if [[ ! -s "${CONFIG}" ]]; then
    echo "[ERROR] missing config: ${CONFIG}" >&2
    exit 1
  fi

  echo "== Higashi ${DATASET_ID} nbr ${NEIGHBOR} =="
  date
  if [[ "${HIGASHI_FORCE_RERUN:-0}" == "1" ]]; then
    rm -f "${OUT}" "${DENSE_OUT}" "${HDF5_OUT}"
  fi
  "${PYTHON_BIN}" "${BASE_DIR}/scripts/run_higashi_one.py" \
    --config "${CONFIG}" \
    --training-updates "${HIGASHI_TRAINING_UPDATES:-200}" \
    --eval-updates "${HIGASHI_EVAL_UPDATES:-10}"

  "${PYTHON_BIN}" "${BASE_DIR}/scripts/hicimpute_higashi_pipeline.py" convert-output \
    --manifest "${MANIFEST}" \
    --input-root "${BASE_DIR}/input" \
    --output-root "${BASE_DIR}/output" \
    --dataset-id "${DATASET_ID}" \
    --dataset-dir-name "${DATASET_DIR}" \
    --neighbor "${NEIGHBOR}"
done
