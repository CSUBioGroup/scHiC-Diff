#!/usr/bin/env bash
#SBATCH -J higashi_hicimp_slow20
#SBATCH -A pi_limin_r
#SBATCH -p gpu2Q
#SBATCH --qos=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/1_HiCImputeData/logs/%x_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/1_HiCImputeData/logs/%x_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-20}"

BASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/1_HiCImputeData"
PYTHON_BIN="/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python"
MANIFEST="${BASE_DIR}/manifest.tsv"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

DATASETS=(K562_T1_4k K562_T1_7k K562_T2_7k K562_T1_4k K562_T1_7k K562_T2_7k)
NEIGHBORS=(0 0 0 5 5 5)
MODES=(resume0 resume0 resume0 full5 full5 full5)

DATASET_ID="${DATASETS[$TASK_ID]}"
NEIGHBOR="${NEIGHBORS[$TASK_ID]}"
MODE="${MODES[$TASK_ID]}"

if [[ "${MODE}" == "resume0" ]]; then
  DATASET_DIR="${DATASET_ID}_nbr0"
  RUNNER="${BASE_DIR}/scripts/run_higashi_resume_after_stage1.py"
else
  DATASET_DIR="${DATASET_ID}_nbr5_cpu20"
  RUNNER="${BASE_DIR}/scripts/run_higashi_one.py"
fi

CONFIG="${BASE_DIR}/input/${DATASET_DIR}/config.JSON"
OUT="${BASE_DIR}/output/npz_lower_tri/${DATASET_ID}_higashi_nbr_${NEIGHBOR}_lower_tri.npz"
DENSE_OUT="${BASE_DIR}/output/npz_dense/${DATASET_ID}_higashi_nbr_${NEIGHBOR}.npz"
HDF5_OUT="${BASE_DIR}/input/${DATASET_DIR}/temp/chrHICIMPUTE_hicimpute_higashi_nbr_${NEIGHBOR}_impute.hdf5"

hostname
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
echo "[INFO] task=${TASK_ID} dataset=${DATASET_ID} neighbor=${NEIGHBOR} mode=${MODE} cpu_num=20 dataset_dir=${DATASET_DIR}"
date

if [[ -s "${OUT}" && "${HIGASHI_FORCE_RERUN:-0}" != "1" ]]; then
  echo "[SKIP] existing output: ${OUT}"
  exit 0
fi
if [[ ! -s "${CONFIG}" ]]; then
  echo "[ERROR] missing config: ${CONFIG}" >&2
  exit 1
fi

if [[ "${HIGASHI_FORCE_RERUN:-0}" == "1" ]]; then
  rm -f "${OUT}" "${DENSE_OUT}" "${HDF5_OUT}"
  if [[ "${MODE}" == "resume0" ]]; then
    rm -f "${BASE_DIR}/input/${DATASET_DIR}/temp/model/model.chkpt_stage2" \
      "${BASE_DIR}/input/${DATASET_DIR}/temp/model/model.chkpt_stage2_model"
  fi
fi

"${PYTHON_BIN}" "${RUNNER}" \
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

date
echo "[OK] task finished: ${DATASET_ID} nbr${NEIGHBOR}"
