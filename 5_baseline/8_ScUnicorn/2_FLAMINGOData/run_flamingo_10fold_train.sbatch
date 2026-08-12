#!/usr/bin/env bash
#SBATCH -J scuni_flam_train
#SBATCH -A pi_limin_r
#SBATCH -p gpu2Q,gpu4Q,gpu8Q
#SBATCH --qos=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=1-10%10
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData/logs/%x_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData/logs/%x_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData"
REF_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/4_ImputationCriteria/benchmark_criteria_master/8_ScUnicron"
PYTHON_BIN="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python"
FOLD_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
FOLD_NAME="$(printf 'fold_%02d' "${FOLD_ID}")"
FOLD_DIR="${BASE_DIR}/splits/${FOLD_NAME}"
DATA_ROOT="${BASE_DIR}/datasets/${FOLD_NAME}"
CKPT_DIR="${BASE_DIR}/checkpoints/${FOLD_NAME}"
CKPT="${CKPT_DIR}/scunicorn_${FOLD_NAME}_best.pytorch"

mkdir -p "${BASE_DIR}/logs" "${DATA_ROOT}" "${CKPT_DIR}"

echo "== task =="
echo "fold=${FOLD_NAME}"
hostname
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true

echo "== build fold dataset =="
"${PYTHON_BIN}" "${BASE_DIR}/scripts/flamingo_scunicorn_pipeline.py" build-fold \
  --manifest "${BASE_DIR}/manifest.tsv" \
  --fold-dir "${FOLD_DIR}" \
  --output-root "${DATA_ROOT}" \
  --patch-size "${SCUNICORN_PATCH_SIZE:-40}"

echo "== train =="
"${PYTHON_BIN}" "${REF_DIR}/trainAll/train_scunicorn_k562.py" \
  --data-root "${DATA_ROOT}" \
  --save-path "${CKPT}" \
  --epochs "${SCUNICORN_EPOCHS:-50}" \
  --batch-size "${SCUNICORN_BATCH_SIZE:-64}" \
  --lr "${SCUNICORN_LR:-0.0003}" \
  --cuda 0
