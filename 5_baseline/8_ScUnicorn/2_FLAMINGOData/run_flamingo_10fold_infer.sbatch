#!/usr/bin/env bash
#SBATCH -J scuni_flam_infer
#SBATCH -A pi_limin_r
#SBATCH -p gpu2Q,gpu4Q,gpu8Q
#SBATCH --qos=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=1-10%10
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData/logs/%x_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData/logs/%x_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

BASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData"
PYTHON_BIN="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python"
FOLD_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
FOLD_NAME="$(printf 'fold_%02d' "${FOLD_ID}")"
CKPT="${BASE_DIR}/checkpoints/${FOLD_NAME}/scunicorn_${FOLD_NAME}_best.pytorch"

mkdir -p "${BASE_DIR}/logs" "${BASE_DIR}/output/${FOLD_NAME}"

if [[ ! -s "${CKPT}" ]]; then
  echo "[ERROR] missing checkpoint: ${CKPT}" >&2
  exit 1
fi

echo "== task =="
echo "fold=${FOLD_NAME}"
hostname
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true

"${PYTHON_BIN}" "${BASE_DIR}/scripts/flamingo_scunicorn_pipeline.py" infer-fold \
  --manifest "${BASE_DIR}/manifest.tsv" \
  --checkpoint "${CKPT}" \
  --output-root "${BASE_DIR}/output" \
  --datasets-root "${BASE_DIR}/datasets" \
  --fold-name "${FOLD_NAME}" \
  --patch-size "${SCUNICORN_PATCH_SIZE:-40}" \
  --batch-size "${SCUNICORN_INFER_BATCH_SIZE:-64}" \
  --cuda 0
