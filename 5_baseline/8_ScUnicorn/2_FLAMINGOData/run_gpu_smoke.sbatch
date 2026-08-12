#!/usr/bin/env bash
#SBATCH -J scuni_flam_smoke
#SBATCH -A pi_limin_r
#SBATCH -p gpu2Q,gpu4Q,gpu8Q
#SBATCH --qos=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData/logs/%x_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData/logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

BASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/8_ScUnicorn/2_FLAMINGOData"
REF_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/4_ImputationCriteria/benchmark_criteria_master/8_ScUnicron"
PYTHON_BIN="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python"
FOLD_NAME="fold_01"
DATA_ROOT="${BASE_DIR}/datasets_smoke/${FOLD_NAME}"
CKPT="${BASE_DIR}/checkpoints_smoke/${FOLD_NAME}/scunicorn_${FOLD_NAME}_best.pytorch"
OUT_ROOT="${BASE_DIR}/output_smoke"

mkdir -p "${BASE_DIR}/logs" "$(dirname "${CKPT}")" "${OUT_ROOT}"
rm -rf "${OUT_ROOT}/${FOLD_NAME}"

echo "== node =="
hostname
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
PY

echo "== train smoke =="
"${PYTHON_BIN}" "${REF_DIR}/trainAll/train_scunicorn_k562.py" \
  --data-root "${DATA_ROOT}" \
  --save-path "${CKPT}" \
  --epochs 1 \
  --batch-size 32 \
  --lr 0.0003 \
  --cuda 0

echo "== infer smoke =="
"${PYTHON_BIN}" "${BASE_DIR}/scripts/flamingo_scunicorn_pipeline.py" infer-fold \
  --manifest "${BASE_DIR}/manifest.tsv" \
  --checkpoint "${CKPT}" \
  --output-root "${OUT_ROOT}" \
  --datasets-root "${BASE_DIR}/datasets_smoke" \
  --fold-name "${FOLD_NAME}" \
  --patch-size 40 \
  --batch-size 32 \
  --cuda 0

echo "== outputs =="
find "${OUT_ROOT}" -maxdepth 4 -type f -printf '%p %s\n' | sort
