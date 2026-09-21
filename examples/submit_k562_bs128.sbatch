#!/usr/bin/env bash
#SBATCH -J k562_bs128
#SBATCH -A pi_limin_r
#SBATCH -p gpu2Q,gpu4Q,gpu8Q
#SBATCH --qos=gpuq
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --spread-job
#SBATCH --exclude=gpu207
#SBATCH --array=0-11%12
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/scHiC-Diff_github_import/logs/k562_bs128_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/scHiC-Diff_github_import/logs/k562_bs128_%A_%a.err

set -euo pipefail

REPO="/public/home/hpc254701055/2_projects/10_schicdiff/scHiC-Diff_github_import"

INPUT_DIR="${REPO}/5_baseline/7_scHiCDiff/1_HiCImputeData/input"
PYTHON_BIN="${PYTHON_BIN:-/public/home/hpc254701055/micromamba/envs/scdiff2/bin/python}"
SAVE_ROOT="${REPO}/results/training_results_v5fast_bs128"
LOG_DIR="${REPO}/logs/recon_masked_v5fast_bs128"

DATASETS=(
  K562_T1_1k K562_T1_2k K562_T1_4k K562_T1_7k
  K562_T2_1k K562_T2_2k K562_T2_4k K562_T2_7k
  K562_T3_1k K562_T3_2k K562_T3_4k K562_T3_7k
)

DS="${DATASETS[${SLURM_ARRAY_TASK_ID:?}]}"
H5AD="${INPUT_DIR}/${DS}_sim.h5ad"
SAVE="${SAVE_ROOT}/${DS}_sim"

echo "task=${SLURM_ARRAY_TASK_ID} dataset=${DS} host=$(hostname) save=${SAVE}"

if [[ -f "${SAVE}/denoise_recon_inv.npz" ]]; then
  echo "[SKIP] ${SAVE}/denoise_recon_inv.npz already exists"
  exit 0
fi

mkdir -p "${SAVE}" "${LOG_DIR}" "${REPO}/logs"
cd "${REPO}"

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

"${PYTHON_BIN}" main.py \
  -t True \
  --base configs/recon_masked.yaml \
  -n "${DS}_sim.seed10" \
  -l "${LOG_DIR}" \
  --save_path "${SAVE}" \
  data.params.batch_size=128 \
  data.params.test_batch_size=9999 \
  data.params.num_workers=4 \
  lightning.callbacks.early_stopping_callback.params.patience=25 \
  lightning.modelcheckpoint.params.every_n_epochs=50 \
  data.params.train.params.dataset=K562 \
  data.params.train.params.fname="${H5AD}" \
  data.params.validation.params.dataset=K562 \
  data.params.validation.params.fname="${H5AD}" \
  data.params.test.params.dataset=K562 \
  data.params.test.params.fname="${H5AD}"

echo "DONE ${DS} $(date)"
