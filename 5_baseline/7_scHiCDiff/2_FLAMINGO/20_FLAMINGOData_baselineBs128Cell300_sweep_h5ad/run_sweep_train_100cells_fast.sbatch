#!/usr/bin/env bash
#SBATCH -J sweep100_fast
#SBATCH -A pi_limin_r
#SBATCH -p gpu4Q
#SBATCH --qos=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=UNLIMITED
#SBATCH --array=0-5
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/0_retention_datasets/retention_sweep_100cells_h5ad/logs/sweep_fast_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/0_retention_datasets/retention_sweep_100cells_h5ad/logs/sweep_fast_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

SWEEP_H5AD_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/0_retention_datasets/retention_sweep_100cells_h5ad"
HELDOUT_NPZ="${SWEEP_H5AD_DIR}/fixed_heldout_indices.npz"
MANIFEST="${SWEEP_H5AD_DIR}/sweep_h5ad_manifest.csv"

PROJECT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/v5_scdiff_fast/3_DiffusionModel/scHiC-Diff-master"
PYTHON_EXEC="/public/home/hpc254701055/micromamba/envs/scdiff2/bin/python"
CONFIG_PATH="configs/recon_masked.yaml"
RESULT_ROOT="${SWEEP_H5AD_DIR}/training_results_fast"
LOGDIR="${PROJECT_DIR}/logs"

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
ROW_IDX=$((TASK_ID + 2))
H5AD_PATH="$(awk -F',' -v idx="${ROW_IDX}" 'NR==idx{print $3}' "${MANIFEST}")"
RETENTION="$(awk -F',' -v idx="${ROW_IDX}" 'NR==idx{print $1}' "${MANIFEST}")"
TAG="$(awk -F',' -v idx="${ROW_IDX}" 'NR==idx{print $2}' "${MANIFEST}")"
CASE_NAME="beads_500_W0.7_level1_${TAG}_100cells"
SAVE_PATH="${RESULT_ROOT}/${CASE_NAME}"

mkdir -p "${SWEEP_H5AD_DIR}/logs" "${RESULT_ROOT}" "${SAVE_PATH}" "${LOGDIR}"

hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "[INFO] FAST variant: val=5epoch, no GradNorm, no EarlyStopping"
echo "[INFO] task=${TASK_ID} retention=${RETENTION} tag=${TAG} cells=300"
echo "[INFO] h5ad=${H5AD_PATH}"
echo "[INFO] save_path=${SAVE_PATH}"
date

# Apply fixed heldout mask
HELDOUT_H5AD="${SWEEP_H5AD_DIR}/${CASE_NAME}_heldout_masked_fast.h5ad"
"${PYTHON_EXEC}" -c "
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
a = ad.read_h5ad('${H5AD_PATH}')
heldout = np.load('${HELDOUT_NPZ}')['heldout']
X = a.X.toarray().copy()
for r, c in heldout:
    if r < X.shape[0] and c < X.shape[1]:
        X[r, c] = 0.0
a.X = csr_matrix(X)
if 'counts' in a.layers:
    a.layers['counts'] = csr_matrix(X)
a.write_h5ad('${HELDOUT_H5AD}')
print(f'Saved: {X.shape}, heldout={len(heldout)} zeroed')
"

cd "${PROJECT_DIR}"

data_args=(
    "data.params.train.params.dataset=FLAMINGO"
    "data.params.train.params.fname=${HELDOUT_H5AD}"
    "data.params.validation.params.dataset=FLAMINGO"
    "data.params.validation.params.fname=${HELDOUT_H5AD}"
    "data.params.test.params.dataset=FLAMINGO"
    "data.params.test.params.fname=${HELDOUT_H5AD}"
)

"${PYTHON_EXEC}" main.py \
    -b "${CONFIG_PATH}" \
    --train True \
    --seed 10 \
    --save_path "${SAVE_PATH}" \
    --logdir "${LOGDIR}" \
    --postfix "sweep100_fast_${CASE_NAME}.seed10" \
    "lightning.trainer.devices=[0]" \
    "lightning.trainer.max_epochs=500" \
    "data.params.num_workers=10" \
    "${data_args[@]}"

echo "[DONE] FAST retention=${RETENTION} cells=300 finished"
date