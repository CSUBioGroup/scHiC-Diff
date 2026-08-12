#!/usr/bin/env bash
#SBATCH -J paramsweep_v5fast_bt_bs1500
#SBATCH -A pi_limin_r
#SBATCH -p gpu2Q
#SBATCH --qos=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --time=UNLIMITED
#SBATCH --array=0-5
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets/logs/v5fast_batched_test_paramsweep_bs1500_testbs9999_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets/logs/v5fast_batched_test_paramsweep_bs1500_testbs9999_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

DATA_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets"
HELDOUT_NPZ="${DATA_DIR}/fixed_heldout_indices.npz"
PYTHON_EXEC="/public/home/hpc254701055/micromamba/envs/scdiff2/bin/python"
PROJECT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/v5_scdiff_fast_batched_test/3_DiffusionModel/scHiC-Diff-master"
CONFIG_PATH="configs/recon_masked.yaml"
BATCH_SIZE=1500
TEST_BATCH_SIZE=9999
WORKERS=20
RUN_TAG="v5fast_batched_test_bs1500_testbs9999"
RESULT_ROOT="${DATA_DIR}/training_results_v5fast_batched_test_bs1500_testbs9999"

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
H5AD_NAMES=(
    "v3_hybrid_W0p5_500cells_level0_scdiff2.h5ad"
    "v3_hybrid_W0p6_500cells_level0_scdiff2.h5ad"
    "v3_hybrid_W0p7_500cells_level0_r0p01_scdiff2.h5ad"
    "v3_hybrid_W0p7_500cells_level0_r0p05_scdiff2.h5ad"
    "v3_hybrid_W0p8_500cells_level0_scdiff2.h5ad"
    "v3_hybrid_W0p9_500cells_level0_scdiff2.h5ad"
)

if (( TASK_ID < 0 || TASK_ID >= ${#H5AD_NAMES[@]} )); then
    echo "[ERROR] Unsupported array task: ${TASK_ID}" >&2
    exit 2
fi

H5AD_NAME="${H5AD_NAMES[$TASK_ID]}"
H5AD_PATH="${DATA_DIR}/${H5AD_NAME}"
CASE_NAME="${H5AD_NAME%.h5ad}"
SAVE_PATH="${RESULT_ROOT}/${CASE_NAME}_${RUN_TAG}"
LOGDIR="${PROJECT_DIR}/logs"
HELDOUT_H5AD="${DATA_DIR}/${CASE_NAME}_heldout_masked_${RUN_TAG}.h5ad"

mkdir -p "${DATA_DIR}/logs" "${RESULT_ROOT}" "${SAVE_PATH}" "${LOGDIR}"

hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "[INFO] variant=v5fast_batched_test"
echo "[INFO] task=${TASK_ID}"
echo "[INFO] partition=${SLURM_JOB_PARTITION:-unset} cpus-per-task=${SLURM_CPUS_PER_TASK:-unset} workers=${WORKERS}"
echo "[INFO] batch_size=${BATCH_SIZE}"
echo "[INFO] test_batch_size=${TEST_BATCH_SIZE}"
echo "[INFO] h5ad=${H5AD_PATH}"
echo "[INFO] heldout=${HELDOUT_H5AD}"
echo "[INFO] save_path=${SAVE_PATH}"
date

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

"${PYTHON_EXEC}" main.py \
    -b "${CONFIG_PATH}" \
    --train True \
    --seed 10 \
    --save_path "${SAVE_PATH}" \
    --logdir "${LOGDIR}" \
    --postfix "${RUN_TAG}_${CASE_NAME}.seed10" \
    "lightning.trainer.devices=[0]" \
    "lightning.trainer.max_epochs=500" \
    "data.params.batch_size=${BATCH_SIZE}" \
    "data.params.test_batch_size=${TEST_BATCH_SIZE}" \
    "data.params.num_workers=${WORKERS}" \
    "data.params.train.params.dataset=FLAMINGO" \
    "data.params.train.params.fname=${HELDOUT_H5AD}" \
    "data.params.validation.params.dataset=FLAMINGO" \
    "data.params.validation.params.fname=${HELDOUT_H5AD}" \
    "data.params.test.params.dataset=FLAMINGO" \
    "data.params.test.params.fname=${HELDOUT_H5AD}"

echo "[DONE] v5fast_batched_test ${H5AD_NAME} finished"
date
