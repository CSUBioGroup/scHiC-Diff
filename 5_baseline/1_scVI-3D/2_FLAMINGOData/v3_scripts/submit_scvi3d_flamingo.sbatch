#!/usr/bin/env bash
#SBATCH -J scvi3d_flamingo
#SBATCH -A pi_limin_r
#SBATCH -p gpu4Q
#SBATCH --qos=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-6
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/logs/scvi3d_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/logs/scvi3d_%A_%a.err

# Submit:
#   sbatch v3_scripts/submit_scvi3d_flamingo.sbatch
#
# 7 datasets, each on its own GPU (gpu4Q has 4 V100 per node).
# Each task: 1 GPU, 10 CPUs, 80G RAM.
# scVI-3D uses <1 GB VRAM, so GPU is underutilized but training is fastest
# (no GPU sharing overhead).

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_inputData"
OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_outputData"
PYTHON="/public/home/hpc254701055/micromamba/envs/2_schic-scvi-3d/bin/python"

IMPUTE_PY="${SCRIPT_DIR}/run_scvi3d_flamingo.py"
RESOLUTION=1000000
N_BINS=500
N_LATENT=100
BAND_MAX="whole"

mkdir -p "${SCRIPT_DIR}/logs" "${OUTPUT_ROOT}/impute_work" "${OUTPUT_ROOT}/npz_lower_tri"

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
STEM="${DATASETS[${TASK_ID}]}"

IN_DIR="${INPUT_ROOT}/${STEM}"
WORK_DIR="${OUTPUT_ROOT}/impute_work/${STEM}"
NPZ_OUT="${OUTPUT_ROOT}/npz_lower_tri/${STEM}_scVI3D_imputed.npz"

echo "Started at: $(date)"
echo "Host: $(hostname)  Task: ${TASK_ID}  Dataset: ${STEM}"
"${PYTHON}" --version

if [[ -f "${NPZ_OUT}" ]]; then
    echo "[SKIP] ${NPZ_OUT} already exists"
    exit 0
fi

echo "===== scVI-3D IMPUTE (band_max=${BAND_MAX}) ====="
"${PYTHON}" "${IMPUTE_PY}" \
    --input-dir "${IN_DIR}" \
    --output-dir "${WORK_DIR}" \
    --collect-dir "${OUTPUT_ROOT}/npz_lower_tri" \
    --stem "${STEM}" \
    --resolution "${RESOLUTION}" \
    --n-bins "${N_BINS}" \
    --band-max "${BAND_MAX}" \
    --n-latent "${N_LATENT}" \
    --gpu \
    --overwrite

echo "Finished at: $(date)"