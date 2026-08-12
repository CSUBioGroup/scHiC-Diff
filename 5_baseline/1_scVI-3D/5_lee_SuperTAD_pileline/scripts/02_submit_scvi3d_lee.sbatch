#!/usr/bin/env bash
#SBATCH --job-name=lee_scvi3d
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/5_lee_SuperTAD_pileline/scripts/logs/scvi3d_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/5_lee_SuperTAD_pileline/scripts/logs/scvi3d_%A_%a.log
#SBATCH -A pi_limin_r
#SBATCH -p gpu4Q
#SBATCH --qos=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-3

# scVI-3D imputation for Lee SuperTAD pipeline
# 4 cell types, 1 GPU per array task
#
# Submit:
#   sbatch 02_submit_scvi3d_lee.sbatch

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/5_lee_SuperTAD_pileline/scripts"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/5_lee_SuperTAD_pileline/scvi3d_input"
OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/5_lee_SuperTAD_pileline/output"
PYTHON="/public/home/hpc254701055/micromamba/envs/2_schic-scvi-3d/bin/python"
IMPUTE_PY="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/run_scvi3d_flamingo.py"
RESOLUTION=10000
N_BINS=49
N_LATENT=100
BAND_MAX="whole"

DATASETS=("Astro" "Endo" "ODC" "OPC")
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
STEM="${DATASETS[${TASK_ID}]}"
IN_DIR="${INPUT_ROOT}/${STEM}"
WORK_DIR="${OUTPUT_ROOT}/impute_work/${STEM}"
NPZ_OUT="${OUTPUT_ROOT}/npz_lower_tri/${STEM}_scVI3D_imputed.npz"

mkdir -p "${SCRIPT_DIR}/logs" "${OUTPUT_ROOT}/impute_work" "${OUTPUT_ROOT}/npz_lower_tri"

echo "=============================================="
echo "Started at: $(date)"
echo "Host: $(hostname) Task: ${TASK_ID} Dataset: ${STEM}"
echo "=============================================="

"${PYTHON}" --version

if [[ -f "${NPZ_OUT}" && -f "${WORK_DIR}/done.flag" ]]; then
  echo "[SKIP] ${STEM} already complete"
  exit 0
fi

rm -f "${WORK_DIR}/done.flag"

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

touch "${WORK_DIR}/done.flag"
echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="