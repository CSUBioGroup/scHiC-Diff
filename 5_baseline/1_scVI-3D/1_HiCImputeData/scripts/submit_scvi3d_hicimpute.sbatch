#!/usr/bin/env bash
#SBATCH -J scvi3d_hicimp
#SBATCH -A pi_limin_r
#SBATCH -p gpu2Q
#SBATCH --qos=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-11
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/1_HiCImputeData/logs/scvi3d_hicimp_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/1_HiCImputeData/logs/scvi3d_hicimp_%A_%a.err

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/1_HiCImputeData/scripts"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/1_HiCImputeData/input"
OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/1_HiCImputeData/output"
PYTHON="/public/home/hpc254701055/micromamba/envs/2_schic-scvi-3d/bin/python"
IMPUTE_PY="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/run_scvi3d_flamingo.py"
RESOLUTION=1
N_BINS=61
N_LATENT=100
BAND_MAX="whole"

DATASETS=(
  "K562_T1_1k"
  "K562_T1_2k"
  "K562_T1_4k"
  "K562_T1_7k"
  "K562_T2_1k"
  "K562_T2_2k"
  "K562_T2_4k"
  "K562_T2_7k"
  "K562_T3_1k"
  "K562_T3_2k"
  "K562_T3_4k"
  "K562_T3_7k"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
STEM="${DATASETS[${TASK_ID}]}"
IN_DIR="${INPUT_ROOT}/${STEM}"
WORK_DIR="${OUTPUT_ROOT}/impute_work/${STEM}"
NPZ_OUT="${OUTPUT_ROOT}/npz_lower_tri/${STEM}_scVI3D_imputed.npz"

mkdir -p "${SCRIPT_DIR}" "${OUTPUT_ROOT}/impute_work" "${OUTPUT_ROOT}/npz_lower_tri"

echo "Started at: $(date)"
echo "Host: $(hostname) Task: ${TASK_ID} Dataset: ${STEM}"
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
echo "Finished at: $(date)"
