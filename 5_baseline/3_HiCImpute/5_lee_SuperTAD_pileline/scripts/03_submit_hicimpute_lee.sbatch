#!/usr/bin/env bash
#SBATCH --job-name=lee_hicimpute
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/5_lee_SuperTAD_pileline/scripts/logs/hicimpute_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/5_lee_SuperTAD_pileline/scripts/logs/hicimpute_%A_%a.log
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --array=0-3

# HiCImpute MCMC imputation for Lee SuperTAD pipeline
# 4 cell types (Astro/Endo/ODC/OPC), 1 per array task
#
# Submit:
#   NITER=5000 BURNIN=1000 sbatch 03_submit_hicimpute_lee.sbatch

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/5_lee_SuperTAD_pileline/scripts"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/5_lee_SuperTAD_pileline/hicimpute_input"
OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/5_lee_SuperTAD_pileline/output"
PY_ENV="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python3"
RSCRIPT="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/Rscript"
NITER="${NITER:-5000}"
BURNIN="${BURNIN:-1000}"

mkdir -p "${SCRIPT_DIR}/logs" "${INPUT_ROOT}" "${OUTPUT_ROOT}/bin" "${OUTPUT_ROOT}/rds" "${OUTPUT_ROOT}/npz_triu"

DATASETS=("Astro" "Endo" "ODC" "OPC")
TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
STEM="${DATASETS[${TASK_ID}]}"

echo "=============================================="
echo "Started at: $(date)"
echo "Host: $(hostname) Task: ${TASK_ID} Dataset: ${STEM}"
echo "NITER=${NITER} BURNIN=${BURNIN} CPUs: ${SLURM_CPUS_PER_TASK}"
echo "=============================================="

"${PY_ENV}" --version
"${RSCRIPT}" --version

IN_DIR="${INPUT_ROOT}/${STEM}"
if [[ ! -f "${IN_DIR}/.complete" ]]; then
  echo "===== STEP 1: PREPARE ====="
  "${PY_ENV}" "${SCRIPT_DIR}/01_prepare_lee_hicimpute.py" \
    --cell-types "${STEM}" \
    --overwrite
else
  echo "[skip] prepare, ${IN_DIR}/.complete exists"
fi

NPZ_PATH="${OUTPUT_ROOT}/npz_triu/${STEM}_hicimpute_Impute_All_triu.npz"
if [[ ! -f "${NPZ_PATH}" ]]; then
  echo "===== STEP 2: HICIMPUTE R ====="
  "${RSCRIPT}" "${SCRIPT_DIR}/02_run_hicimpute_lee.R" \
    --dataset "${STEM}" \
    --input-root "${INPUT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --n-beads 49 \
    --niter "${NITER}" \
    --burnin "${BURNIN}" \
    --mc-cores "${SLURM_CPUS_PER_TASK}" \
    --seed "$((1234 + TASK_ID))" \
    --python "${PY_ENV}"
else
  echo "[skip] R impute, ${NPZ_PATH} exists"
fi

echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="