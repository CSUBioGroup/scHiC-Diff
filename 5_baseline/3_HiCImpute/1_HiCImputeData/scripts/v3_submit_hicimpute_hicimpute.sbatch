#!/usr/bin/env bash
#SBATCH --job-name=hicimpute_hicimp
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/1_HiCImputeData/logs/hicimpute_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/1_HiCImputeData/logs/hicimpute_%A_%a.log
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --time=UNLIMITED
#SBATCH --array=0-11

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/1_HiCImputeData/scripts"
RAW_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/0_gtData/0_downsampled_HiCImputeData"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/1_HiCImputeData/input"
OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/1_HiCImputeData/output"
PY_ENV="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python3"
RSCRIPT="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/Rscript"
NITER="${NITER:-50000}"
BURNIN="${BURNIN:-5000}"

mkdir -p "${SCRIPT_DIR}" "${INPUT_ROOT}" "${OUTPUT_ROOT}/bin" "${OUTPUT_ROOT}/rds" "${OUTPUT_ROOT}/npz_lower_tri"

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

echo "Started at: $(date)"
echo "Host: $(hostname) Task: ${TASK_ID} Dataset: ${STEM}"
echo "NITER=${NITER} BURNIN=${BURNIN}"
"${PY_ENV}" --version
"${RSCRIPT}" --version

IN_DIR="${INPUT_ROOT}/${STEM}"
if [[ ! -f "${IN_DIR}/.complete" ]]; then
  echo "===== STEP 1: PREPARE ====="
  "${PY_ENV}" "${SCRIPT_DIR}/v3_prepare_hicimpute_hicimputedata.py" \
    --raw-root "${RAW_ROOT}" \
    --input-root "${INPUT_ROOT}" \
    --datasets "${STEM}" \
    --overwrite
else
  echo "[skip] prepare, ${IN_DIR}/.complete exists"
fi

NPZ_PATH="${OUTPUT_ROOT}/npz_lower_tri/${STEM}_hicimpute_Impute_All_lower_tri.npz"
if [[ ! -f "${NPZ_PATH}" ]]; then
  echo "===== STEP 2: HICIMPUTE R ====="
  "${RSCRIPT}" "${SCRIPT_DIR}/v3_run_hicimpute_hicimputedata.R" \
    --dataset "${STEM}" \
    --input-root "${INPUT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --niter "${NITER}" \
    --burnin "${BURNIN}" \
    --mc-cores "${SLURM_CPUS_PER_TASK}" \
    --seed "$((1234 + TASK_ID))" \
    --python "${PY_ENV}"
else
  echo "[skip] R impute, ${NPZ_PATH} exists"
fi

echo "Finished at: $(date)"
