#!/usr/bin/env bash
#SBATCH --job-name=v3_hicimpute
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_scripts/logs/impute_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_scripts/logs/impute_%A_%a.log
#SBATCH --account=pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=20
#SBATCH --mem=60G
#SBATCH --time=UNLIMITED
#SBATCH --array=0-6

set -euo pipefail
export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_scripts"
RAW_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/1_RawData/3_fixed_flamnigoGen"
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_inputData"
OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_outputData"
GT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets"
EVAL_SCRIPT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/1_pccAndMae_all/13_cal_FLAMINGO_Baseline_metrics.py"

PY_ENV="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python3"
RSCRIPT="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/Rscript"

# MCMC parameters (production: niter=50000). Override via NITER/BURNIN env.
NITER="${NITER:-50000}"
BURNIN="${BURNIN:-5000}"

mkdir -p "${SCRIPT_DIR}/logs" "${INPUT_ROOT}" \
         "${OUTPUT_ROOT}/bin" "${OUTPUT_ROOT}/rds" \
         "${OUTPUT_ROOT}/npz_lower_tri" "${OUTPUT_ROOT}/metrics"

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

echo "Started at: $(date)"
echo "Host: $(hostname)  Task: ${TASK_ID}  Dataset: ${STEM}"
echo "NITER=${NITER}  BURNIN=${BURNIN}"
"${PY_ENV}" --version 2>&1 | tail -1

# 1) Prepare input (raw -> R col-major binary)
IN_DIR="${INPUT_ROOT}/${STEM}"
if [[ ! -f "${IN_DIR}/.complete" ]]; then
  echo "===== STEP 1: PREPARE ====="
  "${PY_ENV}" "${SCRIPT_DIR}/v3_prepare_hicimpute_flamingo.py" \
    --raw-root "${RAW_ROOT}" --input-root "${INPUT_ROOT}" \
    --datasets "${STEM}" --overwrite
else
  echo "[skip] prepare, ${IN_DIR}/.complete exists"
fi

# 2) Run HiCImpute R (impute + collect into numpy row-major npz)
NPZ_PATH="${OUTPUT_ROOT}/npz_lower_tri/${STEM}_hicimpute_Impute_All_lower_tri.npz"
if [[ ! -f "${NPZ_PATH}" ]]; then
  echo "===== STEP 2: HiCImpute R ====="
  "${RSCRIPT}" "${SCRIPT_DIR}/v3_run_hicimpute_flamingo.R" \
    --dataset "${STEM}" \
    --input-root "${INPUT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --niter "${NITER}" --burnin "${BURNIN}" \
    --mc-cores "${SLURM_CPUS_PER_TASK}" \
    --seed "$((1234 + TASK_ID))" \
    --python "${PY_ENV}"
else
  echo "[skip] R impute, ${NPZ_PATH} exists"
fi

echo "Finished at: $(date)"