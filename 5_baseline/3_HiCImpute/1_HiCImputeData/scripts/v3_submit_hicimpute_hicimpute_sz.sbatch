#!/usr/bin/env bash
#SBATCH --job-name=hicimpute_sz
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/1_HiCImputeData/logs/hicimpute_sz_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/1_HiCImputeData/logs/hicimpute_sz_%A_%a.log
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
INPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/1_HiCImputeData/input"
OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/1_HiCImputeData/output"
PY_ENV="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python3"
RSCRIPT="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/Rscript"
NITER="5000"
BURNIN="1000"

mkdir -p "${OUTPUT_ROOT}/npz_dxy"

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
echo "NITER=${NITER} BURNIN=${BURNIN} (Impute_SZ output)"

"${RSCRIPT}" "${SCRIPT_DIR}/v3_run_hicimpute_hicimputedata_sz.R" \
  --dataset "${STEM}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --niter "${NITER}" \
  --burnin "${BURNIN}" \
  --mc-cores "${SLURM_CPUS_PER_TASK}" \
  --seed "$((1234 + TASK_ID))" \
  --python "${PY_ENV}"

echo "Finished at: $(date)"