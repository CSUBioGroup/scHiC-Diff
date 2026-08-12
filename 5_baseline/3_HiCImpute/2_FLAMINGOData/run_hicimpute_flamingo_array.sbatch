#!/usr/bin/env bash
#SBATCH -J hicimp_flam
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-35%4
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGO/logs/slurm_%x_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGO/logs/slurm_%x_%A_%a.err

set -euo pipefail

export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

BASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGO"
PYTHON_BIN="${PYTHON_BIN:-/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python}"
RSCRIPT_BIN="${RSCRIPT_BIN:-/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/Rscript}"
IO_SCRIPT="${BASE_DIR}/scripts/flamingo_hicimpute_io.py"
R_SCRIPT="${BASE_DIR}/scripts/run_hicimpute_one.R"
MANIFEST="${BASE_DIR}/input/manifest.tsv"
INPUT_ROOT="${BASE_DIR}/input"
OUTPUT_ROOT="${BASE_DIR}/output"

mkdir -p "${BASE_DIR}/logs" "${OUTPUT_ROOT}/rds" "${OUTPUT_ROOT}/bin" "${OUTPUT_ROOT}/npz_lower_tri"

if [[ ! -s "${MANIFEST}" ]]; then
  "${PYTHON_BIN}" "${IO_SCRIPT}" write-manifest --manifest "${MANIFEST}"
fi

"${PYTHON_BIN}" "${IO_SCRIPT}" prep \
  --manifest "${MANIFEST}" \
  --input-root "${INPUT_ROOT}" \
  --task-id "${SLURM_ARRAY_TASK_ID}"

DATASET="$(awk -F '\t' -v i="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==i {print $1}' "${MANIFEST}")"
N_CELLS="$(awk -F '\t' -v i="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==i {print $4}' "${MANIFEST}")"
N_FEATURES="$(awk -F '\t' -v i="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==i {print $5}' "${MANIFEST}")"
N_BEADS="$(awk -F '\t' -v i="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==i {print $6}' "${MANIFEST}")"

echo "task_id=${SLURM_ARRAY_TASK_ID}"
echo "dataset=${DATASET}"
echo "node=$(hostname)"
echo "python=${PYTHON_BIN}"
echo "rscript=${RSCRIPT_BIN}"
echo "niter=${HICIMPUTE_NITER:-5000}"
echo "burnin=${HICIMPUTE_BURNIN:-1000}"
echo "mc_cores=${HICIMPUTE_MC_CORES:-${SLURM_CPUS_PER_TASK:-1}}"

"${RSCRIPT_BIN}" "${R_SCRIPT}" \
  --dataset "${DATASET}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --n-beads "${N_BEADS}" \
  --n-features "${N_FEATURES}" \
  --n-cells "${N_CELLS}" \
  --niter "${HICIMPUTE_NITER:-5000}" \
  --burnin "${HICIMPUTE_BURNIN:-1000}" \
  --mc-cores "${HICIMPUTE_MC_CORES:-${SLURM_CPUS_PER_TASK:-1}}" \
  --seed "$((1234 + SLURM_ARRAY_TASK_ID))"

"${PYTHON_BIN}" "${IO_SCRIPT}" collect \
  --manifest "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset "${DATASET}" \
  --mode Impute_All

"${PYTHON_BIN}" "${IO_SCRIPT}" collect \
  --manifest "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset "${DATASET}" \
  --mode Impute_SZ
