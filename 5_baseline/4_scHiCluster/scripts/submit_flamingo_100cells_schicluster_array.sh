#!/usr/bin/env bash
#SBATCH --job-name=schicluster_100cells
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/0_scHiCDiff/logs/schicluster_100cells_%A_%a.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/0_scHiCDiff/logs/schicluster_100cells_%A_%a.log
#SBATCH --account=pi_limin
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-36%6

set -euo pipefail

export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=8

WORK_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/0_scHiCDiff"
SCRIPT_DIR="${WORK_DIR}/scripts"
ENV_DIR="/public/home/hpc254701055/micromamba/envs/3_schicluster_python38"
PYTHON="${ENV_DIR}/bin/python"
HICLUSTER="${ENV_DIR}/bin/hicluster"
MANIFEST="${WORK_DIR}/manifest.tsv"

mkdir -p "${WORK_DIR}/logs" "${WORK_DIR}/output"
export PATH="${ENV_DIR}/bin:${PATH}"

DATASET="$(awk -v task="${SLURM_ARRAY_TASK_ID}" 'NR == task + 1 {print $1}' "${MANIFEST}")"
if [[ -z "${DATASET}" ]]; then
  echo "No dataset found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"

echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "Task: ${SLURM_ARRAY_TASK_ID}"
echo "Dataset: ${DATASET}"
echo "Python: ${PYTHON}"
"${PYTHON}" --version

"${PYTHON}" process_flamingo_100cells_schicluster.py \
  --work-dir "${WORK_DIR}" \
  --datasets "${DATASET}" \
  --skip-prepare \
  --chrom chr19 \
  --resolution 1 \
  --pad 1 \
  --std 1 \
  --rp 0.5 \
  --tol 0.01 \
  --window-size 500 \
  --step-size 500 \
  --output-format npz \
  --hicluster "${HICLUSTER}"

echo "Finished at: $(date)"
