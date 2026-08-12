#!/usr/bin/env bash
#SBATCH -J tan_cluster_ari
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-0
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/4_tanClusterAndARI_pipeline/output/logs/slurm_%x_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/4_tanClusterAndARI_pipeline/output/logs/slurm_%x_%A_%a.err

set -euo pipefail

export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

PIPELINE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/4_tanClusterAndARI_pipeline"
PYTHON_BIN="${PYTHON_BIN:-/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python}"

mkdir -p "${PIPELINE_DIR}/output/logs"

echo "job_id=${SLURM_JOB_ID:-NA}"
echo "node=$(hostname)"
echo "start=$(date -Is)"

cd "${PIPELINE_DIR}"
"${PYTHON_BIN}" cluster_ARI_confusion_matrix.py --methods all

echo "end=$(date -Is)"
