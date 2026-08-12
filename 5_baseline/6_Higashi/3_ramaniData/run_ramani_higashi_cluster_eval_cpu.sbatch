#!/usr/bin/env bash
#SBATCH -J ramani_hig_eval
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH --qos=cpuq
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/3_ramaniData/logs/ramani_hig_eval_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/3_ramaniData/logs/ramani_hig_eval_%j.err

set -euo pipefail

BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/3_ramaniData"
PYTHON="/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python"
SCRIPT="${BASE}/scripts/ramani_cluster_eval.py"
LOG_DIR="${BASE}/logs/cluster_eval_${SLURM_JOB_ID}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "${LOG_DIR}"

run_eval() {
  local method="$1"
  local chrom_dir="${BASE}/output/${method}/chrom_npz"
  local output_dir="${BASE}/output/${method}"
  local result_dir="${output_dir}/${method}_cluster_results"

  if [[ ! -d "${chrom_dir}" ]]; then
    echo "[ERROR] missing chrom_npz directory: ${chrom_dir}" >&2
    return 1
  fi

  echo "[START] ${method} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} time=$(date)"
  rm -rf "${result_dir}"
  "${PYTHON}" "${SCRIPT}" \
    --chrom-dir "${chrom_dir}" \
    --method-name "${method}" \
    --output-dir "${output_dir}"
  echo "[DONE] ${method} host=$(hostname) time=$(date)"
}

export BASE PYTHON SCRIPT LOG_DIR
export -f run_eval

echo "JOB ${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} started=$(date)"

srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  bash -lc 'run_eval higashi_nbr0' \
  > "${LOG_DIR}/higashi_nbr0.out" 2> "${LOG_DIR}/higashi_nbr0.err" &
pid0=$!

srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  bash -lc 'run_eval higashi_nbr5' \
  > "${LOG_DIR}/higashi_nbr5.out" 2> "${LOG_DIR}/higashi_nbr5.err" &
pid5=$!

wait "${pid0}"
wait "${pid5}"

echo "JOB ${SLURM_JOB_ID} finished=$(date)"
