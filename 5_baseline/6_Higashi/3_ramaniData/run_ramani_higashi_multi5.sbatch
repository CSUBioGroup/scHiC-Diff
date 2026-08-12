#!/usr/bin/env bash
#SBATCH -J ramani_hig_m5
#SBATCH -A pi_limin_r
#SBATCH -p gpu4Q
#SBATCH --qos=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=UNLIMITED
#SBATCH --array=0-0
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/3_ramaniData/logs/multi5_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/3_ramaniData/logs/multi5_%A_%a.err

set -uo pipefail

BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/3_ramaniData"
HIGASHI_PYTHON="/public/home/hpc254701055/micromamba/envs/6_higashi/bin/python"
RUNNER="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/1_HiCImputeData/scripts/run_higashi_one.py"
TASKS="${BASE}/input/ramani_higashi_multi5_pending.tsv"
LOG_DIR="${BASE}/logs/multi5_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
DONE_DIR="${BASE}/logs/multi5_done"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

mkdir -p "${LOG_DIR}" "${DONE_DIR}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?}"
START=$((TASK_ID * 5))
END=$((START + 4))

echo "GPU_TASK=${TASK_ID} host=$(hostname) range=${START}-${END} started=$(date)"

pids=()
names=()

run_one() {
  local list_idx="$1"
  local line_no=$((list_idx + 2))
  local row
  row=$(awk -v n="${line_no}" 'NR==n {print}' "${TASKS}")
  if [[ -z "${row}" ]]; then
    return 0
  fi

  local nbr chrom_index chrom config dataset_root final_hdf5
  IFS=$'\t' read -r nbr chrom_index chrom config dataset_root final_hdf5 <<< "${row}"
  local name="nbr${nbr}_${chrom}"
  local done_flag="${DONE_DIR}/${name}.done"
  local out_log="${LOG_DIR}/${name}.out"
  local err_log="${LOG_DIR}/${name}.err"

  if [[ -s "${final_hdf5}" || -f "${done_flag}" ]]; then
    echo "[SKIP] ${name} final_hdf5_or_done_exists"
    return 0
  fi

  (
    set -euo pipefail
    echo "[START] ${name} config=${config} final=${final_hdf5} time=$(date)"
    local runner_args=(
      --config "${config}"
      --training-updates "${TRAINING_UPDATES:-1000}"
      --eval-updates "${EVAL_UPDATES:-10}"
    )
    if [[ -s "${dataset_root}/temp/node_feats.hdf5" ]]; then
      runner_args+=(--skip-process)
      echo "[INFO] ${name} using existing node_feats.hdf5; skip process"
    else
      echo "[INFO] ${name} node_feats.hdf5 missing; running process step"
    fi
    "${HIGASHI_PYTHON}" "${RUNNER}" "${runner_args[@]}"
    if [[ ! -s "${final_hdf5}" ]]; then
      echo "[ERROR] ${name} missing final hdf5: ${final_hdf5}" >&2
      exit 1
    fi
    touch "${done_flag}"
    echo "[DONE] ${name} time=$(date)"
  ) > "${out_log}" 2> "${err_log}" &

  pids+=("$!")
  names+=("${name}")
}

for list_idx in $(seq "${START}" "${END}"); do
  run_one "${list_idx}"
done

echo "Launched ${#pids[@]} tasks: ${names[*]}"

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[FAIL] ${names[$i]}"
    fail=1
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "SOME RUN FAILED GPU_TASK=${TASK_ID} finished=$(date)"
  exit 1
fi

echo "ALL_DONE GPU_TASK=${TASK_ID} finished=$(date)"
