#!/usr/bin/env bash
#SBATCH -J ramani_scdiff_v12
#SBATCH -A pi_limin_r
#SBATCH -p gpu2Q
#SBATCH --qos=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=10G
#SBATCH --time=UNLIMITED
#SBATCH --array=0-5
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/3_ramaniData/logs/ramani_scdiff_v12_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/3_ramaniData/logs/ramani_scdiff_v12_%A_%a.err

set -uo pipefail
export PYTHONUNBUFFERED=1

BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/3_ramaniData"
PYTHON="/public/home/hpc254701055/micromamba/envs/scdiff2/bin/python"
PROJECT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/3_DiffusionModel/scHiC-Diff-master"
CONFIG_PATH="configs/recon_masked_ramani_3000.yaml"
MANIFEST="${BASE}/input/ramani_scdiff_h5ad_manifest.csv"
RESULT_ROOT="${BASE}/output/training_results_v12"
LOGDIR="${PROJECT_DIR}/logs"
MAX_EPOCHS="${MAX_EPOCHS:-3000}"

# 23 chromosomes split into 6 groups of 4 (last group has 3)
CHROM_GROUPS=(
  "chr1 chr2 chr3 chr4"
  "chr5 chr6 chr7 chr8"
  "chr9 chr10 chr11 chr12"
  "chr13 chr14 chr15 chr16"
  "chr17 chr18 chr19 chr20"
  "chr21 chr22 chrX"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
read -r -a CHROMS <<< "${CHROM_GROUPS[$TASK_ID]}"
NUM_WORKERS_PER_TASK=10

mkdir -p "${BASE}/logs" "${BASE}/output" "${RESULT_ROOT}" "${LOGDIR}"

hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "[INFO] code=v1.2, config=${CONFIG_PATH}, max_epochs=${MAX_EPOCHS}, no early stopping"
echo "[INFO] ${#CHROMS[@]} tasks parallel on 1 GPU, num_workers=${NUM_WORKERS_PER_TASK} each"
echo "[INFO] task=${TASK_ID} chroms=${CHROMS[*]}"
date

# Build chrom -> h5ad map from manifest
declare -A H5AD_MAP
while IFS=',' read -r chrom h5ad n_cells n_features n_bins; do
  [[ "$chrom" == "chrom" ]] && continue
  H5AD_MAP["$chrom"]="$h5ad"
done < "${MANIFEST}"

pids=(); names=()

run_one() {
  local chrom="$1"
  local h5ad="${H5AD_MAP[$chrom]}"
  local save_path="${RESULT_ROOT}/${chrom}"

  mkdir -p "${save_path}"
  (
    set -e
    cd "${PROJECT_DIR}"
    "${PYTHON}" main.py \
      -b "${CONFIG_PATH}" \
      --train True \
      --seed "${SEED:-10}" \
      --save_path "${save_path}" \
      --logdir "${LOGDIR}" \
      --postfix "ramani_v12_${chrom}.seed${SEED:-10}" \
      "lightning.trainer.devices=[0]" \
      "lightning.trainer.max_epochs=${MAX_EPOCHS}" \
      "data.params.num_workers=${NUM_WORKERS_PER_TASK}" \
      "data.params.batch_size=1024" \
      "lightning.modelcheckpoint.params.every_n_epochs=500" \
      "lightning.modelcheckpoint.params.save_top_k=-1" \
      "data.params.train.params.dataset=Ramani" \
      "data.params.train.params.fname=${h5ad}" \
      "data.params.validation.params.dataset=Ramani" \
      "data.params.validation.params.fname=${h5ad}" \
      "data.params.test.params.dataset=Ramani" \
      "data.params.test.params.fname=${h5ad}" \
      > "${save_path}/train.log" 2>&1
    touch "${save_path}/done.flag"
  ) &
  pids+=($!); names+=("${chrom}")
}

for chrom in "${CHROMS[@]}"; do
  run_one "$chrom"
done

echo "Launched ${#pids[@]} parallel tasks: ${names[*]}"

fail=0
for j in "${!pids[@]}"; do
  wait "${pids[$j]}" || { echo "[FAIL] ${names[$j]}"; fail=1; }
done
[[ $fail -eq 0 ]] || { echo "SOME RUN FAILED"; exit 1; }
echo "ALL_DONE"; date