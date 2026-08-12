#!/usr/bin/env bash
#SBATCH --job-name=v3_hic_eval
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_scripts/logs/eval_%j.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_scripts/logs/eval_%j.log
#SBATCH --account=pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=UNLIMITED

set -euo pipefail
export LC_ALL=C
export LANG=C
export PYTHONUNBUFFERED=1

OUTPUT_ROOT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_outputData"
GT_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets"
EVAL_SCRIPT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/1_pccAndMae_all/13_cal_FLAMINGO_Baseline_metrics.py"
PY_ENV="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python3"

echo "Started at: $(date)"
echo "===== EVALUATE (raw) ====="
"${PY_ENV}" "${EVAL_SCRIPT}" \
  --v3-lower-tri-pred-dir "${OUTPUT_ROOT}/npz_lower_tri" \
  --v3-lower-tri-gt-dir "${GT_DIR}" \
  --v3-lower-tri-output-csv "${OUTPUT_ROOT}/metrics/HiCImpute_FLAMINGO_v3_raw.csv"

echo "===== EVALUATE (log1p) ====="
"${PY_ENV}" "${EVAL_SCRIPT}" \
  --v3-lower-tri-pred-dir "${OUTPUT_ROOT}/npz_lower_tri" \
  --v3-lower-tri-gt-dir "${GT_DIR}" \
  --v3-log1p \
  --v3-lower-tri-output-csv "${OUTPUT_ROOT}/metrics/HiCImpute_FLAMINGO_v3_log1p.csv"

echo "Finished at: $(date)"
