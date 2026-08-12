#!/usr/bin/env bash
#SBATCH --job-name=flam_20kb_coll
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/6_caseStudy/2_call_loops/scripts/logs/collect_%j.log
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/6_caseStudy/2_call_loops/scripts/logs/collect_%j.log
#SBATCH --account=pi_limin_r
#SBATCH --partition=cpuQ
#SBATCH --qos=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00

# Collect completed FLAMINGO tensors into per-chromosome sparse NPZ
# plus a dense (n_cells, n_bins, n_bins) tensor for APA / loop calling.
# Run AFTER the impute job finishes.
#
# Submit:
#   jid=$(sbatch --parsable 04_submit_impute.sbatch)
#   sbatch --dependency=afterok:${jid} 06_submit_collect.sbatch

set -euo pipefail
export LC_ALL=C LANG=C PYTHONUNBUFFERED=1

CASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/6_caseStudy/2_call_loops"
PYTHON="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python"

echo "=============================================="
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "=============================================="

"${PYTHON}" --version
"${PYTHON}" "${CASE_DIR}/scripts/03_collect_flamingo_to_npz.py" \
  --work-root "${CASE_DIR}/work" \
  --output-root "${CASE_DIR}/imputedData/FLAMINGO" \
  --overwrite

echo "=============================================="
echo "Finished at: $(date)"
echo "=============================================="