#!/usr/bin/env bash
#SBATCH -J ramani_hicimp
#SBATCH -A pi_limin_r
#SBATCH -p cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=UNLIMITED
#SBATCH --array=0-22
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/3_ramaniData/logs/ramani_hicimp_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/3_ramaniData/logs/ramani_hicimp_%A_%a.err

set -euo pipefail

BASE="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/3_ramaniData"
PYTHON="/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python"
RSCRIPT="${RSCRIPT:-Rscript}"
MANIFEST="${BASE}/input/ramani_hicimpute_manifest.tsv"

mkdir -p "${BASE}/logs" "${BASE}/output"
"${PYTHON}" "${BASE}/scripts/ramani_hicimpute_io.py" prepare --manifest "${MANIFEST}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ROW=$(awk -v n="$((TASK_ID + 2))" 'NR==n {print}' "${MANIFEST}")
IFS=$'\t' read -r chrom source_npz input_dir n_cells n_features n_bins <<< "${ROW}"

"${RSCRIPT}" "${BASE}/scripts/run_hicimpute_ramani_one.R" \
  --chrom "${chrom}" \
  --input-root "${BASE}/input" \
  --output-root "${BASE}/output" \
  --n-bins "${n_bins}" \
  --n-features "${n_features}" \
  --n-cells "${n_cells}" \
  --niter "${NITER:-5000}" \
  --burnin "${BURNIN:-1000}" \
  --mc-cores "${SLURM_CPUS_PER_TASK:-1}" \
  --seed "${SEED:-1234}"
