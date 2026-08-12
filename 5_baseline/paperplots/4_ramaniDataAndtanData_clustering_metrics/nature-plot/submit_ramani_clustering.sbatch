#!/usr/bin/env bash
#SBATCH -J ramani_cluster
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH --qos=cpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ramani_clustering_%j.out
#SBATCH --error=logs/ramani_clustering_%j.err

set -euo pipefail

PLOT_DIR="${SLURM_SUBMIT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/micromamba/envs/2_schic-scvi-3d/bin/python}"
MODE="${MODE:-plots}"
FIGURES="${FIGURES:-all}"

if [[ "${MODE}" != "all" && "${MODE}" != "metrics" && "${MODE}" != "plots" ]]; then
  echo "MODE must be one of: all, metrics, plots" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export JOBLIB_TEMP_FOLDER="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"

cd "${PLOT_DIR}"
mkdir -p logs results/schicluster_office_code_Ramani figures

if [[ "${MODE}" == "all" || "${MODE}" == "metrics" ]]; then
  "${PYTHON_BIN}" schicluster_office_code_generate_ramani_cell_embeddings.py \
    --manifest RamaniData_clustering_input_paths.csv \
    --output-dir results/schicluster_office_code_Ramani \
    --dimensions 64,128 \
    --svd-seed 100

  "${PYTHON_BIN}" schicluster_office_code_calculate_ramani_ari.py \
    --embedding-root results/schicluster_office_code_Ramani \
    --manifest RamaniData_clustering_input_paths.csv \
    --output-dir results/schicluster_office_code_Ramani

  "${PYTHON_BIN}" schicluster_office_code_calculate_ramani_plot_ari.py \
    --official-ari results/schicluster_office_code_Ramani/schicluster_office_code_Ramani_ARI_long.csv \
    --output-dir results/schicluster_office_code_Ramani

  "${PYTHON_BIN}" schicluster_office_code_generate_ramani_umap_coordinates.py \
    --embedding-root results/schicluster_office_code_Ramani \
    --manifest RamaniData_clustering_input_paths.csv \
    --output results/schicluster_office_code_Ramani/schicluster_office_code_Ramani_cluster_coordinates.csv

  "${PYTHON_BIN}" calculate_tan_confusion.py \
    --manifest TanData_confusion_input_paths.csv \
    --output-dir results \
    --n-init 100 \
    --random-state 0
fi

if [[ "${MODE}" == "all" || "${MODE}" == "plots" ]]; then
  read -r -a FIGURE_ARGS <<< "${FIGURES}"
  "${PYTHON_BIN}" render_paper_figures.py \
    --figures "${FIGURE_ARGS[@]}" \
    --formats pdf png
fi
