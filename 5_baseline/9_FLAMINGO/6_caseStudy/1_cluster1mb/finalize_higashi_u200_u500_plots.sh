#!/usr/bin/env bash
# Finish the 1Mb Higashi u200/u500 HDF5-to-plot workflow after collection.
set -euo pipefail

BASE="${BASE:-/home/limin2/2_projects/6_Higashi/6_caseStudy}"
CLUSTER_DIR="${CLUSTER_DIR:-${BASE}/1_cluster1mb}"
PYTHON="${PYTHON:-/cdata/micromamba/envs/lwh_scvi-3d/bin/python3.9}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EXPECTED_CHROMS="${EXPECTED_CHROMS:-20}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${CLUSTER_DIR}/.cache/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${CLUSTER_DIR}/.cache/numba}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

for method in Higashi_nbr0_u200 Higashi_nbr5_u200 Higashi_nbr0_u500 Higashi_nbr5_u500; do
  chrom_dir="${BASE}/output/${method}/chrom_npz"
  while [[ "$(find "${chrom_dir}" -maxdepth 1 -type f -name 'chr*.npz' 2>/dev/null | wc -l)" -lt "${EXPECTED_CHROMS}" ]]; do
    echo "[WAIT] ${method}: chromosome conversion is incomplete"
    sleep "${POLL_SECONDS}"
  done

  input_dir="${CLUSTER_DIR}/input/${method}"
  svd_dir="${CLUSTER_DIR}/svd_embedding/${method}"
  figure_prefix="${CLUSTER_DIR}/fligures/${method}/${method}_1Mb_umap_celltype_split_stage_with_scores"
  "${PYTHON}" "${CLUSTER_DIR}/00_merge_higashi_chrom_npz.py" \
    --chrom-npz-dir "${chrom_dir}" \
    --labels "${CLUSTER_DIR}/cell_labels.csv" \
    --output "${input_dir}/cell_by_features.npz" \
    --metadata-output "${input_dir}/cell_by_features_metadata.json"
  "${PYTHON}" "${CLUSTER_DIR}/01_make_20d_svd_embedding.py" \
    --input "${input_dir}/cell_by_features.npz" \
    --labels "${CLUSTER_DIR}/cell_labels.csv" \
    --output "${svd_dir}/final_svd_decomp.npz" \
    --model-output "${svd_dir}/final_svd_model.lib" \
    --metadata-output "${svd_dir}/final_svd_metadata.json" \
    --dim 20
  "${PYTHON}" "${CLUSTER_DIR}/02_plot_1mb_stage_umap.py" \
    --embedding "${svd_dir}/final_svd_decomp.npz" \
    --labels "${CLUSTER_DIR}/cell_labels.csv" \
    --out-prefix "${figure_prefix}" \
    --n-neighbors 15 \
    --min-dist 0.1 \
    --point-size 6 \
    --dpi 400
  echo "[DONE] ${method}"
done
