#!/usr/bin/env python3
"""
通用待插补数据生成脚本
======================
将 raw Tan TSV 数据转为标准化 NPZ + CSV 元数据格式，
供任意新插补方法使用。

输出结构:
  universal_input/
    {seg}/
      cells/{cell_name}.npz     每细胞 n×n 对称稀疏接触矩阵 (COO)
      cell_metadata.csv         细胞元数据 (名称/类型/标签/片段/bin范围)
      segment_info.json         片段信息 (染色体/分辨率/bin坐标)

NPZ 格式说明:
  每个 .npz 存一个 scipy.sparse.coo_matrix (n_bins × n_bins)
  矩阵为对称矩阵，包含对角线
  值为原始接触计数 (整数)
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz, triu

# ============================================================
# 路径配置 (复用 prepare_data.py 的路径)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAN_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..")  # 5_tanData/

RAW_SEGMENT_DIR = os.path.join(
    TAN_DATA_DIR,
    "rawTanData",
    "GSE117874_Count_Bin_Frequency_diag_Segment",
)

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "universal_input")

SEGMENTS = {
    "2050": {"tsv_dir": "chr1_20_50", "sub_start": 20, "sub_end": 50},
    "160190": {"tsv_dir": "chr1_160_190", "sub_start": 160, "sub_end": 190},
}

RESOLUTION = 1_000_000
CHROMOSOME = "chr1"

# 细胞列表
GM12878_IDS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17]
PBMC_IDS = list(range(1, 19))


def get_cell_info():
    """返回 [(cell_name, cell_type, label), ...] 列表"""
    cells = []
    for cid in GM12878_IDS:
        cells.append((f"gm12878_{cid:02d}", "GM12878", 0))
    for cid in PBMC_IDS:
        cells.append((f"pbmc_{cid:02d}", "PBMC", 1))
    return cells


def load_tsv_to_matrix(tsv_path):
    """读取 TSV → 对称接触矩阵 (dense)

    TSV 格式: chr1  pos1  chr2  pos2  count (tab 分隔)
    只加载数值列 (pos1=col1, pos2=col3, count=col4)
    """
    data = np.loadtxt(tsv_path, dtype=int, usecols=(1, 3, 4))
    if data.ndim == 1:
        data = data.reshape(1, -1)

    pos1 = data[:, 0]
    pos2 = data[:, 1]
    counts = data[:, 2]

    n = max(pos1.max(), pos2.max()) + 1
    matrix = np.zeros((n, n), dtype=np.float64)
    for p1, p2, c in zip(pos1, pos2, counts):
        matrix[p1, p2] = c
        matrix[p2, p1] = c

    return matrix


def generate_universal_input(seg_key, seg_info, cell_info):
    """为一个片段生成通用待插补数据"""
    seg_out_dir = os.path.join(OUTPUT_DIR, seg_key)
    cells_dir = os.path.join(seg_out_dir, "cells")
    os.makedirs(cells_dir, exist_ok=True)

    tsv_dir = os.path.join(RAW_SEGMENT_DIR, seg_info["tsv_dir"])
    n_bins = None
    rows = []

    for cell_name, cell_type, label in cell_info:
        tsv_path = os.path.join(tsv_dir, f"{cell_name}.txt")
        matrix = load_tsv_to_matrix(tsv_path)

        if n_bins is None:
            n_bins = matrix.shape[0]

        # 保存为稀疏 NPZ
        sparse = coo_matrix(matrix)
        npz_path = os.path.join(cells_dir, f"{cell_name}.npz")
        save_npz(npz_path, sparse)

        rows.append({
            "cell_name": cell_name,
            "cell_type": cell_type,
            "label": label,
            "segment": seg_key,
            "chromosome": CHROMOSOME,
            "resolution": RESOLUTION,
            "bin_start": seg_info["sub_start"],
            "bin_end": seg_info["sub_end"],
            "n_bins": n_bins,
            "n_nonzero": int((matrix != 0).sum()),
            "total_contacts": int(matrix.sum()),
            "sparsity": round(1 - (matrix != 0).sum() / (n_bins * n_bins), 4),
        })

    # 细胞元数据 CSV
    meta_df = pd.DataFrame(rows)
    meta_path = os.path.join(seg_out_dir, "cell_metadata.csv")
    meta_df.to_csv(meta_path, index=False)

    # 片段信息 JSON
    seg_info_data = {
        "segment": seg_key,
        "chromosome": CHROMOSOME,
        "resolution_bp": RESOLUTION,
        "bin_start": seg_info["sub_start"],
        "bin_end": seg_info["sub_end"],
        "n_bins": n_bins,
        "n_cells": len(cell_info),
        "cell_types": {"GM12878": 14, "PBMC": 18},
        "matrix_format": "symmetric COO sparse, includes diagonal, raw integer counts",
        "npz_key": "data (scipy.sparse.coo_matrix, load with scipy.sparse.load_npz)",
    }
    info_path = os.path.join(seg_out_dir, "segment_info.json")
    with open(info_path, "w") as f:
        json.dump(seg_info_data, f, indent=2)

    # 打印摘要
    print(f"  [{seg_key}] {len(cell_info)} cells, {n_bins}×{n_bins} matrices")
    print(f"    平均稀疏度: {meta_df['sparsity'].mean():.4f}")
    print(f"    平均接触数: {meta_df['total_contacts'].mean():.0f}")
    print(f"    → {cells_dir}/")
    print(f"    → {meta_path}")
    print(f"    → {info_path}")

    return meta_df


def main():
    cell_info = get_cell_info()
    print(f"细胞数: {len(cell_info)} (14 GM12878 + 18 PBMC)")
    print(f"片段: {list(SEGMENTS.keys())}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    all_meta = []
    for seg_key, seg_info in SEGMENTS.items():
        print(f"--- 片段 {seg_key} (bins {seg_info['sub_start']}-{seg_info['sub_end']}) ---")
        meta = generate_universal_input(seg_key, seg_info, cell_info)
        all_meta.append(meta)
        print()

    # 合并元数据
    combined = pd.concat(all_meta, ignore_index=True)
    combined_path = os.path.join(OUTPUT_DIR, "all_cell_metadata.csv")
    combined.to_csv(combined_path, index=False)

    print(f"{'='*60}")
    print("通用待插补数据生成完成！")
    print(f"输出目录: {OUTPUT_DIR}/")
    print(f"  每个片段:")
    print(f"    cells/{{cell}}.npz      — 对称稀疏接触矩阵")
    print(f"    cell_metadata.csv      — 细胞元数据")
    print(f"    segment_info.json      — 片段信息")
    print(f"  all_cell_metadata.csv    — 合并元数据")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
