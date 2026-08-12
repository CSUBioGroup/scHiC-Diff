#!/usr/bin/env python3
"""
Tan 数据处理脚本
=================
将 raw Tan 数据（TSV）处理为各插补方法可用的 input 格式，
同时将所有数据（Raw + 4种插补结果）统一为 (32, n_features) NPZ 矩阵，
供 cluster_ARI_confusion_matrix.py 使用。

输入：
  - rawTanData/GSE117874_Count_Bin_Frequency_diag_Segment/chr1_20_50/{cell}.txt
  - rawTanData/GSE117874_Count_Bin_Frequency_diag_Segment/chr1_160_190/{cell}.txt
  - 2_Tan_Dataset/ 下的 4 种插补结果

输出（全部存于 prepareData/ 目录）：
  Part 1 — 各插补方法 input 格式：
    input_for_scVI-3D/{seg}/{cell}.npz        每细胞 31×31 稀疏矩阵
    input_for_HiCImpute/{seg}.csv             特征×细胞 CSV
    input_for_scHiC-Diff/{seg}.npz            (32, 465) 组合 NPZ
  Part 2 — 统一聚类格式：
    raw_{seg}.npz                             (32, n_raw)   上三角 k=1
    scHiCluster_{seg}.npz                     (32, 435)
    HiCImpute_{seg}.npz                       (32, 435)
    scVI-3D_{seg}.npz                         (32, 465)
    scHiC-Diff_{seg}.npz                      (32, 435)
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz, coo_matrix, vstack
import cooler

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
TAN_DATA_DIR = os.path.join(PIPELINE_DIR, "..")  # results/5_tanData/

RAW_SEGMENT_DIR = os.path.join(
    TAN_DATA_DIR,
    "rawTanData",
    "GSE117874_Count_Bin_Frequency_diag_Segment",
)

IMPUTE_DIR = os.path.join(
    TAN_DATA_DIR,
    "..",
    "4_ramani_results",
    "4_ImputationCriteria",
    "results",
    "2_Tan_Dataset",
)

CELL_LIST_PATH = os.path.join(IMPUTE_DIR, "cell_list.txt")

OUTPUT_DIR = SCRIPT_DIR  # prepareData/ 自身

# 片段定义
SEGMENTS = {
    "2050": {"tsv_dir": "chr1_20_50", "sub_start": 20, "sub_end": 50},
    "160190": {"tsv_dir": "chr1_160_190", "sub_start": 160, "sub_end": 190},
}

RESOLUTION = 1_000_000

# 细胞列表（顺序固定：14 GM12878 + 18 PBMC = 32）
GM12878_IDS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17]
PBMC_IDS = list(range(1, 19))


def get_cell_names():
    """返回 32 个细胞名列表，顺序与 cell_list.txt 一致"""
    cells = []
    for cid in GM12878_IDS:
        cells.append(f"gm12878_{cid:02d}")
    for cid in PBMC_IDS:
        cells.append(f"pbmc_{cid:02d}")
    return cells


def get_true_labels():
    """返回真实标签：GM12878=0, PBMC=1"""
    return np.array([0] * 14 + [1] * 18)


# ============================================================
# Part 1: Raw TSV → 各插补方法 input 格式
# ============================================================

def load_tsv_to_matrix(tsv_path):
    """读取 TSV → 构建对称接触矩阵 (dense)

    TSV 格式: chr1  pos1  chr2  pos2  count (tab分隔)
    只加载数值列 (pos1=col1, pos2=col3, count=col4)，跳过 chr 字符串列
    pos 是 0-indexed bin ID (1Mb 分辨率)
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
        matrix[p2, p1] = c  # 对称化

    return matrix


def prepare_input_for_scVI3D(seg_key, seg_info, cell_names):
    """Raw TSV → 每细胞 NPZ 矩阵 (scVI-3D input 格式)"""
    out_dir = os.path.join(OUTPUT_DIR, "input_for_scVI-3D", seg_key)
    os.makedirs(out_dir, exist_ok=True)

    tsv_dir = os.path.join(RAW_SEGMENT_DIR, seg_info["tsv_dir"])
    for cell in cell_names:
        tsv_path = os.path.join(tsv_dir, f"{cell}.txt")
        matrix = load_tsv_to_matrix(tsv_path)
        sparse = coo_matrix(matrix)
        save_npz(os.path.join(out_dir, f"{cell}.npz"), sparse)

    print(f"  [scVI-3D input] {seg_key}: {len(cell_names)} cells saved to {out_dir}")


def prepare_input_for_HiCImpute(seg_key, seg_info, cell_names):
    """Raw TSV → CSV (HiCImpute input 格式: 行=特征, 列=细胞)"""
    out_dir = os.path.join(OUTPUT_DIR, "input_for_HiCImpute")
    os.makedirs(out_dir, exist_ok=True)

    tsv_dir = os.path.join(RAW_SEGMENT_DIR, seg_info["tsv_dir"])
    vectors = []
    for cell in cell_names:
        tsv_path = os.path.join(tsv_dir, f"{cell}.txt")
        matrix = load_tsv_to_matrix(tsv_path)
        # 上三角不含对角线
        idx = np.triu_indices(matrix.shape[0], k=1)
        vectors.append(matrix[idx])

    # vstack → (n_cells, n_features) → 转置 → (n_features, n_cells)
    mat = np.vstack(vectors).T
    df = pd.DataFrame(mat, columns=cell_names)
    out_path = os.path.join(out_dir, f"{seg_key}.csv")
    df.to_csv(out_path, index=False)
    print(f"  [HiCImpute input] {seg_key}: shape={mat.shape} saved to {out_path}")


def prepare_input_for_scHiCDiff(seg_key, seg_info, cell_names):
    """Raw TSV → 组合 NPZ (scHiC-Diff input 格式: 32×n_features)"""
    out_dir = os.path.join(OUTPUT_DIR, "input_for_scHiC-Diff")
    os.makedirs(out_dir, exist_ok=True)

    tsv_dir = os.path.join(RAW_SEGMENT_DIR, seg_info["tsv_dir"])
    vectors = []
    for cell in cell_names:
        tsv_path = os.path.join(tsv_dir, f"{cell}.txt")
        matrix = load_tsv_to_matrix(tsv_path)
        idx = np.triu_indices(matrix.shape[0], k=1)
        vectors.append(matrix[idx])

    mat = np.vstack(vectors)  # (32, n_features)
    sparse = coo_matrix(mat)
    out_path = os.path.join(out_dir, f"{seg_key}.npz")
    save_npz(out_path, sparse)
    print(f"  [scHiC-Diff input] {seg_key}: shape={mat.shape} saved to {out_path}")


# ============================================================
# Part 2: 所有数据 → 统一 (32, n_features) NPZ 格式
# ============================================================

def prepare_raw_unified(seg_key, seg_info, cell_names):
    """Raw TSV → (32, n_features) NPZ"""
    tsv_dir = os.path.join(RAW_SEGMENT_DIR, seg_info["tsv_dir"])
    vectors = []
    for cell in cell_names:
        tsv_path = os.path.join(tsv_dir, f"{cell}.txt")
        matrix = load_tsv_to_matrix(tsv_path)
        idx = np.triu_indices(matrix.shape[0], k=1)
        vectors.append(matrix[idx])

    mat = np.vstack(vectors)  # (32, n_features)
    out_path = os.path.join(OUTPUT_DIR, f"raw_{seg_key}.npz")
    save_npz(out_path, coo_matrix(mat))
    print(f"  [Raw unified] {seg_key}: shape={mat.shape} → {out_path}")


def prepare_raw_legacy(seg_key, seg_info, cell_names):
    """Raw TSV → 30×30 → 上三角 k=0 (含对角线) → log2(x+1) → (32, 465) NPZ

    复刻原 Raw_cluster_only_use_one_segment_of_chr1.ipynb 的预处理:
    - 从 31×31 裁剪为 30×30 (匹配原 [20:50] 切片)
    - triu_indices(k=0) 包含对角线
    - np.log2(x+1) 变换
    """
    tsv_dir = os.path.join(RAW_SEGMENT_DIR, seg_info["tsv_dir"])
    vectors = []
    for cell in cell_names:
        tsv_path = os.path.join(tsv_dir, f"{cell}.txt")
        matrix = load_tsv_to_matrix(tsv_path)
        # 裁剪为 30×30 (去掉最后一个 bin，匹配原 [20:50] 切片)
        matrix = matrix[:30, :30]
        # 上三角含对角线 (k=0)
        idx = np.triu_indices(matrix.shape[0], k=0)
        vectors.append(matrix[idx])

    mat = np.vstack(vectors)  # (32, 465)
    # log2 变换
    mat = np.log2(mat + 1)

    out_path = os.path.join(OUTPUT_DIR, f"raw_legacy_{seg_key}.npz")
    save_npz(out_path, coo_matrix(mat))
    print(f"  [Raw legacy] {seg_key}: shape={mat.shape} → {out_path}")


def prepare_scHiCluster_unified(seg_key, seg_info, cell_names):
    """scHiCluster .cool → 提取片段 → 上三角 k=1 → (32, 435) NPZ"""
    cool_dir = os.path.join(
        IMPUTE_DIR, "1_scHiCluster", f"impute_result_{seg_key}", "cool"
    )
    sub_start = seg_info["sub_start"]
    sub_end = seg_info["sub_end"]

    vectors = []
    for cell in cell_names:
        cool_path = os.path.join(cool_dir, f"{cell}.cool")
        c = cooler.Cooler(cool_path)
        full_matrix = np.array(c.matrix(balance=False)[:])
        sub_matrix = full_matrix[sub_start:sub_end, sub_start:sub_end]
        idx = np.triu_indices(sub_matrix.shape[0], k=1)
        vectors.append(sub_matrix[idx])

    mat = np.vstack(vectors)  # (32, 435)
    out_path = os.path.join(OUTPUT_DIR, f"scHiCluster_{seg_key}.npz")
    save_npz(out_path, coo_matrix(mat))
    print(f"  [scHiCluster unified] {seg_key}: shape={mat.shape} → {out_path}")


def prepare_HiCImpute_unified(seg_key, seg_info, cell_names):
    """HiCImpute CSV → 转置 → (32, 435) NPZ"""
    csv_path = os.path.join(
        IMPUTE_DIR, "2_HiCImpute", f"imputed_{seg_key}_nier10000_burnin500.csv"
    )
    df = pd.read_csv(csv_path, index_col=0)
    mat = df.T.values  # (32, n_features)

    out_path = os.path.join(OUTPUT_DIR, f"HiCImpute_{seg_key}.npz")
    save_npz(out_path, coo_matrix(mat))
    print(f"  [HiCImpute unified] {seg_key}: shape={mat.shape} → {out_path}")


def prepare_scVI3D_unified(seg_key, seg_info, cell_names):
    """scVI-3D per-cell NPZ → 上三角 k=1 → (32, n_features) NPZ"""
    npz_dir = os.path.join(
        IMPUTE_DIR, "3_scVI-3D", f"chr1_{seg_key}_imputed_matrix"
    )
    vectors = []
    for cell in cell_names:
        npz_path = os.path.join(npz_dir, f"{cell}.npz")
        matrix = load_npz(npz_path).toarray()
        idx = np.triu_indices(matrix.shape[0], k=1)
        vectors.append(matrix[idx])

    mat = np.vstack(vectors)  # (32, n_features)
    out_path = os.path.join(OUTPUT_DIR, f"scVI-3D_{seg_key}.npz")
    save_npz(out_path, coo_matrix(mat))
    print(f"  [scVI-3D unified] {seg_key}: shape={mat.shape} → {out_path}")


def prepare_scHiCDiff_unified(seg_key, seg_info, cell_names):
    """scHiC-Diff denoise_recon.npz → 已是 (32, 435) → 直接保存"""
    npz_path = os.path.join(
        IMPUTE_DIR, "4_scHiC-Diff", f"chr1-{seg_key}-mask80", "denoise_recon.npz"
    )
    mat = load_npz(npz_path).toarray()  # (32, 435)

    out_path = os.path.join(OUTPUT_DIR, f"scHiC-Diff_{seg_key}.npz")
    save_npz(out_path, coo_matrix(mat))
    print(f"  [scHiC-Diff unified] {seg_key}: shape={mat.shape} → {out_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    cell_names = get_cell_names()
    print(f"细胞列表 ({len(cell_names)} cells): {cell_names[:3]}...{cell_names[-3:]}")
    print(f"真实标签: {get_true_labels()[:14]} (GM12878=0) | {get_true_labels()[14:]} (PBMC=1)")
    print()

    for seg_key, seg_info in SEGMENTS.items():
        print(f"{'='*60}")
        print(f"片段: {seg_key} (chr1 bins {seg_info['sub_start']}-{seg_info['sub_end']})")
        print(f"{'='*60}")

        # Part 1: Raw → 各插补方法 input 格式
        print("\n--- Part 1: Raw 数据 → 插补方法 input 格式 ---")
        prepare_input_for_scVI3D(seg_key, seg_info, cell_names)
        prepare_input_for_HiCImpute(seg_key, seg_info, cell_names)
        prepare_input_for_scHiCDiff(seg_key, seg_info, cell_names)

        # Part 2: 所有数据 → 统一聚类格式
        print("\n--- Part 2: 所有数据 → 统一 (32, n_features) NPZ ---")
        prepare_raw_unified(seg_key, seg_info, cell_names)
        prepare_raw_legacy(seg_key, seg_info, cell_names)
        prepare_scHiCluster_unified(seg_key, seg_info, cell_names)
        prepare_HiCImpute_unified(seg_key, seg_info, cell_names)
        prepare_scVI3D_unified(seg_key, seg_info, cell_names)
        prepare_scHiCDiff_unified(seg_key, seg_info, cell_names)

    print(f"\n{'='*60}")
    print("全部数据处理完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
