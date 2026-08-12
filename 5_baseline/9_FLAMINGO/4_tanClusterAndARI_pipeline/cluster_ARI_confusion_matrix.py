#!/usr/bin/env python3
"""
Tan 聚类 + ARI + 混淆矩阵绘图
==============================
统一管线：PCA(2) 取 PC1 → KMeans(2, n_init=100, random_state=0) → ARI + 混淆矩阵

输入：prepareData/ 下的 10 个 NPZ 文件 (5 方法 × 2 片段)
输出：混淆矩阵热图 PNG + ARI 汇总 CSV

绘图风格与原图 Tan_混淆矩阵图.py 一致：
  - 2 行 (2050 / 160190) × 5 列 (Raw/scHiCluster/HiCImpute/scVI-3D/scHiC-Diff)
  - Blues colormap, vmin=0, vmax=1
  - ARI 在标题, 方法名在 xlabel, "Ground Truth" 在 ylabel
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ============================================================
# 配置
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREPARE_DIR = os.path.join(SCRIPT_DIR, "prepareData")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 方法和片段
DEFAULT_METHODS = ["Raw", "scHiCluster", "HiCImpute", "scVI-3D", "scHiC-Diff"]
METHODS = DEFAULT_METHODS  # 可被 main() --methods 覆盖
SEGMENTS = ["2050", "160190"]
SEGMENT_LABELS = {"2050": "20-50", "160190": "160-190"}

# 细胞标签
LABELS_TRUE = np.array([0] * 14 + [1] * 18, dtype=int)  # GM12878=0, PBMC=1
CELL_LABELS = ["GM12878", "PBMC"]

# 聚类参数 (方案 A 统一管线)
PCA_COMPONENTS = 2
USE_PC = 1  # 只用前 1 个主成分
KMEANS_N_CLUSTERS = 2
KMEANS_N_INIT = 100  # 默认值，可通过 --n-init 覆盖
KMEANS_RANDOM_STATE = 0


# ============================================================
# 聚类 + 指标计算
# ============================================================

def cluster_and_evaluate(data):
    """统一管线：PCA → KMeans → ARI + 混淆矩阵

    参数:
        data: (n_cells, n_features) 矩阵

    返回:
        ari: ARI 分数
        conf_matrix_norm: row-normalized 混淆矩阵 (2×2)
        labels_pred: 预测标签
    """
    # PCA 降维
    pca = PCA(n_components=PCA_COMPONENTS)
    embedding = pca.fit_transform(data)
    use_feature = embedding[:, :USE_PC]  # 只取 PC1

    # KMeans 聚类
    kmeans = KMeans(
        n_clusters=KMEANS_N_CLUSTERS,
        init="k-means++",
        n_init=KMEANS_N_INIT,
        random_state=KMEANS_RANDOM_STATE,
    )
    labels_pred = kmeans.fit_predict(use_feature)

    # ARI
    ari = adjusted_rand_score(LABELS_TRUE, labels_pred)

    # 混淆矩阵 (raw counts)
    cm = confusion_matrix(LABELS_TRUE, labels_pred)

    # Hungarian 算法对齐标签 (解决标签翻转问题)
    row_ind, col_ind = linear_sum_assignment(-cm)

    # 重排列混淆矩阵，使对角线为正确分类
    cm_aligned = cm[:, col_ind]

    # Row-normalize (每行归一化到 0-1)
    row_sums = cm_aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    conf_matrix_norm = cm_aligned.astype(float) / row_sums

    return ari, conf_matrix_norm, labels_pred


# ============================================================
# 绘图
# ============================================================

def plot_confusion_matrices(results, suffix=""):
    """绘制 2×5 混淆矩阵热图

    results: dict[(seg, method)] = (ari, conf_matrix_norm, labels_pred)
    suffix: 输出文件名后缀 (如 "_legacy")
    """
    n_rows = len(SEGMENTS)
    n_cols = len(METHODS)
    figsize_width = n_cols * 4.5 + 0.5
    figsize_height = n_rows * 5.0

    fig = plt.figure(figsize=(figsize_width, figsize_height))

    # 统一 GridSpec: 2 行 × 6 列 (5 方法 + 1 colorbar)
    gs = gridspec.GridSpec(
        nrows=n_rows,
        ncols=n_cols + 1,
        width_ratios=[1] * n_cols + [0.05],
        wspace=0.3,
        hspace=0.5,
    )

    for row_idx, seg in enumerate(SEGMENTS):
        axes = [fig.add_subplot(gs[row_idx, i]) for i in range(n_cols)]
        cbar_ax = fig.add_subplot(gs[row_idx, n_cols])

        for col_idx, method in enumerate(METHODS):
            ax = axes[col_idx]
            key = (seg, method)
            ari, conf_mat, _ = results[key]

            is_last = (col_idx == n_cols - 1)

            sns.heatmap(
                conf_mat,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                vmin=0,
                vmax=1,
                xticklabels=CELL_LABELS,
                yticklabels=CELL_LABELS,
                ax=ax,
                cbar=is_last,
                cbar_ax=cbar_ax if is_last else None,
                annot_kws={"size": 14},
            )

            ax.set_xticklabels(CELL_LABELS, fontsize=12, rotation=0)
            ax.set_yticklabels(CELL_LABELS, fontsize=12, rotation=90)
            ax.set_title(f"ARI: {ari:.3f}", fontsize=16, pad=8)
            ax.set_xlabel(method, fontsize=14, labelpad=8)

            if col_idx == 0:
                ax.set_ylabel("Ground Truth", fontsize=13)

        # 每行标题
        seg_label = SEGMENT_LABELS[seg]
        fig.text(
            0.5,
            1.0 - row_idx * (1.0 / n_rows) - 0.01,
            f"Clustering performance on the {seg_label} Mbp Segment of Chromosome 1",
            fontsize=16,
            fontweight="bold",
            horizontalalignment="center",
            transform=fig.transFigure,
        )

    out_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_Tan{suffix}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n混淆矩阵图已保存: {out_path}")

    return out_path


def save_ari_summary(results, suffix=""):
    """保存 ARI 汇总 CSV"""
    rows = []
    for seg in SEGMENTS:
        for method in METHODS:
            ari, conf_mat, labels_pred = results[(seg, method)]
            rows.append({
                "Segment": SEGMENT_LABELS[seg],
                "Method": method,
                "ARI": round(ari, 4),
                "GM12878_correct_ratio": round(conf_mat[0, 0], 4),
                "PBMC_correct_ratio": round(conf_mat[1, 1], 4),
                "Predicted_labels": str(labels_pred.tolist()),
            })

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, f"ari_summary{suffix}.csv")
    df.to_csv(out_path, index=False)
    print(f"ARI 汇总已保存: {out_path}")
    return df


# ============================================================
# 主函数
# ============================================================

def main():
    global KMEANS_N_INIT, METHODS

    parser = argparse.ArgumentParser(
        description="Tan 聚类 + ARI + 混淆矩阵",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认 5 种方法
  %(prog)s

  # 指定方法 (含新方法)
  %(prog)s --methods Raw,scHiCluster,scHiC-Diff,NewMethod

  # 自动发现 prepareData/ 下所有方法
  %(prog)s --methods all

  # Legacy Raw + 指定 n_init
  %(prog)s --legacy-raw --n-init 10
        """,
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="逗号分隔的方法名 (如 Raw,scHiC-Diff,NewMethod)，或 all 自动发现",
    )
    parser.add_argument(
        "--legacy-raw",
        action="store_true",
        help="Raw 列使用 k=0+log2 预处理 (复刻原 notebook)",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=100,
        help="KMeans n_init 参数 (默认 100, 原脚本用 10)",
    )
    args = parser.parse_args()

    KMEANS_N_INIT = args.n_init

    # 确定方法列表
    if args.methods is None:
        methods = DEFAULT_METHODS
    elif args.methods.lower() == "all":
        # 自动发现 prepareData/ 下所有方法
        import glob
        found = set()
        for seg in SEGMENTS:
            for path in glob.glob(os.path.join(PREPARE_DIR, f"*_{seg}.npz")):
                basename = os.path.basename(path)
                method = basename.replace(f"_{seg}.npz", "")
                if method != "raw_legacy":  # raw_legacy 是 Raw 的变体，不单独列出
                    found.add(method)
        # 排序: raw 在前，已有方法按默认顺序，新方法按字母序
        # 先把 raw 映射为 Raw (文件名是小写 raw，显示名是 Raw)
        found_normalized = {"Raw" if m == "raw" else m for m in found}
        order = {m: i for i, m in enumerate(DEFAULT_METHODS)}
        methods = sorted(found_normalized, key=lambda m: (order.get(m, len(order)), m))
    else:
        methods = [m.strip() for m in args.methods.split(",")]

    METHODS = methods

    raw_mode = "legacy (k=0+log2)" if args.legacy_raw else "unified (k=1, no log)"
    suffix = ""
    if args.legacy_raw:
        suffix += "_legacy"
    if args.n_init != 100:
        suffix += f"_ninit{args.n_init}"
    if args.methods is not None and args.methods.lower() != "all":
        # 自定义方法时，文件名包含方法列表
        method_tag = args.methods.replace(",", "_").replace(" ", "")
        suffix += f"_{method_tag}"
    elif args.methods is not None and args.methods.lower() == "all":
        suffix += "_all"
    raw_file_prefix = "raw_legacy" if args.legacy_raw else "raw"

    print("=" * 60)
    print("Tan 聚类 + ARI + 混淆矩阵 (统一 PCA+KMeans 管线)")
    print(f"  PCA components: {PCA_COMPONENTS}, use PC: {USE_PC}")
    print(f"  KMeans: n_clusters={KMEANS_N_CLUSTERS}, n_init={KMEANS_N_INIT}, seed={KMEANS_RANDOM_STATE}")
    print(f"  Raw 预处理: {raw_mode}")
    print(f"  Methods: {METHODS}")
    print(f"  Segments: {SEGMENTS}")
    print("=" * 60)

    # 检查 prepareData 是否存在
    if not os.path.exists(PREPARE_DIR):
        print(f"\n错误: prepareData 目录不存在: {PREPARE_DIR}")
        print("请先运行 prepareData/prepare_data.py")
        return

    results = {}

    for seg in SEGMENTS:
        print(f"\n--- 片段 {seg} ({SEGMENT_LABELS[seg]} Mbp) ---")
        for method in METHODS:
            # 方法名 → 文件名前缀
            if method == "Raw":
                file_prefix = raw_file_prefix
            else:
                file_prefix = method

            npz_path = os.path.join(PREPARE_DIR, f"{file_prefix}_{seg}.npz")

            if not os.path.exists(npz_path):
                print(f"  跳过 {method} {seg}: 文件不存在 {npz_path}")
                continue

            # 加载稀疏 NPZ → dense
            data = load_npz(npz_path).toarray()
            print(f"  [{method}] {seg}: data shape = {data.shape}")

            # 聚类 + 评估
            ari, conf_mat, labels_pred = cluster_and_evaluate(data)
            results[(seg, method)] = (ari, conf_mat, labels_pred)

            print(f"    ARI = {ari:.4f}")
            print(f"    混淆矩阵 (row-normalized):")
            print(f"      GM12878 → [{conf_mat[0, 0]:.3f}, {conf_mat[0, 1]:.3f}]")
            print(f"      PBMC    → [{conf_mat[1, 0]:.3f}, {conf_mat[1, 1]:.3f}]")

    if not results:
        print("\n错误: 没有找到任何数据文件！")
        return

    # 绘图
    print("\n--- 绘制混淆矩阵 ---")
    plot_confusion_matrices(results, suffix=suffix)

    # 保存 ARI 汇总
    print("\n--- 保存 ARI 汇总 ---")
    df = save_ari_summary(results, suffix=suffix)
    print("\n" + df.to_string(index=False))

    print(f"\n{'='*60}")
    print("完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"  - confusion_matrix_Tan.png")
    print(f"  - ari_summary.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
