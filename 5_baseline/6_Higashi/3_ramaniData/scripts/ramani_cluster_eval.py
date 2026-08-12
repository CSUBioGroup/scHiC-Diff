#!/usr/bin/env python3
"""
Ramani 聚类通用评估脚本
======================
适用于任意插补方法的 Ramani 数据集聚类评估。

功能：
  1. 从预计算 embedding (115维) 或 per-chrom NPZ 出发
  2. UMAP sweep dim 1~10，每个维度跑 KMeans 计算 ARI/NMI/Purity/Homo/ErrRate
  3. 保存每个维度下的 UMAP embedding 为 {method}_dim{d}_ramani_embedding.npz
  4. 绘制每个维度的 cluster 散点图（标题为方法正式名称）
  5. 输出汇总结果

用法:
  # 方式1: 从预计算 embedding 出发（ramani_embedding.npz 含 data + cells）
  python ramani_cluster_eval.py \\
      --embedding-path /path/to/ramani_embedding.npz \\
      --method-name "scHiC-Diff" \\
      --output-dir /path/to/output

  # 方式2: 从 per-chrom NPZ 出发（先做 log1p + SVD + hstack）
  python ramani_cluster_eval.py \\
      --chrom-dir /path/to/chrom_npz \\
      --method-name "scVI-3D" \\
      --svd-seed 100 \\
      --output-dir /path/to/output

  # 可选参数:
  --umap-seed 500    # UMAP random_state
  --kmeans-n-init 100  # KMeans n_init
  --kmeans-seed 100    # KMeans random_state
  --max-dim 10   # UMAP sweep 最大维度
  --n-clusters 4  # 聚类数

注意: 细胞类型标签（626个）已硬编码在代码中，无需外部文件。
"""

import argparse
import os
import sys
import time
import warnings
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import homogeneity_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================
# 默认参数
# ============================================================
DEFAULT_UMAP_SEED = 500
DEFAULT_KMEANS_N_INIT = 100
DEFAULT_KMEANS_SEED = 100
DEFAULT_SVD_DIM = 5
DEFAULT_SVD_SEED = 100
DEFAULT_MAX_DIM = 10
DEFAULT_N_CLUSTERS = 4

# 4 种细胞类型 + 固定配色
CELL_TYPE_ORDER = ['HeLa', 'HAP1', 'K562', 'GM12878']
CELL_TYPE_COLORS = {
    'HeLa':   '#BE2700',  # 红
    'HAP1':   '#FF7F00',  # 橙
    'K562':   '#669933',  # 绿
    'GM12878': '#0083BE',  # 蓝
}

# ============================================================
# 硬编码的 626 个细胞类型标签（对应 chrom_npz 行顺序）
# 来源: ML1_ML3_cell_list.txt 经 filter_630_to_626 (indices [16,414,556,577]) 过滤
# 编码: H=HeLa(258), A=HAP1(214), K=K562(110), G=GM12878(44), 总计 626
# ============================================================
_CELL_TYPE_ENCODED = (
    'HHHAAAHAHAHAAHHAAHAAAAHAAAHHAHAAAAHHAAHAAAAHAAHHHHAHHHHAAHHAHHHAHHHAHA'
    'HHAHHHHHHAAHHHAAHAHHAHAAAHHHHAAHAAAHAAHAHHHHAHHHAAAHAHHAAHAHHAAAHHHHAA'
    'AHAHAHAHAAAAAHHAHHHHAHAAHHAAAHHHAAAAAHAAAHAHHAAHAAAHHHHHHHAHHHHHHHAHHH'
    'AHAHHHHHHAHAAHHHAHHHAHHHAAHHAAHHHAAHHHAHHHAHHAHAAHAAHHHAHHHHHAHAHAAHAH'
    'HHHAHAAHAAAHAHHHAHHAAHAAAHHAAAAHAHAHAHHHAHAAAHHAAAAHAAAAHAHHHHAAHAAHAH'
    'HHHAAHHAHAAHHHAHAHHHAHHHHHAHAAHAAHAHAHHHHHAHAAHHAHHAAAAHHHAHAHHHHHAAHH'
    'HHHHAHHAAHHAAAHAHAHHAAHAAHAHHHHAHHHHHHHAAAHAAAAHAAHAKKKKKGKKGKKGKGGKKK'
    'KKKKKKKGKKKGGGGKGGKGKGKKGKKKKGKKKKKKGKKKKKGKKKKKKKGKGKKKGGKKKKKGKGKKGK'
    'KKKKGKKKKKKGKGKKKKKGGKKKKKKKGKKGGKKGKGGKKKKGKKKKKKGGKGKKGGKGKKGKKK'
)
_CELL_TYPE_DECODE = {'H': 'HeLa', 'A': 'HAP1', 'K': 'K562', 'G': 'GM12878'}


def get_hardcoded_cell_types():
    """返回 626 个细胞类型标签列表（对应 chrom_npz 行顺序）。"""
    return [_CELL_TYPE_DECODE[c] for c in _CELL_TYPE_ENCODED]


# ============================================================
# 数据加载
# ============================================================
def load_from_embedding_npz(path):
    """从预计算 embedding npz 加载（需含 'data' 和 'cells' 键）。"""
    data = np.load(path, allow_pickle=True)
    X = np.asarray(data['data'], dtype=np.float32)
    cells = data['cells']
    return X, cells


def load_from_chrom_npz(chrom_dir, svd_seed=DEFAULT_SVD_SEED,
                        svd_dim=DEFAULT_SVD_DIM, log1p=True):
    """从 per-chrom NPZ 出发：log1p → SVD → hstack。"""
    chroms = [f'chr{i}' for i in range(1, 23)] + ['chrX']
    features = []
    for chrom in chroms:
        path = os.path.join(chrom_dir, f'{chrom}.npz')
        x = sparse.load_npz(path).tocsr().toarray().astype(np.float64)
        if log1p:
            x = np.log1p(x)
        svd = TruncatedSVD(n_components=svd_dim, random_state=svd_seed)
        emb = svd.fit_transform(x)
        features.append(emb)
    X = np.hstack(features).astype(np.float32)
    return X


# ============================================================
# 指标计算
# ============================================================
def purity(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    lt = np.reshape(labels_true, (-1, 1))
    lp = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(lp == c)[0]
        lt_tmp = lt[idx, :].reshape(-1)
        count.append(np.bincount(lt_tmp).max())
    return np.sum(count) / lt.shape[0]


def compute_all_metrics(labels_true, labels_pred):
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    pur = purity(labels_true, labels_pred)
    homo = homogeneity_score(labels_true, labels_pred)
    cm = confusion_matrix(labels_true, labels_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    err = 1 - cm[row_ind, col_ind].sum() / len(labels_true)
    return {
        'ARI': ari, 'NMI': nmi, 'Purity': pur,
        'Homo': homo, 'ErrRate': err,
    }


# ============================================================
# 散点图绘制
# ============================================================
def plot_scatter(emb_2d, labels_true, label_mapping, method_name,
                 dim, out_path, dot_size=5, dpi=300):
    """
    绘制 cluster 散点图。
    emb_2d: (n_cells, 2) — 若原始只有 1 维则自动补零列。
    """
    x = np.asarray(emb_2d)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[1] == 1:
        x = np.hstack([x, np.zeros_like(x)])

    fig = plt.figure(figsize=(5, 5), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.grid(True, which='both', color='#d3d3d3', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    for ctype in CELL_TYPE_ORDER:
        if ctype not in label_mapping:
            continue
        typeid = label_mapping[ctype]
        mask = labels_true == typeid
        ax.scatter(x[mask, 0], x[mask, 1],
                   s=dot_size, label=ctype, marker='o',
                   color=CELL_TYPE_COLORS[ctype], alpha=1)

    ax.set_xlabel('UMAP1', fontsize=18)
    ax.set_ylabel('UMAP2', fontsize=18)
    ax.tick_params(axis='both', which='both', bottom=False, top=False,
                   left=False, right=False, labelbottom=False, labelleft=False)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_title(method_name, fontsize=20, pad=10)
    ax.legend(fontsize=14, markerscale=3, loc='center left',
              bbox_to_anchor=(1.01, 0.5), frameon=False)

    fig.subplots_adjust(left=0.12, right=0.68, bottom=0.11, top=0.88)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def resolve_method_dir(output_dir, method):
    """Return the two-level result directory: output/{method}_cluster_results."""
    output_dir = os.path.normpath(output_dir)
    method_dir_name = f'{method}_cluster_results'
    if os.path.basename(output_dir).lower() == method_dir_name.lower():
        return output_dir
    return os.path.join(output_dir, method_dir_name)


# ============================================================
# 主流程
# ============================================================
def run(args):
    # ---- 1. 加载数据 ----
    cells = None
    if args.embedding_path:
        print(f'[加载] 从 embedding npz: {args.embedding_path}')
        X, cells = load_from_embedding_npz(args.embedding_path)
    elif args.chrom_dir:
        print(f'[加载] 从 chrom_npz: {args.chrom_dir} (log1p={not args.no_log1p}, SVD seed={args.svd_seed})')
        X = load_from_chrom_npz(
            args.chrom_dir, svd_seed=args.svd_seed,
            svd_dim=args.svd_dim, log1p=not args.no_log1p,
        )
    else:
        print('[错误] 必须指定 --embedding-path 或 --chrom-dir', file=sys.stderr)
        sys.exit(1)

    print(f'  X shape: {X.shape}')

    # ---- 2. 加载细胞标签 ----
    if cells is None:
        # 从 chrom_npz 出发时，使用硬编码的 626 个细胞类型标签
        cell_types = get_hardcoded_cell_types()
        cells = np.array(cell_types)
        print(f'  [标签] 使用硬编码的 {len(cell_types)} 个细胞类型标签')
    else:
        cell_types = [c.split('_')[0] for c in cells]
    if X.shape[0] != len(cell_types):
        print(
            f'[错误] 样本数不匹配: X 有 {X.shape[0]} 行，但标签有 {len(cell_types)} 个',
            file=sys.stderr,
        )
        sys.exit(1)
    le = LabelEncoder()
    labels_true = le.fit_transform(cell_types)
    label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print(f'  Label mapping: {label_mapping}')
    print(f'  Cell counts: { {t: cell_types.count(t) for t in sorted(set(cell_types))} }')
    print(f'  Total cells: {len(cell_types)}\n')

    # ---- 3. 输出目录 ----
    method = args.method_name
    n_clusters = args.n_clusters
    umap_seed = args.umap_seed
    km_n_init = args.kmeans_n_init
    km_seed = args.kmeans_seed
    max_dim = args.max_dim

    method_dir = resolve_method_dir(args.output_dir, method)
    emb_dir = os.path.join(method_dir, 'embeddings')
    plot_dir = os.path.join(method_dir, 'plots')
    os.makedirs(emb_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # ---- 4. UMAP sweep + KMeans + 指标 ----

    print(f'[参数] UMAP seed={umap_seed}, KMeans n_init={km_n_init} seed={km_seed}, max_dim={max_dim}')
    print(f'[参数] n_clusters={n_clusters}, method="{method}"\n')

    print(f'{"dim":>4} | {"ARI":>8} | {"NMI":>8} | {"Purity":>8} | {"Homo":>8} | {"ErrRate":>8}')
    print('-' * 70)

    ari_list = []
    best_ari = -1
    best_dim = -1
    best_labels = None
    best_emb = None

    for dim in range(1, max_dim + 1):
        # UMAP
        t0 = time.time()
        reducer = umap.UMAP(random_state=umap_seed, n_components=dim)
        emb = reducer.fit_transform(X)
        umap_time = time.time() - t0

        # KMeans
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++',
                        n_init=km_n_init, random_state=km_seed)
        labels_pred = kmeans.fit_predict(emb)

        # 指标
        m = compute_all_metrics(labels_true, labels_pred)
        ari_list.append(m['ARI'])

        print(f'{dim:>4} | {m["ARI"]:>8.4f} | {m["NMI"]:>8.4f} | '
              f'{m["Purity"]:>8.4f} | {m["Homo"]:>8.4f} | {m["ErrRate"]:>8.4f}')

        # 保存 embedding
        emb_file = os.path.join(emb_dir, f'{method}_dim{dim}_ramani_embedding.npz')
        np.savez_compressed(emb_file,
                            data=emb.astype(np.float32),
                            cells=cells,
                            labels_true=labels_true,
                            labels_pred=labels_pred,
                            dim=dim,
                            method=method,
                            umap_seed=umap_seed,
                            kmeans_seed=km_seed)

        # 绘制散点图（用前2维，dim=1时补零）
        plot_file = os.path.join(plot_dir, f'{method}_cluster_dim{dim}.png')
        plot_scatter(emb[:, :2] if dim >= 2 else emb,
                     labels_true, label_mapping, method,
                     dim, plot_file)

        # 记录最佳
        if m['ARI'] > best_ari:
            best_ari = m['ARI']
            best_dim = dim
            best_labels = labels_pred
            best_emb = emb

    # ---- 5. 汇总 ----
    mean_ari = np.mean(ari_list)
    best_metrics = compute_all_metrics(labels_true, best_labels)

    print()
    print(f'[汇总] Best dim: {best_dim}, Best ARI: {best_ari:.4f}, Mean ARI: {mean_ari:.4f}')
    print(f'[Best Metrics] NMI: {best_metrics["NMI"]:.4f}, '
          f'Purity: {best_metrics["Purity"]:.4f}, '
          f'Homo: {best_metrics["Homo"]:.4f}')

    # ---- 6. 保存汇总到文件 ----
    summary_file = os.path.join(method_dir, f'{method}_ari_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f'Method: {method}\n')
        f.write(f'UMAP seed: {umap_seed}, KMeans n_init: {km_n_init}, KMeans seed: {km_seed}\n')
        f.write(f'X shape: {X.shape}\n')
        f.write(f'Label mapping: {label_mapping}\n\n')
        f.write(f'{"dim":>4} | {"ARI":>8} | {"NMI":>8} | {"Purity":>8} | {"Homo":>8} | {"ErrRate":>8}\n')
        f.write('-' * 70 + '\n')
        for dim in range(1, max_dim + 1):
            # 重新计算各维度指标用于写入文件
            emb_loaded = np.load(
                os.path.join(emb_dir, f'{method}_dim{dim}_ramani_embedding.npz'),
                allow_pickle=True)
            m = compute_all_metrics(labels_true, emb_loaded['labels_pred'])
            f.write(f'{dim:>4} | {m["ARI"]:>8.4f} | {m["NMI"]:>8.4f} | '
                    f'{m["Purity"]:>8.4f} | {m["Homo"]:>8.4f} | {m["ErrRate"]:>8.4f}\n')
        f.write(f'\n[汇总] Best dim: {best_dim}, Best ARI: {best_ari:.4f}, Mean ARI: {mean_ari:.4f}\n')
        f.write(f'[Best Metrics] NMI: {best_metrics["NMI"]:.4f}, '
                f'Purity: {best_metrics["Purity"]:.4f}, '
                f'Homo: {best_metrics["Homo"]:.4f}\n')

    print(f'\n[输出] Embeddings: {emb_dir}/')
    print(f'[输出] Plots:      {plot_dir}/')
    print(f'[输出] Summary:    {summary_file}')


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Ramani 聚类通用评估脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_argument_group('输入数据（二选一）')
    src.add_argument('--embedding-path', type=str, default=None,
                     help='预计算 embedding npz 路径（含 data + cells 键）')
    src.add_argument('--chrom-dir', type=str, default=None,
                     help='per-chrom NPZ 目录路径（将自动做 log1p + SVD + hstack，'
                          '细胞类型标签使用硬编码的 626 个 Ramani 标签）')

    out = parser.add_argument_group('输出')
    out.add_argument('--output-dir', type=str, required=True,
                     help='输出根目录')
    out.add_argument('--method-name', type=str, required=True,
                     help='方法正式名称，用于图标题和文件命名（如 scHiC-Diff）')

    param = parser.add_argument_group('参数（可选）')
    param.add_argument('--svd-seed', type=int, default=DEFAULT_SVD_SEED,
                       help=f'SVD random_state (默认 {DEFAULT_SVD_SEED}，仅 --chrom-dir 时使用)')
    param.add_argument('--svd-dim', type=int, default=DEFAULT_SVD_DIM,
                       help=f'SVD n_components (默认 {DEFAULT_SVD_DIM}，仅 --chrom-dir 时使用)')
    param.add_argument('--no-log1p', action='store_true',
                       help='禁用 log1p 变换（仅 --chrom-dir 时有效）')
    param.add_argument('--umap-seed', type=int, default=DEFAULT_UMAP_SEED,
                       help=f'UMAP random_state (默认 {DEFAULT_UMAP_SEED})')
    param.add_argument('--kmeans-n-init', type=int, default=DEFAULT_KMEANS_N_INIT,
                       help=f'KMeans n_init (默认 {DEFAULT_KMEANS_N_INIT})')
    param.add_argument('--kmeans-seed', type=int, default=DEFAULT_KMEANS_SEED,
                       help=f'KMeans random_state (默认 {DEFAULT_KMEANS_SEED})')
    param.add_argument('--max-dim', type=int, default=DEFAULT_MAX_DIM,
                       help=f'UMAP sweep 最大维度 (默认 {DEFAULT_MAX_DIM})')
    param.add_argument('--n-clusters', type=int, default=DEFAULT_N_CLUSTERS,
                       help=f'KMeans n_clusters (默认 {DEFAULT_N_CLUSTERS})')

    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
