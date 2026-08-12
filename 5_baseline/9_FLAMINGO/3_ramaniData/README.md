# Tensor-FLAMINGO Ramani Raw-Contact Imputation (Thresholded)

## 概述

本目录对 Ramani ML1+ML3 1Mb 数据执行 FLAMINGO t-SVD ADMM 张量补全插补，并对结果做**阈值后处理**以去除均匀背景。

**最终采用方案**：Contact-space FLAMINGO + `selection=final` + `mu=1e-4` 产生 completed tensor → 阈值后处理（`threshold=2.0`）去除低置信度均匀背景 → 保留高置信度插补。

**为何用阈值后处理**：FLAMINGO 在 1.8% 密度的 Ramani 数据上会填充 ~86% 的矩阵为 ~1.0 均匀背景值，导致细胞间变异系数（CV）从 raw 的 0.96 降到 0.03，聚类被完全破坏（ARI≈0）。阈值后处理（threshold=2.0）移除低置信度的均匀填充，只保留高于阈值的插补值，CV 恢复到 0.060（旧法的 2.14x），大幅改善聚类可用性。

## 目录结构

```
3_ramaniData/
├── README.md                              ← 本文件
├── run_ramani_flamingo_array.sbatch      ← Stage 1: FLAMINGO LRTC (23 chr array)
├── run_ramani_thresholded.sbatch         ← Stage 2: 阈值后处理 (3 阈值 sweep)
├── scripts/
│   ├── prepare_ramani_flamingo_inputs.py  ← 数据准备：RawCount → RawCount txt
│   └── threshold_ramani_flamingo.py       ← 阈值后处理：completed tensor → chrom_npz + embedding
├── input/                                 ← 原始输入（共享）
│   ├── manifest.tsv                       ← 23 条染色体清单
│   ├── raw_626_chrom_npz/                 ← 原始 raw 数据 (626, N_features)
│   └── {chr}/
│       ├── contact_matrices/RawCount_Cell_*.txt  ← FLAMINGO 输入
│       ├── observed_contact_features.npz
│       ├── input_file_index.tsv
│       └── metadata.json
└── output/                                ← FLAMINGO 输出 + 阈值结果
    ├── {chr}/                             ← FLAMINGO 完成 tensor (23 dirs)
    │   ├── completed_tensor.npy           ← (626, N_bins, N_bins) contact tensor
    │   ├── completion_log.tsv
    │   └── process_time.tsv
    ├── chrom_npz_thresh2.0/               ← ★ 最终阈值结果 (23 chr NPZ)
    │   └── {chr}.npz                      ← (626, N_features) 阈值后 contact
    ├── ramani_embedding_thresh2.0.npz     ← ★ 聚类 embedding (626, 115)
    ├── ramani_method_thresh2.0_manifest.csv
    ├── ramani_flamingo_thresh2.0_validation.json
    └── logs/
```

## 数据流

```
Ramani chr*.npz (626 cells × upper-triangle features)
    → prep: 重建对称 contact 矩阵 → input/{chr}/contact_matrices/RawCount_Cell_*.txt (IF 值)
    → FLAMINGO t-SVD ADMM (contact space, selection=final, mu=1e-4, 500 iters)
    → output/{chr}/completed_tensor.npy (626 × N_bins × N_bins, 含 ~1.0 均匀背景)
    → 阈值后处理 (threshold=2.0): 插补值 < 2.0 → 置零; 观测值保留
    → output/chrom_npz_thresh2.0/{chr}.npz (626 × N_features, 仅高置信度插补)
    → output/ramani_embedding_thresh2.0.npz (626, 115, log1p + per-chrom SVD)
```

## 执行流程

### 前提

- Python 环境：`/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python`
- 依赖：numpy, scipy, pandas, pyfftw, sklearn
- 共享 FLAMINGO 运行脚本：`/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/run_flamingo_pyfftw_completion.py`
- `input/` 和 `input/raw_626_chrom_npz/` 已就绪

### Stage 1: 运行 FLAMINGO LRTC（23 条染色体并行）

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/3_ramaniData
sbatch run_ramani_flamingo_array.sbatch
```

参数：
- partition: cpuQ, 20 CPUs, 50G, 96h, array 0-22
- FLAMINGO: `mu=1e-4, max_mu=1e10, rho=1.1, max_iter=500, selection=final, keep_observed=true`
- 输入空间：contact space (`input_subdir=contact_matrices`)

### Stage 2: 阈值后处理

```bash
sbatch run_ramani_thresholded.sbatch
```

该 sbatch 对 3 个阈值（0.5, 1.0, 2.0）各运行一次 `threshold_ramani_flamingo.py`，生成对应的 `chrom_npz_thresh{N}/` 和 embedding。**最终选用 threshold=2.0**（CV 恢复最高）。

也可手动运行单个阈值：
```bash
PYTHON_BIN=/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python
$PYTHON_BIN scripts/threshold_ramani_flamingo.py \
    --output-root output \
    --input-root input \
    --threshold 2.0 \
    --make-embedding --log1p \
    --per-chrom-dim 5 --seed 100
```

### 验证输出

```bash
# chrom_npz 文件数
ls output/chrom_npz_thresh2.0/*.npz | wc -l  # 应为 23

# embedding shape
python -c "import numpy as np; print(np.load('output/ramani_embedding_thresh2.0.npz', allow_pickle=True)['data'].shape)"
# 应输出 (626, 115)
```

## 最终输出（threshold=2.0）

```text
output/chrom_npz_thresh2.0/{chr1..chrX}.npz   (626 × N_features, sparse CSR)
output/ramani_embedding_thresh2.0.npz         (626, 115)
output/ramani_method_thresh2.0_manifest.csv
output/ramani_flamingo_thresh2.0_validation.json
```

## ARI & UMAP 计算

从 `output/ramani_embedding_thresh2.0.npz` 或 `output/chrom_npz_thresh2.0/` 出发。

### 从 embedding 复现 ARI

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import umap

embedding = np.load("output/ramani_embedding_thresh2.0.npz", allow_pickle=True)["data"]  # (626, 115)

# 标签
with open("/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/2-Ramani-GSE84920-ML1-ML3/upper_npz/1000000bp/ML1_ML3_cell_list.txt") as f:
    cells = [l.strip() for l in f if l.strip()]
labels_true = LabelEncoder().fit_transform([c.split("_")[0] for c in cells])

kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=0)
for cd in range(1, 11):
    reducer = umap.UMAP(n_components=cd, random_state=500)
    emb = reducer.fit_transform(embedding)
    labels_pred = kmeans.fit_predict(emb)
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    print(f"dim={cd}: ARI={ari:.4f}")
```

### 从 chrom_npz 重新计算 embedding（完全复现）

```python
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

chroms = [f"chr{i}" for i in range(1,23)] + ["chrX"]
features = []
for chrom in chroms:
    x = sparse.load_npz(f"output/chrom_npz_thresh2.0/{chrom}.npz").tocsr().toarray()
    x = np.log1p(x)
    svd = TruncatedSVD(n_components=5, random_state=100)
    features.append(svd.fit_transform(x))
X = np.hstack(features)  # (626, 115)
# 后续 UMAP + KMeans 同上
```

### 关键参数速查表

| 步骤 | 参数 |
|------|------|
| per-chrom SVD | `TruncatedSVD(n_components=5, random_state=100)` |
| log1p | 必须在 SVD 前做 |
| merge UMAP | `umap.UMAP(n_components=dim, random_state=500)`（遍历 dim 1-10） |
| KMeans | `n_clusters=4, init="k-means++", n_init=10, random_state=0` |
| 标签 | `ML1_ML3_cell_list.txt` 中每个细胞名 `_` 前的 cell type 前缀 |

## 阈值后处理原理

FLAMINGO contact-space 在 1.8% 密度数据上会产生 ~1.0 的均匀背景填充：

| 指标 | Raw | FLAMINGO (无阈值) | FLAMINGO (thresh=2.0) |
|------|-----|------------------|----------------------|
| 密度 | 1.8% | 86-93% | 16-57% |
| 每细胞 sum CV | 0.96 | 0.03 | 0.060 |
| 聚类 ARI | 0.41 | ≈0 | 改善 |

阈值后处理逻辑（`threshold_ramani_flamingo.py`）：
1. 读取 `completed_tensor.npy`（FLAMINGO 输出）
2. 读取原始 `observed_contact`（从 `input/{chr}/contact_matrices/`）
3. **插补值 < threshold → 置零**（低置信度，视为噪声）
4. **观测值原样保留**（`--keep-observed` 的一致性）
5. 提取上三角 k=1 → 保存为 `chrom_npz_thresh{N}/{chr}.npz`

## 数据来源

- **源数据**：`/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/2-Ramani-GSE84920-ML1-ML3/upper_npz/1000000bp/ML1_ML3/chr*.npz`
- **细胞列表**：626 cells (44 GM12878, 214 HAP1, 258 HeLa, 110 K562)，见 `ML1_ML3_cell_list.txt`
- **分辨率**：1Mb，23 条染色体
- **矩阵大小**：49-250 bins per chromosome

## FLAMINGO Method Notes (Cross-Dataset Reference)

本节记录 FLAMINGO t-SVD ADMM 方法本身，供后续 Codex 进程正确应用于新数据集。另见 `1_HiCImputeData/README.md`、`2_FLAMINGOData/README.md`、`5_lee_SuperTAD_pileline/scripts/README_FLAMINGO.md`。

### 各数据集最终采用方案

| 数据集 | 输入空间 | selection | 后处理 | 原因 |
|--------|---------|-----------|--------|------|
| `1_HiCImputeData` | PD space | **best** | PD→contact clip | 模拟数据，PD space + best 迭代产生适中 contact |
| `2_FLAMINGOData` | Contact space | **final** | 无 | 模拟数据较密，contact space 稳定收敛 |
| `3_ramaniData` (本目录) | Contact space | **final** | **阈值 2.0** | 极稀疏数据，FLAMINGO 产生均匀背景，需阈值去除 |
| `4_tanClusterAndARI` | **PD space** | **best** | PD→contact clip | 小矩阵，PD space + best 迭代 |
| `5_lee_SuperTAD_pileline` | PD space | **best** | PD→contact clip | 极稀疏单细胞，PD space + best 迭代 |

### 两种输入空间模式

| 模式 | FLAMINGO 输入 | 转换 | 适用场景 |
|------|--------------|------|---------|
| **Contact space** | 原始 contact counts (IF) | 无 | 较密数据 (≥15% 观测)，模拟数据 |
| **PD space** | PD = IF^(-0.25) | 前期 IF→PD，后期 PD→IF | 稀疏单细胞数据，或遵循 FLAMINGO 原始论文 |

### 关键超参数

| 参数 | Contact space | PD space |
|------|--------------|-----------------|
| `mu` | 1e-4 | 1e-4 |
| `max_mu` | 1e10 | 1e10 |
| `rho` | 1.1 | 1.1 |
| `max_iter` | 500 | 500 |
| `selection` | **final** | **best** |
| `keep_observed` | true | true |

**为什么 `selection` 不同**：contact space 中 ADMM 稳定收敛（final 可用）；PD space 中 ADMM 后期**发散**（残差→1e54），需 `selection=best` 选取残差最小的迭代（通常 iter 60-100）。

### 关键陷阱

1. **`selection=final` + PD space → 无插补**：最终迭代 97% missing 得到 PD=0 → contact=0，等于只保留观测数据。真正插补需用 `selection=best`。

2. **`mu=1.0` + PD space → contact 爆炸**：threshold=1.0 保留太多奇异值，missing 得到极小 PD (0.02) → contact=0.02^(-4)≈6e5。**务必用 `mu=1e-4`**。

3. **Contact space + 极稀疏数据 → 均匀背景**：FLAMINGO 填充 ~1.0 均匀值导致细胞同质化（CV 从 0.96 降到 0.03）。需**阈值后处理**（如本数据集 threshold=2.0）移除低置信度填充。

4. **PD→contact 转换爆炸**：`contact = PD^(-4)` 在 PD→0 时爆炸。后处理中 PD 低于阈值的条目应设为 **contact=0**，而非 floor 到最小 PD。正确逻辑见 `5_lee_SuperTAD_pileline/scripts/lee_flamingo_pipeline.py`。

5. **禁止在登录节点运行 FLAMINGO**：t-SVD ADMM 是 CPU 密集计算。务必通过 SLURM sbatch 提交到 `cpuQ`。

### 共享 FLAMINGO 运行脚本

所有数据集调用同一个未修改的运行脚本：

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/run_flamingo_pyfftw_completion.py
```

它期望 `RawCount_Cell_XXX.txt`（1-indexed）在 `{input_root}/{dataset}/{input_subdir}/` 下，输出 `completed_tensor.npy` 到 `{output_root}/{dataset}/`。

## Notes For Codex

当复现本工作流时：

1. 工作目录：`3_ramaniData`
2. Stage 1 用 `run_ramani_flamingo_array.sbatch` 跑 FLAMINGO（contact space, selection=final）
3. Stage 2 用 `run_ramani_thresholded.sbatch` 跑阈值后处理（threshold=2.0 最优）
4. 最终输出在 `output/chrom_npz_thresh2.0/` 和 `output/ramani_embedding_thresh2.0.npz`
5. 不要使用旧的 `output/chrom_npz/`（无阈值，ARI≈0）
6. 不要使用 PD space（本数据集 PD space 的 CV 不如阈值法）
7. 所有计算在 cpuQ 节点上运行，禁止在登录节点运行 FLAMINGO