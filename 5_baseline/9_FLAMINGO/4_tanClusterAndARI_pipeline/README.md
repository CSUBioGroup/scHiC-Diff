# Tan 聚类 + ARI + 混淆矩阵 实验流程

## 概述

本管线对 Tan 数据集（GM12878 + PBMC，32 细胞，chr1 两个片段）的多种插补方法进行统一聚类评估。
统一管线：**PCA(2) 取 PC1 → KMeans(2, n_init=100, random_state=0) → ARI + row-normalized 混淆矩阵**。

FLAMINGO 插补采用 **PD space + `selection=best`** 方案（与 Lee 数据集一致），基于 `5_lee_SuperTAD_pileline/scripts` 的成功经验。

## 目录结构

```
4_tanClusterAndARI_pipeline/
├── README.md                              ← 本文件
├── cluster_ARI_confusion_matrix.py        ← 聚类 + ARI + 混淆矩阵绘图
├── run_cluster_ari_cpu.sbatch             ← 聚类评估 SLURM 提交
├── prepareData/
│   ├── prepare_data.py                    ← 数据处理：raw + 已有插补结果 → 统一 NPZ
│   ├── generate_universal_input.py        ← 生成通用待插补数据（供新方法使用）
│   ├── raw_{seg}.npz                      ← Raw 统一格式 (32, 465) k=1
│   ├── scHiCluster_{seg}.npz             ← (32, 435)
│   ├── HiCImpute_{seg}.npz               ← (32, 435)
│   ├── scVI-3D_{seg}.npz                 ← (32, 465)
│   ├── scHiC-Diff_{seg}.npz              ← (32, 435)
│   ├── FLAMINGO_PD_{seg}.npz             ← ★ FLAMINGO PD-space 最终结果 (32, 465)
│   └── universal_input/                   ← 通用待插补数据
│       ├── 2050/cells/{cell}.npz          ← 31×31 对称稀疏接触矩阵
│       └── 160190/cells/{cell}.npz
├── flamingo_impute/                       ← FLAMINGO 插补工作区
│   ├── run_tan_pd_lee_array.sbatch        ← ★ PD-space FLAMINGO SLURM (array 0-1)
│   ├── input/                             ← 原始 contact 输入 (from universal_input)
│   │   ├── manifest.tsv
│   │   ├── 2050/contact_matrices/RawCount_Cell_*.txt
│   │   └── 160190/contact_matrices/RawCount_Cell_*.txt
│   ├── input_pd_lee/                      ← PD-space 输入 (prep 产出)
│   │   ├── 2050/distance_matrices/RawCount_Cell_*.txt  (PD 值)
│   │   └── 160190/distance_matrices/RawCount_Cell_*.txt
│   ├── output_pd_lee/                     ← FLAMINGO 完成结果
│   │   ├── 2050/completed_tensor.npy      (32, 31, 31) PD tensor
│   │   ├── 160190/completed_tensor.npy
│   │   └── logs/
│   └── scripts/
│       └── pd_flamingo_rerun.py           ← ★ PD-space prep + post 脚本
└── output/                                ← 聚类评估输出（图和 CSV）
    ├── confusion_matrix_Tan.png
    └── ari_summary.csv
```

## 环境要求

```bash
# conda 环境
/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python

# 依赖
numpy, pandas, scipy, sklearn, matplotlib, seaborn, pyfftw
```

## FLAMINGO 插补数据流（PD space，Lee-style）

```
universal_input/{seg}/cells/{cell}.npz (31×31 contact)
    → prep (pd_flamingo_rerun.py): contact → PD (PD = IF^(-0.25)) → RawCount txt (PD 值)
    → FLAMINGO t-SVD ADMM (mu=1e-4, selection=best, 500 iters, keep-observed)
    → post (pd_flamingo_rerun.py): completed PD → contact (IF = PD^(-4)), clip 异常值, 恢复观测值
    → prepareData/FLAMINGO_PD_{seg}.npz (32, 465, 上三角 k=1, count space)
```

### 关键参数（匹配 Lee 成功配置）

| 参数 | 值 | 说明 |
|------|-----|------|
| `mu` | 1e-4 | threshold=1/mu=10000（早期迭代产生适中 PD） |
| `max_mu` | 1e10 | 最大惩罚 |
| `rho` | 1.1 | 惩罚增长因子 |
| `max_iter` | 500 | 完整 500 次迭代 |
| `selection` | **best** | 选取残差最小的迭代（通常 iter 60-100） |
| `keep_observed` | true | 保留观测值 |
| `clip_factor` | 2.0 | contact > max_observed×2 的异常值置零 |

## 执行流程

### Step 1: 数据准备（已有数据，通常只需运行一次）

```bash
cd 4_tanClusterAndARI_pipeline
python prepareData/prepare_data.py
python prepareData/generate_universal_input.py
```

### Step 2: FLAMINGO PD-space 插补

```bash
cd flamingo_impute
sbatch run_tan_pd_lee_array.sbatch    # array 0-1: 2050, 160190
```

该 sbatch 对每个 segment 执行：
1. `pd_flamingo_rerun.py prep` — contact → PD → RawCount txt
2. `run_flamingo_pyfftw_completion.py` — t-SVD ADMM (mu=1e-4, selection=best)
3. `pd_flamingo_rerun.py post` — PD → contact → clip → `prepareData/FLAMINGO_PD_{seg}.npz`

### Step 3: 聚类 + ARI + 混淆矩阵绘图

```bash
cd ..  # 回到 pipeline 根目录
python cluster_ARI_confusion_matrix.py --methods Raw,scHiCluster,HiCImpute,scVI-3D,scHiC-Diff,FLAMINGO_PD

# 或自动发现 prepareData/ 下所有方法
python cluster_ARI_confusion_matrix.py --methods all
```

也可通过 SLURM 提交：
```bash
sbatch run_cluster_ari_cpu.sbatch
```

**输出**：
- `output/confusion_matrix_Tan.png`：混淆矩阵热图
- `output/ari_summary.csv`：ARI 汇总表

## FLAMINGO PD-space 插补结果

| Segment | Raw CV | FLAMINGO PD CV | 旧 contact-space CV | 插补值范围 |
|---------|--------|---------------|---------------------|-----------|
| 2050 | 0.212 | 0.158 | 0.126 | [0.55, 143.5] |
| 160190 | 0.289 | 0.197 | 0.170 | [0.61, 177.9] |

PD-space 比 contact-space CV 提升 ~25%，观测值完整保留，无 blow-up。

## 统一管线参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 降维 | PCA(n_components=2) | 所有方法统一 |
| 聚类特征 | PC1（第 1 主成分） | 只用 1 个维度 |
| KMeans | n_clusters=2, n_init=100, random_state=0 | init='k-means++' |
| 上三角 | k=1（不含对角线） | 所有方法统一 |
| log 变换 | 无 | 所有方法统一 |
| 标签对齐 | Hungarian 算法 | 解决标签翻转 |
| 混淆矩阵 | row-normalized | 每行归一化到 0-1 |

## 数据来源

- **Raw 数据**：`rawTanData/GSE117874_Count_Bin_Frequency_diag_Segment/` 下的 TSV 文件
- **已有插补结果**：`4_ramani_results/4_ImputationCriteria/results/2_Tan_Dataset/` 下各方法目录
- **细胞列表**：14 GM12878 (01,02,03,05,06,07,09,10,11,12,13,14,15,17) + 18 PBMC (01-18)
- **片段**：chr1 的 20-50 Mbp 和 160-190 Mbp，1Mb 分辨率
- **矩阵大小**：31×31（upper triangle k=1 → 465 features）

## FLAMINGO Method Notes (Cross-Dataset Reference)

本节记录 FLAMINGO t-SVD ADMM 方法本身，供后续 Codex 进程正确应用于新数据集。另见 `1_HiCImputeData/README.md`、`2_FLAMINGOData/README.md`、`3_ramaniData/README.md`、`5_lee_SuperTAD_pileline/scripts/README_FLAMINGO.md`。

### 各数据集最终采用方案

| 数据集 | 输入空间 | selection | 后处理 | 原因 |
|--------|---------|-----------|--------|------|
| `1_HiCImputeData` | PD space | **best** | PD→contact clip | 模拟数据，PD space + best 迭代产生适中 contact |
| `2_FLAMINGOData` | Contact space | **final** | 无 | 模拟数据较密，contact space 稳定收敛 |
| `3_ramaniData` | Contact space | **final** | **阈值 2.0** | 极稀疏数据，FLAMINGO 产生均匀背景，需阈值去除 |
| `4_tanClusterAndARI` (本目录) | **PD space** | **best** | PD→contact clip | 小矩阵，PD space + best 迭代 |
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

3. **Contact space + 极稀疏数据 → 均匀背景**：FLAMINGO 填充 ~1.0 均匀值导致细胞同质化（CV 从 0.96 降到 0.03）。极稀疏数据需用 PD space 或阈值后处理。

4. **PD→contact 转换爆炸**：`contact = PD^(-4)` 在 PD→0 时爆炸。后处理中 PD 低于阈值的条目应设为 **contact=0**，而非 floor 到最小 PD。正确逻辑见 `5_lee_SuperTAD_pileline/scripts/lee_flamingo_pipeline.py`。

5. **禁止在登录节点运行 FLAMINGO**：t-SVD ADMM 是 CPU 密集计算。务必通过 SLURM sbatch 提交到 `cpuQ`。

### 共享 FLAMINGO 运行脚本

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/run_flamingo_pyfftw_completion.py
```

它期望 `RawCount_Cell_XXX.txt`（1-indexed）在 `{input_root}/{dataset}/{input_subdir}/` 下，输出 `completed_tensor.npy` 到 `{output_root}/{dataset}/`。

## Notes For Codex

当复现本工作流时：

1. 工作目录：`4_tanClusterAndARI_pipeline/flamingo_impute`
2. 用 `run_tan_pd_lee_array.sbatch` 跑 PD-space FLAMINGO（array 0-1: segments 2050, 160190）
3. 最终输出：`prepareData/FLAMINGO_PD_{seg}.npz`（由 `pd_flamingo_rerun.py post` 生成）
4. 聚类评估：`cluster_ARI_confusion_matrix.py --methods all`（自动发现 FLAMINGO_PD_*.npz）
5. 不要使用旧的 contact-space FLAMINGO（已删除）
6. 不要使用阈值法（Tan 数据较密，PD-space 直接可用）
7. 所有 FLAMINGO 计算在 cpuQ 节点上运行，禁止在登录节点运行