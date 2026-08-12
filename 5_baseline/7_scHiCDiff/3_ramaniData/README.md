# scHiC-Diff Ramani 插补流程

本目录包含 scHiC-Diff 扩散模型对 Ramani scHi-C 数据的完整插补流程。最终 ARI=0.776。

## 概览

| 项目 | 值 |
|------|-----|
| 模型代码版本 | v1.2（旧版，无 EarlyStopping） |
| 数据集 | Ramani ML1+ML3，626 cells，23 chromosomes，1Mb |
| 训练 epoch | 3000 |
| mask_none_zero | 0.5 |
| batch_size | 1024 |
| 最终 ARI | 0.776（SVD dim=5 + UMAP dim=1） |

## 目录结构

```text
3_ramaniData/
├── README.md                              # 本文件
├── run_ramani_scdiff_v12.sbatch           # 训练提交脚本（GPU，单卡4并行）
├── input/
│   ├── raw_626_chrom_npz/                 # 软链接到共享 626 npz
│   ├── chr*_ramani_scdiff2.h5ad           # 每条染色体的 h5ad 输入
│   └── ramani_scdiff_h5ad_manifest.csv    # 染色体→h5ad 路径映射
├── output/
│   ├── training_results_v12/chr*/         # 各染色体训练结果
│   │   ├── denoise_recon_inv.npz          # 反归一化插补结果（核心输出）
│   │   ├── denoise_recon.npz              # 归一化空间插补结果
│   │   ├── denoise_target.npz             # ground truth（归一化空间）
│   │   ├── raw_x.npz                      # 原始输入
│   │   └── train.log                      # 训练日志
│   ├── chrom_npz/chr*.npz                 # 上三角矩阵（626行，collect 后生成）
│   ├── ramani_embedding.npz              # 626×115 embedding（SVD降维后拼接）
│   ├── ramani_scdiff_validation.json      # 验证信息
│   └── ramani_method_manifest_row.csv     # manifest 片段
├── scripts/
│   ├── prepare_ramani_scdiff_h5ad.py      # 数据预处理：626 npz → h5ad
│   ├── collect_ramani_scdiff_outputs.py   # 结果收集：denoise_recon_inv → chrom_npz + embedding
│   ├── calc_ari_v12_final.py              # ARI 计算（SVD + UMAP pipeline）
│   └── run_calc_ari_v12_final.sbatch      # ARI 计算提交脚本（CPU）
└── logs/                                  # SLURM 日志
```

## 外部依赖

### 模型代码

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/3_DiffusionModel/scHiC-Diff-master/
```

这是 scHiC-Diff v1.2 版本代码，**无 EarlyStopping 回调**，训练会跑到 max_epochs 才停。v5_fast_batched_test 版本有 EarlyStopping（patience=25 硬编码），会导致训练不足、ARI 偏低，**不要使用**。

### Config 文件

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/3_DiffusionModel/scHiC-Diff-master/configs/recon_masked_ramani_3000.yaml
```

关键参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `base_learning_rate` | 2.0e-04 | 学习率（v5_fast 不 scale） |
| `mask_none_zero` | 0.5 | mask 50% 非零元素（1Mb 推荐值 0.3-0.5） |
| `zero_to_none_zero` | 0.1 | mask 零元素比例 |
| `batch_size` | 128（config）→ 1024（sbatch 覆盖） | 626 cells < 1024，每 epoch 1 batch |
| `max_epochs` | 1000（config）→ 3000（sbatch 覆盖） | 充分训练 |
| `monitor` | val/loss_ema | checkpoint 监控指标 |
| `timesteps` | 1000 | 扩散步数 |
| `denoise_t_sample` | 1000 | 采样步数（≤ timesteps） |
| `balance_loss` | false | 不使用平衡 loss |
| `loss_strategy` | recon_masked | 仅重建被 mask 的元素 |
| `parameterization` | x0 | x0 预测模式 |

### Python 环境

```text
/public/home/hpc254701055/micromamba/envs/scdiff2/bin/python
```

包含 torch 1.12.1+cu116, anndata, umap-learn, pytorch-lightning, scanpy 等。

### 共享输入数据

所有方法共享的 626-row 过滤后的上三角矩阵（软链接主副本）：

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/3_ramaniData/input/raw_626_chrom_npz/
```

由 `ramani_imputation_common.py` 的 `filter_630_to_626()` 生成，使用硬编码索引 `[16, 414, 556, 577]` 从 630 行过滤到 626 行，对齐 `ML1_ML3_cell_list.txt`。

## 完整执行流程

### 步骤 1：数据预处理

将共享的 626 npz 转为 scHiC-Diff 需要的 h5ad 格式。

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/3_ramaniData

# 确保 input/raw_626_chrom_npz 软链接存在
ls -l input/raw_626_chrom_npz
# 应指向 /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/3_ramaniData/input/raw_626_chrom_npz

# 生成 h5ad 输入
/public/home/hpc254701055/micromamba/envs/scdiff2/bin/python scripts/prepare_ramani_scdiff_h5ad.py
```

输出：
- `input/chr1_ramani_scdiff2.h5ad` ... `input/chrX_ramani_scdiff2.h5ad`（23 个文件）
- `input/ramani_scdiff_h5ad_manifest.csv`（染色体→h5ad 路径映射）

每个 h5ad 包含：
- `X`：626×N 上三角稀疏矩阵（CSR，float32）
- `obs`：cell_name, cell_type, dataset
- `var`：chrom, feature_index, n_bins

### 步骤 2：提交训练

```bash
sbatch run_ramani_scdiff_v12.sbatch
```

sbatch 配置：
- 分区：gpu2Q，单卡 V100 32GB
- 并行：单卡 4 条染色体并行（bash `&` + `wait`）
- 染色体分组：6 组（4+4+4+4+4+3）
- CPU：20 核，每任务 num_workers=10
- 内存：10G
- max_epochs=3000，batch_size=1024（命令行覆盖 config）
- checkpoint：every_n_epochs=500，save_top_k=-1（全保存）
- 无 EarlyStopping（v1.2 代码无此功能）

训练完成后自动运行 test（`trainer.test`），用 best checkpoint 采样生成 `denoise_recon_inv.npz`。

### 步骤 3：收集结果

```bash
/public/home/hpc254701055/micromamba/envs/scdiff2/bin/python scripts/collect_ramani_scdiff_outputs.py \
  --make-embedding \
  --result-pattern 'output/training_results_v12/{chrom}/denoise_recon_inv.npz' \
  --output-root output
```

输出：
- `output/chrom_npz/chr*.npz`：23 个上三角矩阵（626 行）
- `output/ramani_embedding.npz`：626×115 embedding（per-chrom SVD dim=5 × 23 = 115）
- `output/ramani_scdiff_validation.json`：行数验证
- `output/ramani_method_manifest_row.csv`：manifest 片段

### 步骤 4：计算 ARI

```bash
sbatch scripts/run_calc_ari_v12_final.sbatch
```

ARI 计算 pipeline（与旧参考脚本一致）：
1. 每条染色体：`np.log1p(matrix)` → `TruncatedSVD(n_components=5, random_state=100)`
2. 23 条染色体 embedding `np.hstack` → (626, 115)
3. `umap.UMAP(n_components=dim, random_state=500)` 遍历 dim 1-10
4. `KMeans(n_clusters=4, n_init=10, random_state=0)` 聚类
5. `adjusted_rand_score` 计算 ARI

最佳结果：dim=1, ARI=0.776, NMI=0.783

## 新数据集适配指南

将此流程应用到新数据集时，需要修改以下内容：

### 1. 数据预处理

新数据集需准备为上三角稀疏矩阵格式：
- 每条染色体一个 `.npz`（scipy.sparse CSR）
- 行=细胞，列=上三角特征 `n_bins*(n_bins-1)/2`
- 行顺序对齐 cell_list.txt

如果原始数据是 contact pairs 或 full matrix，需先转换为上三角格式。参考 `ramani_imputation_common.py` 的 `export_full_npz_cells_to_chrom_npz()` 和 `filter_630_to_626()`。

### 2. 修改 prepare 脚本

`prepare_ramani_scdiff_h5ad.py` 中的关键参数：
- `DEFAULT_INPUT_ROOT`：输入 npz 目录
- `DEFAULT_CELL_LIST`：cell list 文件路径
- `chroms`：染色体列表（如 `chr1-chr22+chrX`）
- `obs` 中的 `cell_type` 和 `dataset` 字段需适配新数据集

### 3. 修改 config

`recon_masked_ramani_3000.yaml` 中可能需要调整的参数：
- `mask_none_zero`：1Mb 用 0.3-0.5，更高分辨率用更小值
- `batch_size`：根据细胞数调整（应 ≥ 细胞数，使每 epoch 1 batch）
- `max_epochs`：3000 是验证过的最佳值，不要减少
- `base_learning_rate`：2e-4 适用于 1Mb Ramani 规模数据

### 4. 修改 sbatch

`run_ramani_scdiff_v12.sbatch` 中：
- `MANIFEST`：指向新 manifest csv
- `CHROM_GROUPS`：根据染色体数量调整分组
- `MAX_EPOCHS`：保持 3000
- `batch_size`：命令行覆盖值，≥ 细胞数

### 5. 修改 collect 脚本

`collect_ramani_scdiff_outputs.py` 中：
- `--result-pattern`：指向新训练输出路径
- `--output-root`：输出目录

### 6. 修改 ARI 计算

`calc_ari_v12_final.py` 中：
- `chrom_dir`：指向新 collect 输出
- `labels`：新数据集的 cell type 标签
- `n_clusters`：新数据集的细胞类型数
- `chroms`：新数据集的染色体列表

## 重要注意事项

1. **必须使用 v1.2 代码**，不要用 v5_fast_batched_test（有 EarlyStopping 会导致训练不足）
2. **max_epochs=3000** 是关键，1000 epoch 的 ARI 仅 0.41，3000 epoch 达到 0.78
3. **mask_none_zero=0.5** 是 1Mb 数据的最佳值，0.8 会导致 ARI 降至 0.10
4. **denoise_recon_inv.npz** 是核心输出（反归一化后的插补结果），不是 denoise_recon.npz
5. **ARI 计算时需做 log1p**：denoise_recon_inv 在归一化后但 log 前的空间，SVD 前需 `np.log1p()`
6. **SVD pipeline 优于 UMAP pipeline**：SVD dim=5 + UMAP sweep 的 ARI=0.776，而 per-chrom UMAP12 的 ARI 仅 0.21
7. **共享 626 npz** 是所有方法的统一输入，通过软链接避免重复存储
8. **patience=3000** 在 v5_fast 代码中仍会触发早停（因为 val/loss 降到 0.000 后 1000 epoch 无改善），只有 v1.2 代码无 EarlyStopping 才能真正跑满 3000 epoch

## ARI & UMAP 计算（已完成的成功流程）

scHiC-Diff 产出的 `output/ramani_embedding.npz` 使用与 calc_ari 脚本一致的参数生成（`log1p=True, SVD seed=100`），可直接复现 ARI=0.776。

### 从 embedding.npz 复现 ARI

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import umap

embedding = np.load("output/ramani_embedding.npz", allow_pickle=True)["data"]  # (626, 115)
labels = [c.split("_")[0] for c in cell_list]
labels_true = LabelEncoder().fit_transform(labels)

kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=0)
for cd in range(1, 11):
    reducer = umap.UMAP(n_components=cd, random_state=500)
    emb = reducer.fit_transform(embedding)
    labels_pred = kmeans.fit_predict(emb)
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    print(f"dim={cd}: ARI={ari:.4f}")
# dim=1: ARI=0.776, dim=2: ARI=0.768
```

### 从 embedding.npz 生成 UMAP 图坐标

```python
reducer = umap.UMAP(n_components=2, random_state=500)
umap_coords = reducer.fit_transform(embedding)  # (626, 2)
# umap_coords[:, 0] = UMAP1, umap_coords[:, 1] = UMAP2
```

### 关键参数速查表

| 步骤 | 参数 |
|------|------|
| per-chrom SVD | `TruncatedSVD(n_components=5, random_state=100)` |
| log1p | 必须在 SVD 前做 |
| merge UMAP | `umap.UMAP(n_components=dim, random_state=500)`（遍历 dim 1-10） |
| KMeans | `n_clusters=4, init="k-means++", n_init=10, random_state=0` |
| 标签 | `ML1_ML3_cell_list.txt` 中每个细胞名 `_` 前的 cell type 前缀 |