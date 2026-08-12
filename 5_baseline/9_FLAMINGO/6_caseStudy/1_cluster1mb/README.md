# HiRES 1Mb clustering plot workflow

本目录是 HiRES 1Mb 聚类图实验的可复制模板。设计目标是：把任意插补方法的
1Mb 插补结果整理为统一的 `cell x features` 矩阵，然后生成 20 维 SVD
embedding，最后绘制与已有 raw、scHiCluster、scHiC-Diff 结果一致风格的
stage 分面 UMAP 图。

后续使用时，可以把整个 `1_cluster1mb` 目录复制到新的插补方法结果目录下。
Codex 应先阅读本 README，再阅读该插补方法自己的 README，完成插补结果整理、
20d SVD 降维和最终绘图。

## 文件

- `01_make_20d_svd_embedding.py`
  - 输入：`scipy sparse npz` 格式的 `cell x features` 矩阵，主推荐格式。
  - 兼容输入：`.h5ad`，读取 `adata.X`；普通 dense `.npz`，读取 `arr_0` 或 `X`。
  - 默认输出：`svd_embedding/method/final_svd_decomp.npz`，数组 key 固定为 `arr_0`。
  - 默认参数：`dim=20`，`TruncatedSVD(algorithm="arpack")`，`random_state=42`，
    `norm_sig=True`。

- `02_plot_1mb_stage_umap.py`
  - 输入：上一步保存的 `final_svd_decomp.npz`。
  - 输入标签：`cell_labels.csv`，必须与 embedding 行顺序一致。
  - 默认输出：`fligures/method/method_1Mb_umap_celltype_split_stage_with_scores.png`
    和 `.pdf`，并默认保存坐标 CSV。
  - 绘图逻辑：UMAP 或缺少 `umap-learn` 时回退到 t-SNE；按 `stage` 分面，按
    `celltype` 着色；每个 stage 标题中加入红/蓝谱系 silhouette score。

## 当前实验目录结构

本目录当前已经整理为：

```text
1_cluster1mb/
  README.md
  01_make_20d_svd_embedding.py
  02_plot_1mb_stage_umap.py
  svd_embedding/
    Raw/
      final_svd_decomp.npz
      final_svd_model.lib
      final_svd_metadata.json
    scHiCDiff/
      final_svd_decomp.npz
      final_svd_model.lib
      final_svd_metadata.json
    scHiCluster/
      legacy_50d_total_decomp.npz
      legacy_50d_metadata.json
      legacy_3288_cell_labels.csv
  fligures/
    Raw/
      raw_02_1Mb_umap_celltype_split_stage_with_scores.png
      raw_02_1Mb_umap_celltype_split_stage_with_scores.pdf
      raw_02_1Mb_umap_celltype_split_stage_with_scores.csv
    scHiCDiff/
      schicdiff02_1Mb_umap_celltype_split_stage_with_scores.png
      schicdiff02_1Mb_umap_celltype_split_stage_with_scores.pdf
      schicdiff02_1Mb_umap_celltype_split_stage_with_scores.csv
    scHiCluster/
      schicluster_legacy_1Mb_umap_celltype_split_stage_with_scores.png
      schicluster_legacy_1Mb_umap_celltype_split_stage_with_scores.pdf
      schicluster_legacy_1Mb_umap_celltype_split_stage_with_scores.csv
```

说明：`fligures` 是当前目录中已经存在的文件夹名，本文档按现有拼写保留。
如果后续想统一为 `figures`，需要同时移动目录并更新 README 中的命令。

## 新方法复制后的标准目录结构

推荐复制到新插补方法目录后整理为：

```text
1_cluster1mb/
  README.md
  01_make_20d_svd_embedding.py
  02_plot_1mb_stage_umap.py
  cell_labels.csv
  input/
    cell_by_features.npz
  svd_embedding/
    <method>/
      final_svd_decomp.npz
      final_svd_model.lib
      final_svd_metadata.json
  fligures/
    <method>/
      <method>_1Mb_umap_celltype_split_stage_with_scores.png
      <method>_1Mb_umap_celltype_split_stage_with_scores.pdf
      <method>_1Mb_umap_celltype_split_stage_with_scores.csv
```

`cell_labels.csv` 至少需要这些列：

```text
cell_id,cellname,stage,celltype
```

允许有额外列，例如 `extra`。最重要的约束是：`cell_labels.csv` 的行顺序必须
和 `cell_by_features.npz` 的行顺序完全一致。

## 完整流程

推荐使用当前已验证的 Python 环境：

```bash
export MPLCONFIGDIR=/private/tmp/mplconfig
export NUMBA_CACHE_DIR=/private/tmp/numba_cache
PYTHON=/Users/wuhaoliu/mamba/envs/10_snaphic_env/bin/python
```

该环境已用于验证 `numpy/scipy/sklearn/matplotlib/pandas/anndata/umap-learn` 相关流程。
下面命令中的 `python3` 可以替换为 `$PYTHON`。

### 1. 整理插补结果为 cell x features

新插补方法可能输出每条染色体一个矩阵，也可能输出全基因组矩阵。

如果是每条染色体一个矩阵，推荐先保证每个矩阵都是：

```text
行：细胞，顺序与 cell_labels.csv 一致
列：该染色体的 1Mb bin/contact/features
```

然后按染色体自然顺序横向拼接：

```python
from scipy import sparse

chr_mats = [sparse.load_npz(path) for path in chr_paths]
cell_by_features = sparse.hstack(chr_mats, format="csr")
sparse.save_npz("input/cell_by_features.npz", cell_by_features)
```

已有 scHiC-Diff 的对应参考代码是：

```text
/Volumes/SumSung500/CSU/1_hires_schicdiff/4_schicdiff_imputed_data_processing/code_on_2080ti/1_merge_npz_script.py
```

如果插补方法已经输出全基因组 `cell x features` 矩阵，直接保存成：

```text
input/cell_by_features.npz
```

### 2. 生成 20d SVD embedding

默认用 sparse NPZ：

```bash
python3 01_make_20d_svd_embedding.py \
  --input input/cell_by_features.npz \
  --labels cell_labels.csv \
  --output svd_embedding/<method>/final_svd_decomp.npz \
  --model-output svd_embedding/<method>/final_svd_model.lib \
  --metadata-output svd_embedding/<method>/final_svd_metadata.json \
  --dim 20
```

如果输入是 raw h5ad：

```bash
python3 01_make_20d_svd_embedding.py \
  --input hires_1Mb_allchr_merged.h5ad \
  --labels cell_labels.csv \
  --output svd_embedding/Raw/final_svd_decomp.npz \
  --model-output svd_embedding/Raw/final_svd_model.lib \
  --metadata-output svd_embedding/Raw/final_svd_metadata.json \
  --dim 20
```

输出 NPZ 必须是：

```python
np.load("svd_embedding/<method>/final_svd_decomp.npz")["arr_0"]
```

形状应为：

```text
n_cells x 20
```

### 3. 绘制 1Mb stage 分面图

```bash
python3 02_plot_1mb_stage_umap.py \
  --embedding svd_embedding/<method>/final_svd_decomp.npz \
  --labels cell_labels.csv \
  --out-prefix fligures/<method>/<method>_1Mb_umap_celltype_split_stage_with_scores \
  --n-neighbors 15 \
  --min-dist 0.1 \
  --point-size 6 \
  --dpi 400
```

把 `<method>` 换成插补方法名称，例如：

```text
raw_02
hirespaper02
schicdiff02
newmethod01
```

## 已有结果对应关系

### Raw

参考目录：

```text
/Volumes/SumSung500/CSU/3_hires_rawdata/0_HiRES_plot
```

关键输入和输出：

```text
hires_1Mb_allchr_merged.h5ad
svd_results/1mb/final_svd_decomp.npz
plot/output/20d_1000000/old_raw_02_1Mb_umap_celltype_split_stage_with_scores.png
```

对应代码：

```text
raw_h5ad_svd.py
01_20d_1mb_npz2h5ad_plot.py
```

### scHiCluster

参考目录：

```text
/Volumes/SumSung500/CSU/2_hires_schicluster
```

原始 embedding 生成命令参考：

```text
2_schicluster_svd_embedding/code/run_hicluster_embedding.sh
```

目标 1Mb 参数：

```text
resolution=1000000
dist=1000000
dim=20
norm_sig=True
```

目标图：

```text
3_plot/output/20d_1000000/hirespaper02_1Mb_umap_celltype_split_stage_with_scores.png
```

注意：当前本地 `2_hires_schicluster/2_schicluster_svd_embedding/output/20d_1000000`
目录为空，但 PNG/PDF 结果已存在。另有旧版真实 schicluster 1Mb embedding 可用于
绘图兼容性验证：

```text
/Volumes/SumSung500/CSU/0_HiRES/31_svd_embedding/output/50d_1000000/decomp/total_decomp.npz
```

该旧 embedding 是 3288 行；同目录 `cell_labels.csv` 是 7469 行，不匹配。验证时
需要用下面的 3288 行细胞表与 stage mapping 重新合并标签：

```text
/Volumes/SumSung500/CSU/0_HiRES/31_svd_embedding/code/cell_table.tsv
/Volumes/SumSung500/CSU/0_HiRES/stage_files_mapping.csv
```

### scHiC-Diff

参考目录：

```text
/Volumes/SumSung500/CSU/1_hires_schicdiff
```

关键输入和输出：

```text
4_schicdiff_imputed_data_processing/1Mb/chr*/denoise_recon_inv.npz
4_schicdiff_imputed_data_processing/cell_by_features_no_Y.npz
4_schicdiff_imputed_data_processing/svd_results/1000000/final_svd_decomp.npz
5_plot/output/20d_1000000/schicdiff02_1Mb_umap_celltype_split_stage_with_scores.png
```

对应代码：

```text
4_schicdiff_imputed_data_processing/code_on_2080ti/1_merge_npz_script.py
4_schicdiff_imputed_data_processing/3_svd_embedding.py
5_plot/01_20d_1Mb_npz2h5ad_plot.ipynb
```

## 本次真实数据验证记录

本次验证使用：

```bash
export MPLCONFIGDIR=/private/tmp/mplconfig
export NUMBA_CACHE_DIR=/private/tmp/numba_cache
PYTHON=/Users/wuhaoliu/mamba/envs/10_snaphic_env/bin/python
ROOT=/Users/wuhaoliu/Downloads/02_First_Review/results/7_caseData/1_cluster1mb
```

### Raw：从真实 h5ad 重新生成 SVD embedding

输入：

```text
/Volumes/SumSung500/CSU/3_hires_rawdata/0_HiRES_plot/hires_1Mb_allchr_merged.h5ad
/Volumes/SumSung500/CSU/3_hires_rawdata/0_HiRES_plot/plot/cell_labels.csv
```

命令：

```bash
$PYTHON $ROOT/01_make_20d_svd_embedding.py \
  --input /Volumes/SumSung500/CSU/3_hires_rawdata/0_HiRES_plot/hires_1Mb_allchr_merged.h5ad \
  --labels /Volumes/SumSung500/CSU/3_hires_rawdata/0_HiRES_plot/plot/cell_labels.csv \
  --output $ROOT/svd_embedding/Raw/final_svd_decomp.npz \
  --model-output $ROOT/svd_embedding/Raw/final_svd_model.lib \
  --metadata-output $ROOT/svd_embedding/Raw/final_svd_metadata.json \
  --dim 20
```

验证输出显示：

```text
Loaded matrix: shape=(7469, 186661), format=h5ad:X, dtype=float32
Saved SVD embedding: .../svd_embedding/Raw/final_svd_decomp.npz shape=(7469, 20)
```

### scHiC-Diff：整理已有真实 20d embedding

由于完整 `cell_by_features_no_Y.npz` 约 5GB，本次未重跑全量 SVD，而是将已有真实
1Mb 20d SVD embedding 和 model 复制到本实验目录：

```text
source:
/Volumes/SumSung500/CSU/1_hires_schicdiff/4_schicdiff_imputed_data_processing/svd_results/1000000/final_svd_decomp.npz
/Volumes/SumSung500/CSU/1_hires_schicdiff/4_schicdiff_imputed_data_processing/svd_results/1000000/final_svd_model.lib

target:
$ROOT/svd_embedding/scHiCDiff/final_svd_decomp.npz
$ROOT/svd_embedding/scHiCDiff/final_svd_model.lib
$ROOT/svd_embedding/scHiCDiff/final_svd_metadata.json
```

### scHiCluster：整理旧版真实 1Mb embedding

当前 `2_hires_schicluster/2_schicluster_svd_embedding/output/20d_1000000`
下没有可用 `total_decomp.npz`。本次验证使用旧版真实 1Mb embedding：

```text
/Volumes/SumSung500/CSU/0_HiRES/31_svd_embedding/output/50d_1000000/decomp/total_decomp.npz
```

该 embedding 为 `3288 x 50`，所以不能直接使用旁边 7469 行的 `cell_labels.csv`。
本次用下面两个文件重建了匹配的 `legacy_3288_cell_labels.csv`：

```text
/Volumes/SumSung500/CSU/0_HiRES/31_svd_embedding/code/cell_table.tsv
/Volumes/SumSung500/CSU/0_HiRES/stage_files_mapping.csv
```

整理后的文件：

```text
$ROOT/svd_embedding/scHiCluster/legacy_50d_total_decomp.npz
$ROOT/svd_embedding/scHiCluster/legacy_3288_cell_labels.csv
$ROOT/svd_embedding/scHiCluster/legacy_50d_metadata.json
```

### 三类真实绘图命令

Raw：

```bash
$PYTHON $ROOT/02_plot_1mb_stage_umap.py \
  --embedding $ROOT/svd_embedding/Raw/final_svd_decomp.npz \
  --labels /Volumes/SumSung500/CSU/3_hires_rawdata/0_HiRES_plot/plot/cell_labels.csv \
  --out-prefix $ROOT/fligures/Raw/raw_02_1Mb_umap_celltype_split_stage_with_scores \
  --n-neighbors 15 --min-dist 0.1 --point-size 6 --dpi 400
```

scHiC-Diff：

```bash
$PYTHON $ROOT/02_plot_1mb_stage_umap.py \
  --embedding $ROOT/svd_embedding/scHiCDiff/final_svd_decomp.npz \
  --labels /Volumes/SumSung500/CSU/1_hires_schicdiff/5_plot/cell_labels.csv \
  --out-prefix $ROOT/fligures/scHiCDiff/schicdiff02_1Mb_umap_celltype_split_stage_with_scores \
  --n-neighbors 15 --min-dist 0.1 --point-size 6 --dpi 400
```

scHiCluster legacy：

```bash
$PYTHON $ROOT/02_plot_1mb_stage_umap.py \
  --embedding $ROOT/svd_embedding/scHiCluster/legacy_50d_total_decomp.npz \
  --labels $ROOT/svd_embedding/scHiCluster/legacy_3288_cell_labels.csv \
  --out-prefix $ROOT/fligures/scHiCluster/schicluster_legacy_1Mb_umap_celltype_split_stage_with_scores \
  --n-neighbors 15 --min-dist 0.1 --point-size 6 --dpi 400
```

三类绘图都已生成 PNG/PDF/CSV 到 `fligures/` 对应子目录。

## 后续验证建议

模板代码修改后，优先做这些验证：

1. 用 raw 的真实 `hires_1Mb_allchr_merged.h5ad` 跑
   `01_make_20d_svd_embedding.py`，确认能生成 `arr_0` 格式的 20d embedding。
2. 用 raw 的真实 20d embedding 跑 `02_plot_1mb_stage_umap.py`，确认 PNG/PDF/CSV 生成。
3. 用 scHiC-Diff 的真实 20d embedding 跑 `02_plot_1mb_stage_umap.py`，确认 PNG/PDF/CSV 生成。
4. 如果当前 scHiCluster 的 `20d_1000000/decomp/total_decomp.npz` 存在，用它和匹配的
   `cell_labels.csv` 跑绘图；不要混用 3288 行 embedding 和 7469 行 labels。

## Codex 后续执行规则

当本目录复制到新的插补方法目录后，Codex 应按以下顺序工作：

1. 阅读本 README，理解本实验只负责 1Mb clustering plot。
2. 阅读插补方法自己的 README，确认它如何输出 1Mb 插补矩阵。
3. 把插补结果整理为 `input/cell_by_features.npz`，矩阵方向必须是 `cell x features`。
4. 准备或复制 `cell_labels.csv`，并确认行数和细胞顺序与矩阵一致。
5. 运行 `01_make_20d_svd_embedding.py`。
6. 运行 `02_plot_1mb_stage_umap.py`。
7. 检查最终 embedding 是否在 `svd_embedding/<method>/` 下生成。
8. 检查最终 PNG/PDF/CSV 是否在 `fligures/<method>/` 下生成。
