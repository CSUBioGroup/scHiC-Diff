# Ramani and Tan clustering evaluation and figures

本目录是 Ramani ML1+ML3 与 Tan 数据的正式聚类评价和论文绘图流程。流程只读取
相邻输入目录中已完成的 Raw/插补矩阵，不重新运行任何插补方法。

正式流程统一完成以下工作：

1. Raw、scHiCluster、HiCImpute、scVI-3D、Tensor-FLAMINGO 和 scHiC-Diff 使用未
   log1p 的统一重建 SVD64 embedding；Higashi-nbr0/5 使用未 log1p 的统一重建 SVD128；
2. 主图采用方法特异的已选前缀：Raw=50、scHiCluster=5、HiCImpute=2、
   Higashi-nbr0/5=10、scVI-3D=10、Tensor-FLAMINGO=20、scHiC-Diff=20，直接运行
   K-means 计算主 ARI，同时保留 2/5/10/20/50 分量敏感性结果；
3. 从与各方法主 ARI 完全相同的 embedding 前缀生成仅用于展示的二维 UMAP 坐标；
4. 对 Tan 两个 chr1 片段计算 ARI、对齐后的混淆矩阵和细胞聚类指派；
5. 由一个绘图入口生成包含 A/B/C 的主图及四类独立图。

## 目录结构

项目根目录只保留四个数据目录和本正式流程：

```text
inputRamaniData/                       # Ramani Raw 输入
imputedRamaniData/                     # Ramani 各方法插补输出
inputTanData/                          # Tan Raw 输入
imputedTanData/                        # Tan 各方法插补输出
nature-plot/                           # 指标计算、绘图和正式结果
```

`nature-plot/` 内部结构如下：

```text
RamaniData_clustering_input_paths.csv  # 八个输入条件、显示名和相对路径
TanData_confusion_input_paths.csv      # Tan 方法×片段相对路径清单
calculate_ramani_clustering.py         # 输入审计、特征构建、UMAP/KMeans 和指标计算
calculate_tan_confusion.py             # Tan PCA/KMeans、ARI 和混淆矩阵计算
schicluster_office_code_generate_ramani_cell_embeddings.py
                                       # Ramani scHiCluster 官方式两级 SVD embedding
schicluster_office_code_calculate_ramani_ari.py
                                       # 前 2/5/10/20/50 分量 K-means/ARI
schicluster_office_code_calculate_ramani_plot_ari.py
                                       # 绘图专用逐方法来源/前缀 ARI 选择表
schicluster_office_code_ramani_plot_config.py
                                       # ARI 与 UMAP 共用的来源/前缀配置
schicluster_office_code_generate_ramani_umap_coordinates.py
                                       # 逐方法已选 embedding 前缀的展示 UMAP
render_paper_figures.py                # 唯一正式绘图入口，一次生成五类图
plot_ramani_style.py                   # 字体、配色、图例和绘图组件
submit_ramani_clustering.sbatch        # CPU 节点统一运行入口
results/                               # 正式指标表、聚类坐标和运行配置
figures/                               # 正式 PDF/PNG 图
logs/                                  # Slurm 日志
```

历史原型、重复绘图入口和旧 Tan 管线均已删除。`render_paper_figures.py` 是唯一正式
绘图入口，避免同一结果存在多套读取和样式逻辑。

## 正式输入

`RamaniData_clustering_input_paths.csv` 是唯一输入清单。八个条件如下：

| 条件 | 输入目录 | 角色 |
| --- | --- | --- |
| Raw | `../inputRamaniData/raw_626_chrom_npz` | 参考 |
| scHiCluster | `../imputedRamaniData/scHiCluster/chrom_npz` | 正式比较 |
| HiCImpute | `../imputedRamaniData/HiCImpute/chrom_npz_triu` | 正式比较 |
| Higashi-nbr0 | `../imputedRamaniData/Higashi/higashi-nbr0_u200/chrom_npz` | 正式比较 |
| Higashi-nbr5 | `../imputedRamaniData/Higashi/higashi-nbr5_u500/chrom_npz` | 正式比较 |
| scVI-3D | `../imputedRamaniData/scVI-3D/chrom_npz` | 正式比较 |
| Tensor-FLAMINGO | `../imputedRamaniData/Tensor-FLAMINGO/chrom_npz_thresh2.0` | 正式比较 |
| scHiC-Diff | `../imputedRamaniData/scHiC-Diff/chrom_npz` | 正式比较 |

每个输入目录必须包含 `chr1`-`chr22` 和 `chrX` 共 23 个稀疏 NPZ。
每个矩阵有 626 行，行顺序由以下 cell list 确定：

```text
../../../../1_Dataset/2-Ramani-GSE84920-ML1-ML3/
  upper_npz/1000000bp/ML1_ML3_cell_list.txt
```

四类细胞为 HeLa 258、HAP1 214、K562 110 和 GM12878 44。

`TanData_confusion_input_paths.csv` 列出相同八个显示条件在 `2050` 和 `160190`
两个片段上的 16 个输入。Raw 直接读取 `../inputTanData/{segment}.npz`，七种插补结果
读取 `../imputedTanData/`。每个稀疏 NPZ 为 32 行，固定顺序为 14 个 GM12878 和
18 个 PBMC；`.npz-new`、`raw_*.npz` 与 `raw_legacy_*.npz` 不参与正式流程。

## 历史 115D/UMAP benchmark（保留用于复核）

所有条件使用相同的特征构建规则：

```text
23 chromosome sparse matrices
  -> validate rows, feature counts, finite values and nonnegative contacts
  -> dense float64
  -> log1p
  -> per-chromosome TruncatedSVD(n_components=5, random_state=100)
  -> concatenate 23 chromosome embeddings
  -> 626 x 115 float32 feature matrix
```

流程不会静默截断负值。若输入存在负值，指标脚本会报错停止。

该历史 benchmark 的 ARI 口径为：

```text
UMAP(n_components=2, random_state=seed), seed=0,1,2,3,4
  -> KMeans(n_clusters=4, n_init=100, random_state=seed)
  -> ARI mean +/- population SD over five seeds
```

它不再作为当前 Ramani 主图的数据源。固定二维在所有方法运行前统一确定，不使用真实标签为每个方法选择最优维度。同时计算
UMAP 维度 1-10，并保留三种汇总规则：

- `fixed_dim=2`：论文主结果；
- `unsupervised_dim_by_silhouette`：每个 seed 通过 silhouette 无标签选维；
- `max_ARI_over_dimensions`：用真实标签选择最大 ARI，仅用于诊断，
  `reportable=False`。

聚类展示坐标读取同一套 `626 x 115` 特征，但独立运行真正的二维 UMAP：

```text
UMAP(n_components=2, random_state=0, n_neighbors=30, min_dist=0.3)
```

该参数对八个条件完全一致；相应历史结果仍保留在 `results/` 根目录中用于复核。

## Ramani scHiCluster 官方式敏感性流程

`schicluster_office_code_*` 是当前 Ramani 主图采用的数据流程：

```text
23 chromosome triu_k1 sparse matrices
  -> select first off-diagonal contacts at 1 Mb
  -> per-chromosome ARPACK TruncatedSVD_d, norm_sig=True
  -> concatenate chromosome embeddings
  -> final ARPACK TruncatedSVD_d, norm_sig=True
  -> 64D or 128D common cell embedding
  -> first 2/5/10/20/50 components
  -> KMeans(k=4, n_init=200, random_state=None)
  -> ARI
```

这里的输入是各方法插补接触矩阵，输出是统一重建的两级 TruncatedSVD embedding，
不是各方法原生 embedding，也不是历史 `hicluster_gpu()` 返回的 native PCA embedding。
ARI 的分量截取与 K-means 表达式对应 scHiCluster `example/example.py`。官方示例没有设置
K-means 随机种子，因此本流程同样记录 `random_state=None` 并保存每次聚类标签。
主 ARI 中 Raw 和除 Higashi 外的插补方法读取未 log1p 的 SVD64，Higashi-nbr0/5 读取
未 log1p 的 SVD128。主前缀维数为 Raw=50、scHiCluster=5、HiCImpute=2、
Higashi-nbr0/5=10、scVI-3D=10、Tensor-FLAMINGO=20、scHiC-Diff=20。UMAP 固定为
`random_state=0, n_neighbors=30, min_dist=0.3`，只用于二维展示，不参与 K-means
或 ARI。

运行顺序：

```bash
/Users/wuhaoliu/mamba/envs/10_snaphic_env/bin/python \
  schicluster_office_code_generate_ramani_cell_embeddings.py
/Users/wuhaoliu/mamba/envs/10_snaphic_env/bin/python \
  schicluster_office_code_calculate_ramani_ari.py
/Users/wuhaoliu/mamba/envs/10_snaphic_env/bin/python \
  schicluster_office_code_calculate_ramani_plot_ari.py
/Users/wuhaoliu/mamba/envs/10_snaphic_env/bin/python \
  schicluster_office_code_generate_ramani_umap_coordinates.py
```

输出位于 `results/schicluster_office_code_Ramani/`，并由当前绘图入口直接读取。

Tan 的正式评价口径从原型脚本完整迁入 `calculate_tan_confusion.py`：

```text
32 x features sparse NPZ -> dense float64 -> validate -> log1p
  -> PCA(n_components=2)
  -> KMeans on PC1 (n_clusters=2, n_init=100, random_state=0)
  -> ARI + Hungarian cluster alignment
  -> row-normalized 2 x 2 confusion matrix
```

两个片段为 chr1 20–50 Mb 和 chr1 160–190 Mb。旧 Tan 管线已由该正式实现完整取代。

## 绘图规则

`render_paper_figures.py` 读取以下正式中间表：

- `results/schicluster_office_code_Ramani/schicluster_office_code_Ramani_plot_ARI_long.csv`：
  绘图专用40行 ARI 表，包含逐方法来源维数、前缀维数及
  `selected_for_main`；
- `results/schicluster_office_code_Ramani/schicluster_office_code_Ramani_cluster_coordinates.csv`：
  与每种方法主 ARI 完全相同的 embedding 前缀生成的展示 UMAP；
- `TanData_confusion_summary.csv`：Tan 方法×片段 ARI 和 PCA 摘要；
- `TanData_confusion_matrices.csv`：Tan 对齐后的混淆矩阵计数和行归一化比例。

五类图共享同一个方法顺序、字体、颜色、标记和保存函数：

- Raw 为浅灰；
- 六个对比方法为冷灰；
- scHiC-Diff 使用深绯红强调，但方法名和数值均不加粗；
- 只有主图面板字母 A/B/C 使用粗体；
- 聚类散点按真实细胞类型使用低饱和蓝、赭黄、紫色和青绿色，并保留形状冗余编码；
- UMAP1/UMAP2 使用标准分面轴标题；
- Tan 热图使用与 HeLa 蓝色协调的蓝灰顺序色带，八个方法采用完全一致的文字和边框；
- PDF 使用可编辑 TrueType 字体，PNG 默认 600 dpi。

## 运行方法

本目录位于 HPC 登录节点环境。指标计算和绘图都必须提交到 CPU 节点，不能直接在登录
节点运行 Python。

当前正式 CSV 已存在，因此默认只读取 `results/` 重绘全部图：

```bash
sbatch submit_ramani_clustering.sbatch
```

只有输入矩阵或评价参数变化时，才显式重算指标。可选择只重算 CSV，或在重算后立即
生成全部图：

```bash
sbatch --export=ALL,MODE=metrics \
  submit_ramani_clustering.sbatch
sbatch --export=ALL,MODE=all \
  submit_ramani_clustering.sbatch
```

Slurm 入口固定使用 `-p cpuQ --qos=cpuq`。默认 Python 为
`${HOME}/micromamba/envs/2_schic-scvi-3d/bin/python`，可在提交时通过
`PYTHON_BIN` 覆盖。

默认任务申请 1 CPU。重算模式依次生成两级 SVD64/128 embedding、固定前缀 ARI、
逐方法已选前缀展示 UMAP 和 Tan 指标。BLAS、OpenMP 和 Numba 线程均固定为 1，保证
结果稳定并避免嵌套并行。

绘图脚本支持五个选择器：`main`、`ari`、`cluster`、`sweep` 和 `tan`。
正式批处理默认使用 `--figures all --formats pdf png`，一次生成全部结果；直接运行
renderer 时默认同时输出 SVG、PDF 和 PNG。
需要只重绘某一类图时，可通过 Slurm 环境变量传入选择器，例如：

```bash
sbatch --export=ALL,MODE=plots,FIGURES=main \
  submit_ramani_clustering.sbatch
```

## 正式结果

`results/`：

```text
RamaniData_*                       # 历史 115D/UMAP benchmark，仅用于复核
TanData_confusion_summary.csv
TanData_confusion_matrices.csv
TanData_cluster_assignments.csv
TanData_confusion_run_config.json
features/<condition>_features.npz
schicluster_office_code_Ramani/        # 64D/128D embedding、ARI、标签和审计文件
  schicluster_office_code_Ramani_cluster_coordinates.csv
                                      # 逐方法已选前缀的展示 UMAP，5008 行
  schicluster_office_code_Ramani_plot_ARI_long.csv
                                      # 绘图专用 ARI，40 行
```

审计表和运行配置中的路径均相对于本 `nature-plot/` 目录保存，不写入服务器相关的绝对
路径，因此整个目录迁移后仍可追溯输入来源。

预期数据行数（不含表头）：

- scHiCluster-style embedding 审计：`8 x 2 x 23 = 368`；
- 正式展示坐标：`8 x 626 = 5008`；
- 完整 SVD64/128 敏感性 ARI：`8 x 2 x 5 = 80`；
- 绘图专用逐方法 ARI：`8 x 5 = 40`；
- Tan ARI/PCA 摘要：`8 x 2 = 16`；
- Tan 混淆矩阵明细：`8 x 2 x 2 x 2 = 64`；
- Tan 细胞聚类指派：`8 x 2 x 32 = 512`。

ARI 表中的 `is_imputed` 字段将 Raw 标记为 `False`，七种插补方法标记为 `True`。
Raw 可以作为图中参考，但不计入七方法正式排名。

`figures/` 中固定输出五组 PDF/PNG：

```text
RamaniData_main_ARI_cluster.pdf
RamaniData_main_ARI_cluster.png
RamaniData_ARI_bar.pdf
RamaniData_ARI_bar.png
RamaniData_cluster_grid.pdf
RamaniData_cluster_grid.png
RamaniData_ARI_dimension_sweep.pdf
RamaniData_ARI_dimension_sweep.png
TanData_confusion_matrix.pdf
TanData_confusion_matrix.png
```

其中：

- `RamaniData_main_ARI_cluster`：A/B 位于上方，C 横跨下方全宽。Panel A/B 的宽度
  比例约为 1.15:3.85。Panel A 使用
  按逐方法已选 embedding/前缀 ARI 降序排列的横向点图；每种方法读取保存的一次
  `KMeans(k=4, n_init=200)` 结果，不绘制跨 seed 误差线，并使用完整方法名
  `Tensor-FLAMINGO`。
  Panel A 的高度低于 Panel B，细胞类型图例以 2 x 2 形式置于 Panel A 下方；Panel B
  保持放大的 2 x 4 聚类网格；左列子图左侧纵向标注 `UMAP2`，第二行各子图下方居中
  标注 `UMAP1`，不再使用箭头式坐标指示器。
  Panel C 为 Tan 两个 chr1 片段、八个方法的 2 x 8 混淆矩阵，使用蓝灰顺序色带，
  所有方法等权显示。主图画布固定为 174 x 142 mm，600 dpi PNG 固定为
  4110 x 3354 px；
- `RamaniData_ARI_bar`：保留历史文件名，内容为与 Panel A 一致的
  逐方法已选 embedding/前缀 ARI 排序点图；
- `RamaniData_cluster_grid`：与 Panel B 一致的八条件 2 x 4 聚类图，使用相同配色、
  方框、紧凑间距及 UMAP1/UMAP2 标签，并在下方显示带样本数的图例；
- `RamaniData_ARI_dimension_sweep`：Higashi 使用 SVD128、其余方法使用 SVD64，绘制
  各自前 2/5/10/20/50 个分量的 ARI 曲线；Raw 使用灰色虚线、scHiC-Diff 使用深红
  强调，黑色空心圆标记主 ARI 面板实际采用的逐方法维数；
- `TanData_confusion_matrix`：Panel C 的独立 2 x 8 Tan 混淆矩阵版本。

## 论文图注

### 主图：Ramani ARI、UMAP 与 Tan 混淆矩阵

**图 X | 不同单细胞 Hi-C 插补方法对细胞类型结构恢复的影响。** 比较 Raw 数据、
scHiCluster、HiCImpute、Higashi-nbr0、Higashi-nbr5、scVI-3D、Tensor-FLAMINGO 和
scHiC-Diff。**(A)** Ramani ML1+ML3 数据的聚类一致性。各条件均由逐染色体近对角线
特征、逐染色体 TruncatedSVD、跨染色体拼接及第二次 TruncatedSVD 构建未 log1p 的
统一 cell embedding。Higashi-nbr0/5 使用128维 embedding，其他条件使用64维
embedding。主前缀维数为 Raw=50、scHiCluster=5、HiCImpute=2、
Higashi-nbr0/5=10、scVI-3D=10、Tensor-FLAMINGO=20、scHiC-Diff=20，运行
`KMeans(k=4, n_init=200)`，并以真实四类标签
计算 adjusted Rand index（ARI）。UMAP 不参与该指标；Raw 仅作为未插补参考，不纳入
七种插补方法的正式排名。**(B)** 与 Panel A 各方法完全相同 embedding 前缀的二维 UMAP
展示。所有条件统一使用
`n_neighbors=30`、`min_dist=0.3` 和 `random_state=0`，点的颜色和形状表示真实细胞
类型（HeLa、HAP1、K562 和 GM12878），而非 KMeans 预测标签。**(C)** Tan 数据 chr1
20-50 Mb 和 160-190 Mb 两个片段的细胞类型混淆矩阵。各条件经 `log1p`、二维 PCA
和基于第一主成分的两类 KMeans 聚类后，使用 Hungarian 匹配对齐聚类标签；矩阵按真实
细胞类型逐行归一化，格内数值为细胞比例，标题下方给出对应 ARI。每个片段包含
14 个 GM12878 和 18 个 PBMC 细胞。

### 独立图：固定二维 ARI

**图 Sx | Ramani 数据中不同插补方法的逐方法 embedding/前缀聚类一致性。** Raw和
除Higashi外的方法使用未log1p的SVD64，Higashi使用未log1p的SVD128；主前缀维数为
Raw=50、scHiCluster=5、HiCImpute=2、Higashi-nbr0/5=10、scVI-3D=10、
Tensor-FLAMINGO=20、scHiC-Diff=20。数据预处理和
聚类流程与主图 Panel A 相同。圆点表示保存的 `KMeans(k=4, n_init=200)` 结果对应的
ARI，方法按 ARI 降序排列，不绘制跨 seed 误差线。Raw 为未插补参考。

### 独立图：Ramani UMAP

**图 Sx | Raw 数据及七种插补结果在 Ramani 数据中的二维 UMAP 表示。** 八个条件均从
各自用于 ARI 的同一 embedding 前缀获得展示坐标，并使用完全相同的二维
UMAP 参数。颜色和形状
表示真实细胞类型，图例括号内为细胞数；坐标仅用于展示细胞群体结构，不直接参与主图
Panel A 所示 K-means 或 ARI。

### 独立图：ARI 维度敏感性

**图 Sx | Ramani 聚类 ARI 对保留 SVD 分量数的敏感性。** Higashi 曲线来自未log1p
的SVD128,其性能更好，其余条件曲线来自未log1p的SVD64。曲线表示各自 embedding
前2、5、10、20和50个分量直接进行K-means后的ARI；黑色空心圆标记主ARI面板采用的
逐方法维数。该图用于透明展示不同前缀选择及其敏感性。

### 独立图：Tan 混淆矩阵

**图 Sx | Tan 数据两个 chr1 片段中不同插补方法的细胞类型恢复结果。** 每列对应一个
数据条件，两行分别对应 chr1 20-50 Mb 和 160-190 Mb。矩阵经聚类标签对齐后按真实
细胞类型逐行归一化，格内数字表示细胞比例，颜色深浅使用全图统一的 0-1 标尺；每个
矩阵上方标注 ARI。Raw 读取原始输入，其他列读取各方法的插补结果。

## 正文 ARI 结果

当前逐方法 embedding/前缀选择的结果为：

| 方法 | ARI |
| --- | ---: |
| scHiC-Diff | 0.849569 |
| scHiCluster | 0.812955 |
| Higashi-nbr0 | 0.796484 |
| Higashi-nbr5 | 0.700758 |
| scVI-3D | 0.151361 |
| HiCImpute | 0.022204 |
| Tensor-FLAMINGO | 0.001030 |
| Raw | -0.001881 |


