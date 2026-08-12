# HiCImputeData contact-map 绘图

本目录集中保存 HiCImputeData contact map 的绘图入口、输入信息表、Slurm
提交脚本、绘图结果和运行日志。正文图、补充图和单图均由同一个 Python
入口生成，避免不同脚本采用不同的数据顺序、归一化或 PCC 来源。

以下路径和命令均以 `nature-style-plot/` 为当前工作目录。

## 目录内容

```text
1_HiCImputedData/
├── plot_hicimpute_heatmaps.py
├── submit_main_fig.sbatch
├── submit_supplement_fig.sbatch
├── submit_single_fig.sbatch
├── HiCImputeData_heatmap_matrix_paths.tsv
├── HiCImputeData_PCC_MAE_metrics.tsv
├── figures/
│   ├── main_fig/
│   ├── supplement_fig/
│   └── single_fig/color=icefire/
└── logs/
```

- `plot_hicimpute_heatmaps.py`：唯一 Python 绘图入口，提供 `grid` 和
  `single` 两个子命令。
- `submit_main_fig.sbatch`：生成正文图，即 `T1/T2/T3 x 1k/2k/4k`。
- `submit_supplement_fig.sbatch`：生成补充图，即 `T1/T2/T3 x 7k`。
- `submit_single_fig.sbatch`：按旧版 `icefire` 风格生成独立 contact map；
  默认生成全部 108 个组合，也可以筛选方法、细胞类型和深度。
- `HiCImputeData_heatmap_matrix_paths.tsv`：九类绘图矩阵的路径模板和
  `tril/triu` 特征顺序。
- `HiCImputeData_PCC_MAE_metrics.tsv`：七种插补方法在 12 个数据条件下的
  PCC/MAE 均值和标准差，共 84 条记录。
- `figures/`：正式图片输出目录。
- `logs/`：Slurm 标准输出和错误日志目录。

## 信息文件来源

### 矩阵路径表

`1_HiCImputedData/HiCImputeData_heatmap_matrix_paths.tsv` 是绘图输入索引，
不是矩阵数据本身。绘图不再直接读取分散在各方法根目录中的原始输出，而是
统一读取：

```text
../imputedData/{method}/{data_name}.npz
```

从路径 TSV 所在的 `1_HiCImputedData/` 目录解析时，对应模板为：

```text
../../imputedData/{method}/{data_name}.npz
```

`imputedData/` 中包含 9 类数据，每类覆盖 `T1/T2/T3 x 1k/2k/4k/7k`
共 12 个条件，总计 108 个 NPZ。路径 TSV 每行包含：

```text
method    path_template    feature_order    notes
```

其中的相对路径以 `1_HiCImputedData/` 为解析基准。

**当前 108 个绘图输入全部采用 NumPy row-major `tril(k=-1)` 特征顺序，
不包含 `triu` 输入，因此当前绘图过程不会执行 `triu -> tril` 转换。**
`feature_order` 列统一记录为 `tril`。脚本仍保留 `triu` 兼容分支，仅用于未来
明确标记为 `triu` 的新输入。

统一绘图副本的原始来源为：

| 绘图名称 | 表中方法名 | 数据来源 |
|---|---|---|
| Input | `Raw` | HiCImpute 模拟数据的观测/降采样矩阵 |
| GT | `True` | `0_gtData/1_Gt_HiCImputeData/` 中的 ground truth |
| scHiCluster | `scHiCluster` | `4_scHiCluster/1_HiCImputeData/output/2_lower_tri_npz/` |
| HiCImpute | `HiCImpute` | `3_HiCImpute/1_HiCImputeData/output/npz_lower_tri/` 中已可直接绘图的 `niter5000_burnin1000` 结果 |
| Higashi-0 | `Higashi_nbr0` | `6_Higashi/1_HiCImputeData/output/npz_lower_tri/`，邻居数为 0 |
| Higashi-5 | `Higashi_nbr5` | 同一 Higashi 目录，邻居数为 5 |
| scVI-3D | `scVI-3D` | scVI-3D 方法输出转换为 lower-triangle 顺序后的绘图副本 |
| T-FLAMINGO | `Tensor-FLAMINGO` | `9_FLAMINGO/1_HiCImputeData/output_distance_best/contact_from_pd/npz_lower_tri/`（`selection=best`） |
| scHiC-Diff | `scHiC-Diff` | `7_scHiCDiff/1_HiCImputeData/output/npz_lower_tri/` |

实际绘图路径模板以 TSV 内容为准。更新某个方法的插补结果时，应先更新
`../imputedData/{method}/` 中的统一副本，再修改路径 TSV 中的顺序声明（如
确有变化），不在 Python 脚本中硬编码方法数据路径。

Tensor-FLAMINGO 的旧 `output_distance` 副本已停用。正式统一副本与指标均
来自 `output_distance_best`；旧输出在多数 heldout 条件中为零方差，不能用于
PCC/SCC 或 contact-map 展示。

### PCC/MAE 指标表

`1_HiCImputedData/HiCImputeData_PCC_MAE_metrics.tsv` 来源于：

```text
../../../1_pccAndMae_all/1_HiCImputedData/HiCImputeData_PCC_MAE_SCC_metrics.csv
```

本地文件是该权威评价表的绘图快照，内容保持一致，只将 CSV 逗号分隔转换为
TSV 制表符分隔，并采用更明确的文件名。上游文件名中的 `unified` 仅表示它
汇总了不同方法的结果，不代表额外的指标处理。

绘图脚本从本地 TSV 读取 `pcc_mean` 作为图中标注，不使用经过归一化的展示
矩阵重新计算 PCC。`Raw` 和 `True` 没有 PCC 标注。上游评价结果更新后，应
同步替换本地 TSV，再重绘图片。

## 推荐运行方式

该项目位于 HPC 环境，正式绘图应通过 Slurm CPU 队列执行。先进入：

```bash
cd nature-style-plot
```

### 正文图

```bash
sbatch 1_HiCImputedData/submit_main_fig.sbatch
```

生成 `T1/T2/T3 x 1k/2k/4k` 的正文大图。

### 补充图

```bash
sbatch 1_HiCImputedData/submit_supplement_fig.sbatch
```

生成 `T1/T2/T3 x 7k` 的补充图。

### 全部单图

```bash
sbatch 1_HiCImputedData/submit_single_fig.sbatch
```

默认生成 `9 方法 x 3 细胞类型 x 4 深度 = 108` 张 PNG。

### 指定单图

```bash
sbatch --export=ALL,METHOD=scVI-3D,CTYPE=T1,DEPTH=1k \
  1_HiCImputedData/submit_single_fig.sbatch
```

可以同时指定多个值，使用空格分隔：

```bash
sbatch --export=ALL,METHODS="HiCImpute scVI-3D",CTYPES="T1 T2",DEPTHS="1k 7k" \
  1_HiCImputedData/submit_single_fig.sbatch
```

可选环境变量：

- `METHOD` 或 `METHODS`
- `CTYPE` 或 `CTYPES`
- `DEPTH` 或 `DEPTHS`
- `CELL_INDEX`，默认 `0`
- `FORMATS`，默认单图为 `png`，例如 `FORMATS="pdf png tiff"`

## Python 命令

以下命令应在已分配的 CPU 节点或 Slurm 作业中运行：

```bash
# 正文图
python 1_HiCImputedData/plot_hicimpute_heatmaps.py grid --mode main

# 补充图
python 1_HiCImputedData/plot_hicimpute_heatmaps.py grid --mode supplement

# 指定单图
python 1_HiCImputedData/plot_hicimpute_heatmaps.py single \
  --method scVI-3D --ctype T1 --depth 1k

# 全部单图
python 1_HiCImputedData/plot_hicimpute_heatmaps.py single
```

通用可选参数：

- `--cell-index`：选择零起始细胞编号，默认绘制第一个细胞，即 `0`。
- `--formats`：输出格式，可选 `pdf`、`png`、`tiff`。
- `--matrix-paths`：指定矩阵路径 TSV。
- `--metrics`：指定 PCC/MAE 指标 TSV。
- `--output-dir`：覆盖默认结果目录。

## 绘图数据处理

所有模式共享以下步骤：

1. 从矩阵路径 TSV 定位 `../imputedData/` 中的统一 NPZ。
2. 校验 `feature_order`；当前全部输入均为 `tril(k=-1)`，不发生三角顺序转换。
3. 重建 61 x 61 contact matrix。
4. 将 NaN、正负无穷处理为 0，并将负值截断为 0。
5. 从指标 TSV 查询 PCC 标注。

正文图和补充图使用完整对称矩阵。每个方法、每个数据条件分别归一化到
总接触量 10,000；同一条件中的全部方法共享 GT 矩阵的第 99 百分位色标
上限。默认输出 PDF 和 PNG。

单图保留旧版逻辑：只显示上三角，每张图独立执行线性 min/max 归一化，使用
seaborn `icefire` 配色。Raw 和 True 不标 PCC，只有 scHiC-Diff 显示颜色条。

## 结果位置

正文图：

```text
1_HiCImputedData/figures/main_fig/
  HiCImputeData_heatmap_grid_main_1k_2k_4k.pdf
  HiCImputeData_heatmap_grid_main_1k_2k_4k.png
```

补充图：

```text
1_HiCImputedData/figures/supplement_fig/
  HiCImputeData_heatmap_grid_supplement_7k.pdf
  HiCImputeData_heatmap_grid_supplement_7k.png
```

单图：

```text
1_HiCImputedData/figures/single_fig/color=icefire/
  K562_{T1|T2|T3}_{1k|2k|4k|7k}_{method}.png
```

运行日志：

```text
1_HiCImputedData/logs/
```
