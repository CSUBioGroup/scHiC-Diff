# FLAMINGOData contact-map 绘图

本目录保存 FLAMINGO 参数扫描数据的 7×9 contact-map 正图绘制入口、输入信息
表、Slurm 提交脚本、正式结果和运行日志。FLAMINGOData 不维护逐方法单图入口。

以下路径和命令均以 `nature-style-plot/` 为当前工作目录。

## 目录内容

```text
2_FLAMINGOData/
├── plot_flamingo_heatmaps.py
├── submit_main_fig.sbatch
├── FLAMINGOData_heatmap_matrix_paths.tsv
├── FLAMINGOData_PCC_MAE_metrics.tsv
├── figures/
│   └── main_fig/
└── logs/
```

- `plot_flamingo_heatmaps.py`：唯一 Python 绘图入口，负责路径解析、不同
  特征编码解码、归一化、PCC 查询和 7×9 排版。
- `submit_main_fig.sbatch`：在 `cpuQ/cpuq` 上生成正式 PDF/PNG。
- `FLAMINGOData_heatmap_matrix_paths.tsv`：Input、GT 和七种方法的当前矩阵
  路径模板、存储格式和特征顺序。
- `FLAMINGOData_PCC_MAE_metrics.tsv`：七种方法在七个参数条件下的原始尺度
  PCC/MAE 统计及其 `transform=raw` 元数据，共 49 条记录。
- `figures/main_fig/`：正式结果目录。
- `logs/`：Slurm 标准输出和错误日志。

## 展示条件

图中七行依次为：

```text
W=0.5
W=0.6
W=0.7
W=0.8
W=0.9
P=1%
P=5%
```

九列依次为：

```text
Input, GT, scHiCluster, HiCImpute, Higashi-0, Higashi-5,
scVI-3D, T-FLAMINGO, scHiC-Diff
```

默认绘制每个数据条件中零起始编号为 `0` 的细胞。

## 信息文件来源

### 矩阵路径表

`2_FLAMINGOData/FLAMINGOData_heatmap_matrix_paths.tsv` 每行包含：

```text
method    path_template    feature_order    notes
```

相对路径以 `2_FLAMINGOData/` 为解析基准。所有 FLAMINGO 模拟数据及插补
结果的规范特征语义均为 NumPy row-major `triu(k=1)`。`feature_order` 列同时
记录实际存储和解码方式，因此 H5AD 与 legacy tensor 使用专用标记，但解码后
仍统一回到 canonical `triu(k=1)`：

| 数据 | `feature_order` | 绘图读取方式 |
|---|---|---|
| Input / GT | `h5ad_var_names` | 特征名顺序为 NumPy `triu(k=1)`；从同一 H5AD 的 `counts`/`gt` 层按 `chrFLAMINGO_i_j` 重建 |
| scVI-3D | `triu` | 读取 early-stop bs1500 稀疏 NPZ，按 NumPy row-major `triu(k=1)` 重建 |
| HiCImpute | `triu` | 读取已修复 R 列主序排列的 `npz_triu_corrected` 结果 |
| scHiCluster | `triu` | 虽然历史目录名含 `lower_tri`，实际特征顺序是 NumPy `triu(k=1)` |
| Higashi-0 / Higashi-5 | `triu` | 实际特征顺序是 NumPy `triu(k=1)` |
| scHiC-Diff | `triu` | 与 H5AD 特征顺序一致的 NumPy `triu(k=1)` |
| T-FLAMINGO | `tri_tensor_tril_encoded_triu` | 唯一特殊存储：canonical triu 特征序列编码在 legacy tensor tril 坐标上，读取时解码回 triu |

当前矩阵来源：

| 方法 | 原始数据位置/版本 |
|---|---|
| Input / GT | FLAMINGO v3 `5_paramsweep_datasets` H5AD 数据集，包括 W=0.7 |
| scVI-3D | `1_scVI-3D/2_FLAMINGOData/v3_outputData_earlystop_bs1500/` |
| HiCImpute | `3_HiCImpute/2_FLAMINGOData/v3_outputData/npz_triu_corrected/` |
| scHiCluster | `4_scHiCluster/2_FLAMINGOData/v3_outputData/2_lower_tri_npz/` |
| Higashi-0 / Higashi-5 | `6_Higashi/2_FLAMINGOData/v3_epoch1000_outputData/npz_lower_tri/` |
| T-FLAMINGO | `9_FLAMINGO/2_FLAMINGOData/v3ContactOutput/` |
| scHiC-Diff | 参数扫描 H5AD 目录下的 `training_results_v5fast_batched_test_bs1500_testbs9999/` |

实际路径模板和顺序声明以 TSV 为准。更新某个方法的结果时，应先确认其真实
特征顺序，再更新 TSV；不能根据目录名中的 `lower_tri` 或 `upper_tri` 推断。

### PCC/MAE 指标表

`2_FLAMINGOData/FLAMINGOData_PCC_MAE_metrics.tsv` 来源于：

```text
../../../1_pccAndMae_all/2_FLAMINGOData/FLAMINGOData_PCC_MAE_SCC_metrics.csv
```

本地 TSV 是上游权威 CSV 的制表符子集，保留 PCC/MAE 字段和 `transform`；
绘图脚本要求全部记录为 `raw`，以拒绝旧的 `log1p(max(value, 0))` 指标。
W=0.7 的 GT、Input、scHiC-Diff 与该权威指标均使用 `5_paramsweep_datasets`。

绘图直接读取本地 TSV 的原始尺度 `pcc_all_mean`，不从归一化后的展示矩阵重新
计算 PCC。Input 和 GT 不显示 PCC。

## 推荐运行方式

该项目位于 HPC 环境，正式绘图必须通过 Slurm CPU 节点运行。先进入：

```bash
cd nature-style-plot
```

提交正式 7×9 图：

```bash
sbatch 2_FLAMINGOData/submit_main_fig.sbatch
```

只验证 63 个矩阵路径和 49 条 PCC 记录，不加载矩阵：

```bash
sbatch --export=ALL,VALIDATE_ONLY=1 \
  2_FLAMINGOData/submit_main_fig.sbatch
```

可选环境变量：

- `CELL_INDEX`：零起始细胞编号，默认 `0`。
- `VMAX_PERCENTILE`：共享色标使用的 GT 正值百分位，默认 `99`。
- `VALIDATE_ONLY=1`：仅验证输入。
- `PYTHON_BIN`：覆盖默认的 `2_schic-scvi-3d` Python。

## Python 命令

以下命令应在已分配的 CPU 节点或 Slurm 作业中运行：

```bash
python 2_FLAMINGOData/plot_flamingo_heatmaps.py
```

常用参数：

```text
--cell-index 0
--vmax-percentile 99
--formats pdf png
--matrix-paths 2_FLAMINGOData/FLAMINGOData_heatmap_matrix_paths.tsv
--metrics 2_FLAMINGOData/FLAMINGOData_PCC_MAE_metrics.tsv
--output-dir 2_FLAMINGOData/figures/main_fig
--validate-only
```

## 绘图数据处理

1. 从矩阵路径 TSV 解析 9 类数据 × 7 个条件，共 63 个输入位置。
2. Input/GT 按 H5AD `var_names` 中的 bead pair 重建；其他方法按 TSV 声明的
   `triu` 或 tensor 编码重建。
3. 将矩阵对称化、对角线置 0、非有限值置 0，并将负值截断为 0。
4. 每个方法、每个条件分别归一化到总接触量 10,000。
5. 七个条件共享全部 GT 正值的第 99 百分位作为色标上限。
6. 不执行 `log` 或 `log1p` 显示变换。
7. PCC 从指标 TSV 读取，与展示矩阵归一化相互独立。

## 结果位置

正式结果：

```text
2_FLAMINGOData/figures/main_fig/
  FLAMINGOData_heatmap_grid_main_7x9.pdf
  FLAMINGOData_heatmap_grid_main_7x9.png
```

运行日志：

```text
2_FLAMINGOData/logs/
```
