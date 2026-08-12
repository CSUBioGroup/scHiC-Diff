# scVI-3D 插补复现与下游使用手册

本文档记录本项目中 scVI-3D 的当前可复现流程。目标是让后续研究人员或 Codex 在更换数据后，能够明确准备什么输入、提交哪个 Slurm 任务、检查哪些输出，以及如何把结果用于统一指标和 contact map 绘图。

## 1. 先看结论

### 1.1 当前基准实际使用的结果

| 数据族 | 当前结果 | shape | 特征语义 |
|---|---|---:|---|
| HiCImputeData | `/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/1_HiCImputeData/output/npz_lower_tri/{data_name}_scVI3D_imputed.npz` | `(100, 1830)` | NumPy `np.tril_indices(61, k=-1)` 行优先 |
| FLAMINGOData | `/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_outputData_earlystop_bs1500/npz_upper_tri/{data_name}_scVI3D_imputed.npz` | `(1500, 124750)` | early-stop bs1500；NumPy `np.triu_indices(500, k=1)` 行优先 |

两个数据族的最终 NPZ 都是“行=细胞、列=三角特征”的 SciPy sparse matrix，但三角顺序不同：HiCImputeData 使用 `tril`，FLAMINGOData 使用 canonical `triu`。不能根据目录名 `npz_lower_tri` 判断 FLAMINGOData 的实际特征顺序。

### 1.2 三类状态

- **当前已验证**：`n_latent=100`、排除主对角线、`band_max=whole`、每个 genomic band 单独训练一个 scVI 模型；HiCImputeData 最终为 `tril`，FLAMINGOData 最终为 `triu`。
- **新数据必须决定**：分辨率、bin 数、要训练的最大 band、是否仍适合 ZINB、细胞类型与细胞顺序。
- **不要直接照搬**：脚本中的 `61 bins/100 cells` 或 `500 bins/1500 cells` 常量，以及不同数据族之间的三角顺序。

## 2. 方法在本项目中的实现

scVI-3D 不是一次对完整 contact map 建模。本项目按基因组距离 `d` 把每个细胞的矩阵拆为 band：

1. band `d` 的第 `k` 列对应接触 `(k, k+d)`。
2. 同一 band 组成 `(n_cells, n_bins-d)` 的 count matrix。
3. 每个 band 建立一个 `scvi.model.SCVI`，当前 `n_latent=100`。
4. 用该 band 的平均 library size 调用 `get_normalized_expression()`。
5. 将所有 band 写回单细胞接触表，再重建完整矩阵；FLAMINGOData 收集 canonical `triu(k=1)`，HiCImputeData 保持其独立的 `tril(k=-1)` 输出约定。

权威实现为：

- 通用训练与收集：`/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/run_scvi3d_flamingo.py`
- HiCImputeData 输入准备：`/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/1_HiCImputeData/scripts/prepare_scvi3d_hicimpute.py`
- FLAMINGOData 输入准备：`/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_scripts/prepare_scvi3d_input.py`

## 3. 环境与 HPC 资源

当前 Python：

```text
/public/home/hpc254701055/micromamba/envs/2_schic-scvi-3d/bin/python
```

当前提交资源：

| 数据族 | 队列 | GPU | CPU | 内存 | 数组 |
|---|---|---:|---:|---:|---:|
| HiCImputeData | `gpu2Q/gpuq` | 1 | 1 | 20G | `0-11` |
| FLAMINGOData | `gpu4Q/gpuq` | 1 | 10 | 80G | `0-6` |

输入准备和输出检查若需要加载全部矩阵，也必须通过 `cpuQ/cpuq` 提交，不能在登录节点执行。

## 4. 输入数据契约

### 4.1 共同契约

每个数据集输入目录必须有：

```text
cell_1.txt ... cell_N.txt
genome.txt
cell_summary.txt
```

单细胞文件为五列、无表头：

```text
chromA  positionA_bp  chromB  positionB_bp  count
```

`cell_summary.txt` 的行顺序必须与 `cell_1.txt ... cell_N.txt` 一致。最终 NPZ 的第 `i` 行保持该顺序；PCC、MAE 和绘图不会根据名称自动重新对齐。

`genome.txt` 的长度必须写为：

```text
(n_bins - 1) * resolution
```

训练脚本用 `chrom_size // resolution + 1` 恢复 bin 数。如果误写 `n_bins * resolution`，会多生成一个 bin，最终特征数也会错误。

### 4.2 HiCImputeData

- 来源 triplet：`0_gtData/0_downsampled_HiCImputeData/{data_name}/cell_*_chr19.txt`
- `chr19`，`resolution=1`，`n_bins=61`，`n_cells=100`。
- 原始文件按 `cell_<number>_chr19.txt` 的数字排序。
- 当前准备脚本固定检查 100 个细胞；迁移到新数据时必须参数化或修改该检查。

### 4.3 FLAMINGOData

- 原始矩阵位于固定 FLAMINGO 模拟数据根目录的 `sim_<stem>/downsampled_contact_data/type_X_cell_Y.txt`。
- `chrFLAMINGO`，`resolution=1,000,000`，`n_bins=500`，`n_cells=1500`。
- 细胞顺序固定为 `type_1 cell_1..500`、`type_2 cell_1..500`、`type_3 cell_1..500`。
- 准备脚本对正 contact 执行 `ceil()` 后写成整数。原因是当前 scVI 使用 ZINB count likelihood；直接输入浮点 IF 不满足该统计模型的输入假设。

必须用 h5ad 的 `obs` 检查该顺序是否与 GT 的 1500 行一致。文件数相同不能证明细胞已对齐。

## 5. 超参数

| 参数 | 当前值 | 注意事项 |
|---|---:|---|
| `n_latent` | 100 | 每个 band 的 latent dimension；小数据上可能过大 |
| `band_max` | `whole` | 训练 `1..n_bins-1` 全部 band，远距离 band 很稀疏且耗时 |
| `include_diag` | false | contact 特征不含主对角线 |
| `max_epochs` | 未显式设置 | 使用当前 scVI 默认 400；若版本变化，默认值也可能变化 |
| likelihood | ZINB | 要求非负整数 count；FLAMINGO 输入因此使用 `ceil()` |
| library size | band 平均深度 | 由每个 band 的 cell total 平均值计算 |

新数据先检查每个 band 的非零比例。若远距离 band 几乎全零，应先做小范围参数试验，而不是直接训练 `whole`。

## 6. HiCImputeData 运行流程

### 6.1 准备输入

当前输入已准备好时可直接跳到训练。新数据需要在 CPU 节点运行准备脚本，例如：

```bash
sbatch -A pi_limin_r -p cpuQ -q cpuq -c 20 --mem=40G -t 02:00:00 \
  --wrap='/public/home/hpc254701055/micromamba/envs/2_schic-scvi-3d/bin/python /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/1_HiCImputeData/scripts/prepare_scvi3d_hicimpute.py --workers 20 --overwrite'
```

### 6.2 训练、插补和收集

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/1_HiCImputeData
sbatch scripts/submit_scvi3d_hicimpute.sbatch
```

脚本对 12 个数据集分别执行：读取 prepared contact list、按 band 训练、写 `impute_work/{data_name}/scVI-3D_norm/`，然后收集到最终 NPZ。已有 `done.flag` 和最终 NPZ 时会跳过。

## 7. FLAMINGOData 运行流程

### 7.1 准备七个条件

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData
sbatch v3_scripts/submit_prepare_all_cpu.sbatch
```

该 CPU 作业顺序准备七个条件，并用多进程写 1500 个 cell 文件。不要在登录节点直接运行 `prepare_scvi3d_input.py`。

### 7.2 训练和收集

```bash
sbatch v3_scripts/submit_scvi3d_flamingo.sbatch
```

数组 `0-6` 每个条件使用一张 GPU。最终文件为：

```text
v3_outputData/npz_lower_tri/{stem}_scVI3D_imputed.npz
```

## 8. 输出验证

只用轻量 shell 先检查数量：

```bash
find 1_HiCImputeData/output/npz_lower_tri -maxdepth 1 -name '*_scVI3D_imputed.npz' | wc -l
find 2_FLAMINGOData/v3_outputData/npz_lower_tri -maxdepth 1 -name '*_scVI3D_imputed.npz' | wc -l
```

预期分别为 12 和 7。随后在 CPU 节点验证：

- shape 分别是 `(100, 1830)` 和 `(1500, 124750)`；
- 无 NaN/Inf、无负值、`nnz > 0`；
- 第 0 行对应 GT/Raw 的同一个细胞；
- 按数据族对应的三角顺序重建后，矩阵应对称且主对角线为 0；FLAMINGOData 必须按 `triu`，HiCImputeData 按 `tril`。

不要只看文件存在。scVI 对空 band、错误 bin 数或细胞列表遗漏仍可能生成结构合法但内容错误的 NPZ。

## 9. PCC、MAE 与 dropout/held-out

统一实现：

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/1_pccAndMae_all/recalc_all_metrics.py
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/recalc_eval_common.py
```

HiCImputeData 对每个 cell 的全部 1830 特征直接计算 PCC 和 MAE；dropout mask 为：

```python
(Raw == 0) & (True != 0)
```

FLAMINGOData 先把 GT 和 prediction 截断为非负并做 `log1p`，再分别计算：

```python
all_mask      = GT > 0
observed_mask = Raw > 0
heldout_mask  = (GT > 0) & ~(Raw > 0)
```

当前 FLAMINGOData scVI-3D 的权威汇总指标为：

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_FLAMINGOData/v3_outputData_earlystop_bs1500/metrics/scVI3D_FLAMINGO_earlystop_bs1500_metrics.csv
```

该文件基于 early-stop bs1500 的 `triu` 结果生成。统一指标表和 7×9 contact map 的 scVI-3D PCC 注释必须与该文件一致。

完整重算应通过 Slurm：先提交 `submit_recalc_metrics_control.sbatch prepare`，再提交 `0-132` 数组 `submit_recalc_metrics.sbatch`，最后依赖数组成功提交 `aggregate`。运行前先核对 `recalc_eval_common.py` 的方法路径与当前 manifest 一致。

## 10. Contact map 绘图

当前 HiCImputeData 单图固定展示 `cell_index=0`。`heatmap_manifest.py` 会把 scVI-3D 的 `tril` 向量重建为 61×61 对称矩阵。普通版和 GR 版分别提交：

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/2_imputedContactHeatmap
sbatch submit_render_all_hicimpute_method_heatmaps.sbatch
sbatch submit_render_all_hicimpute_method_heatmaps_gr.sbatch
```

FLAMINGO 七条件 7×9 图提交：

```bash
sbatch nature-style-plot/submit_flamingo_heatmap_grid.sbatch
```

该图把每个对称 contact map 归一化到总 contact 10,000，并使用 GT 汇总得到的共享色标。解释图片前必须确认第 0 个细胞在 scVI-3D 输入和 h5ad GT 中是同一细胞。

## 11. 新数据适配清单

1. 固定并保存 cell ID 顺序，禁止依赖未经自然排序的文件名。
2. 固定 feature pair 顺序并按数据族记录；当前 HiCImputeData 为 `tril`，FLAMINGOData 为 `triu`。
3. 修改 `n_bins`、`resolution`、染色体名和 `genome.txt` 长度公式。
4. 确认输入是否为非负整数；浮点归一化矩阵不能直接假定适合 ZINB。
5. 先用少量细胞和少量 band 在 Slurm 上试跑。
6. 检查每个 band 的深度、空 band 和训练日志。
7. 全量任务结束后验证 shape、有限值、非负性、细胞顺序和三角顺序。
8. 先更新统一 evaluator/绘图 manifest，再计算指标或绘图；不要硬编码新路径到临时脚本。

## 12. 常见故障

| 症状 | 原因 | 处理 |
|---|---|---|
| 最终多一个 bin | `genome.txt` 写成 `n_bins*resolution` | 改为 `(n_bins-1)*resolution` 后重新准备 |
| scVI 报 count 非整数 | 输入是浮点 IF | 明确统计假设后转整数；当前 FLAMINGO 使用 `ceil()` |
| FLAMINGO contact map 出现竖线或结构错位 | 把实际 `triu` 的 scVI-3D 特征当成 `tril` | 使用 FLAMINGO manifest 的 `feature_order=triu` 重建 |
| 指标异常但文件 shape 正确 | cell 顺序与 GT 不一致 | 用 cell ID/obs 名逐行核对，不要仅比较行数 |
| 远距离 band 训练异常 | band 几乎全零 | 缩小 `band_max` 或单独处理空 band |
| 重跑没有生效 | 最终 NPZ 或 `done.flag` 触发跳过 | 为新实验使用新输出目录；确认后再清理对应标志 |

## 13. 权威文件

- HiCImputeData 提交：`1_HiCImputeData/scripts/submit_scvi3d_hicimpute.sbatch`
- HiCImputeData 准备：`1_HiCImputeData/scripts/prepare_scvi3d_hicimpute.py`
- FLAMINGOData 准备：`2_FLAMINGOData/v3_scripts/prepare_scvi3d_input.py`
- FLAMINGOData 提交：`2_FLAMINGOData/v3_scripts/submit_scvi3d_flamingo.sbatch`
- 通用训练/收集：`2_FLAMINGOData/v3_scripts/run_scvi3d_flamingo.py`
- 当前绘图输入：`paperplots/2_imputedContactHeatmap/hicimpute_heatmap_input_manifest.tsv` 和 `flamingo_heatmap_input_manifest.tsv`
