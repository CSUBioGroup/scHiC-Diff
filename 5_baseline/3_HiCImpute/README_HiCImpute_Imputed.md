# HiCImpute 插补复现与下游使用手册

本文档说明本项目中 HiCImpute 的输入二进制格式、R `upper.tri` 特征顺序、MCMC 参数、输出恢复方式，以及如何把结果接入统一指标和 contact map。HiCImpute 最容易出错的地方不是模型调用，而是 Python 与 R 的三角特征顺序不一致。

## 1. 当前基准结果

### 1.1 HiCImputeData

当前普通版、GR 版和统一绘图 manifest 读取：

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/1_HiCImputeData/output/npz_lower_tri/{data_name}_niter5000_burnin1000.npz
```

- shape：`(100, 1830)`。
- 当前文件已是绘图可用的 NumPy `tril(k=-1)` 顺序。
- **不需要再次转换或重排。**
- 不要用同目录中名称相似的旧文件覆盖这批已确认结果。

### 1.2 FLAMINGOData

当前统一指标和绘图读取：

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_outputData/npz_triu_corrected/{data_name}_hicimpute_Impute_All_triu.npz
```

- shape：`(1500, 124750)`。
- 特征顺序：NumPy `np.triu_indices(500, k=1)` 行优先。
- 该目录由原始 `_Impute_All.bin` 使用正确的逆置换恢复，不是历史 `npz_lower_tri` 的直接复制。

### 1.3 禁止混用的输出

- `2_FLAMINGOData/v3_outputData/npz_lower_tri/*Impute_All_lower_tri.npz`：历史命名不能证明顺序正确；当前不用。
- FLAMINGOData 的 `Impute_SZ`：本次审计中曾出现约 986 byte 的空 NPZ，不能作为结果。
- HiCImputeData 的 `*_hicimpute_Impute_All_lower_tri.npz`：若由当前 v3 R 脚本新生成，必须检查其真实三角顺序，不能仅凭文件名替换已确认的 `*_niter5000_burnin1000.npz`。

## 2. 方法和数据布局

HiCImpute 在 R 中接收：

```text
scHiC:    n_features × n_cells
bulk:     n_features
expected: 可选的 n_features × n_cells
```

本项目用小端 `float64` 二进制传递大矩阵。`schic.bin` 必须按 Fortran/列优先布局写出，R 才能用：

```r
matrix(readBin(...), nrow=n_features, ncol=n_cells)
```

正确的数据流为：

```text
NumPy row-major triu feature matrix
  -> 重排到 R upper.tri column-major
  -> 转为 features × cells
  -> little-endian float64, Fortran order binary
  -> MCMCImpute
  -> 恢复到 NumPy row-major triu
  -> 如 HiCImputeData 下游要求 tril，再做有坐标语义的 triu->tril 重排
```

## 3. 最关键的特征顺序

NumPy `np.triu_indices(n, k=1)` 按“行”遍历；R `upper.tri(m)` 的赋值按 R column-major 内存顺序按“列”遍历。二者包含同一组 `(i,j)`，但向量位置不同。

当前准备脚本使用：

```python
iu, ju = np.triu_indices(n, k=1)
order = np.lexsort((iu, ju))
counts_r = counts_numpy[:, order]
```

`order` 表示把 NumPy row-major triu 排成 R column-major upper triangle。R 输出恢复到 NumPy 顺序时必须使用逆置换：

```python
inverse_order = np.argsort(order)
restored_numpy = impute_r[inverse_order, :]
```

不能用字符串替换、简单转置或仅改目录名修复顺序。必须通过 `(i,j)` 坐标或完整 permutation 重排。

## 4. 环境与 HPC 资源

当前环境：

```text
Python: /public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python3
Rscript: /public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/Rscript
R package: HiCImpute
```

两套主提交脚本均使用 `cpuQ/cpuq`、20 CPU、60G 内存和 Slurm array。MCMC、二进制转换和全量评估不能在登录节点运行。

## 5. 输入契约

### 5.1 HiCImputeData

- 12 个 `K562_T{1,2,3}_{1k,2k,4k,7k}` 数据集。
- `chr19`，61 bins，100 cells，1830 个非对角三角特征。
- 原始 triplet 位于 `0_gtData/0_downsampled_HiCImputeData/{data_name}/cell_*_chr19.txt`。
- cell 文件必须按文件名中的数字自然排序。
- 准备产物：`schic.bin`、`bulk.bin`、`feature_order.npy`、`obs_names.txt`、`var_names.txt`、`metadata.json`。
- 当前 HiCImputeData 调用不显式传 `expected`。

### 5.2 FLAMINGOData

- 七个条件，500 beads，1500 cells，124750 features。
- 细胞顺序为 T1 的 1..500、T2 的 1..500、T3 的 1..500。
- 准备脚本同时读取 `downsampled_contact_data` 和 `gt_contact_data`。
- 准备产物比 HiCImputeData 多一个 `expected.bin`。

**重要的评估边界**：当前 FLAMINGOData R 流程把 `gt_contact_data` 写入 `expected.bin`，并把它传给 `MCMCImpute(expected=...)`。这意味着当前模拟流程使用了 GT 派生的 expected 信息。迁移到没有 GT 的真实数据时，不能复制这一做法；必须根据 HiCImpute 的统计定义从 bulk/距离分层等可观测信息构建 expected，或不传该参数。报告基准结果时也应注明这一点。

## 6. MCMC 超参数

共同参数：

```r
startval = c(100, 100, 10, 8, 10, 0.1, 900, 0.2, 0, rep(8, n_cells))
cutoff = 0.5
seed = 1234 + task_id
```

当前存在两组迭代设置：

| 用途 | `niter` | `burnin` | 状态 |
|---|---:|---:|---|
| 当前 HiCImputeData 已确认绘图文件 | 5000 | 1000 | 当前基准直接使用 |
| v3 production sbatch 默认 | 50000 | 5000 | 更长链，不能假定自动优于当前文件 |

新数据建议先用 `5000/1000` 做 Slurm pilot，检查链是否运行、输出是否有限和排序是否正确；正式采用哪个迭代数应基于收敛诊断和独立指标，而不是只看文件名。`mc.cores` 当前等于 `SLURM_CPUS_PER_TASK=20`。

## 7. HiCImputeData 运行

权威文件：

```text
1_HiCImputeData/scripts/v3_prepare_hicimpute_hicimputedata.py
1_HiCImputeData/scripts/v3_run_hicimpute_hicimputedata.R
1_HiCImputeData/scripts/v3_submit_hicimpute_hicimpute.sbatch
```

提交：

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/1_HiCImputeData
NITER=5000 BURNIN=1000 sbatch scripts/v3_submit_hicimpute_hicimpute.sbatch
```

脚本先准备 R binary，再运行 12 个 array task。若要保留 production 默认，可不设置两个环境变量。新输出不要直接覆盖当前 `*_niter5000_burnin1000.npz`；先写入新实验目录并完成三角顺序和指标验证。

## 8. FLAMINGOData 运行与正确恢复

### 8.1 MCMC

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData
NITER=5000 BURNIN=1000 sbatch v3_scripts/v3_submit_hicimpute_flamingo.sbatch
```

R 运行后必须保留：

```text
v3_outputData/bin/{data_name}_Impute_All.bin
v3_inputData/{data_name}/feature_order.npy
```

### 8.2 恢复为当前权威 triu NPZ

使用已经验证的 converter：

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/2_imputedContactHeatmap
sbatch nature-style-plot/submit_convert_hicimpute_impute_all.sbatch
```

converter 会：

1. 按 `<f8`、Fortran shape `(124750,1500)` memmap `_Impute_All.bin`；
2. 验证 `feature_order.npy` 是完整 permutation；
3. 使用 `np.argsort(feature_order)` 恢复 NumPy triu；
4. 保存到 `npz_triu_corrected`；
5. 拒绝非有限或全空结果。

FLAMINGOData 的当前 R 脚本注释和直接 NPZ 恢复代码曾存在方向不一致；以 `convert_hicimpute_impute_all.py` 为当前权威实现。

## 9. 输出验证

最少检查：

- 文件数量：HiCImputeData 12，FLAMINGOData 7。
- shape：分别 `(100,1830)` 和 `(1500,124750)`。
- `nnz > 0`，数据有限且不含负值。
- raw binary 的字节数恰为 `n_features*n_cells*8`。
- `feature_order.npy` 排序后等于 `0..n_features-1`。
- 用一个已知 feature `(i,j)` 做 round-trip：NumPy -> R order -> inverse order，值必须回到原位置。
- 同一个 cell 的 prediction、Raw 和 GT 行号确实对应同一 cell ID。

不要用“cell 0 PCC 看起来较高”代替 permutation round-trip 检查；错误排列在某些平滑 contact map 上仍可能给出非零相关。

## 10. PCC、MAE 与缺失位点

统一实现位于：

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/1_pccAndMae_all/recalc_all_metrics.py
```

HiCImputeData：每个 cell 在全部 1830 特征上计算 PCC/MAE；dropout mask 为 `(Raw==0) & (True!=0)`。

FLAMINGOData：prediction 和 GT 先截断为非负并做 `log1p`，再计算 GT-positive 的 all、Raw-positive 的 observed，以及 `GT-positive & not Raw-positive` 的 held-out 指标。

统一重算必须走 Slurm 的 `submit_recalc_metrics_control.sbatch`、`submit_recalc_metrics.sbatch` 和 aggregate 阶段。重算前检查 registry 指向 `npz_triu_corrected`，而不是历史 `npz_lower_tri`。

## 11. Contact map 绘图

HiCImputeData manifest 已把当前 `*_niter5000_burnin1000.npz` 标为 `tril`。当前展示 cell 是 index 0：

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/2_imputedContactHeatmap
sbatch submit_render_all_hicimpute_method_heatmaps.sbatch
sbatch submit_render_all_hicimpute_method_heatmaps_gr.sbatch
```

FLAMINGOData manifest 把 corrected 文件标为 `triu`，7×9 图提交：

```bash
sbatch nature-style-plot/submit_flamingo_heatmap_grid.sbatch
```

FLAMINGO grid 会按每个 cell 的总 contact 归一化到 10,000，并使用 GT 派生的共享色标。绘图不会修复 cell 顺序错误。

## 12. 新数据迁移清单

1. 固定 cell ID 和 feature pair 表，保存到独立 manifest。
2. 根据 bin 数计算 `n_features=n*(n-1)/2`，禁止硬编码 1830/124750。
3. 把 NumPy triu 显式重排为 R `upper.tri` column-major。
4. 按 features×cells、小端 float64、Fortran order 写 binary。
5. 仅用可观测数据构造 `bulk`；没有 GT 时不得从 GT 构造 `expected`。
6. 在 Slurm 上做一小批 MCMC pilot，记录 seed、niter、burnin、cutoff、startval 和包版本。
7. 保留 RDS、原始 `Impute_All.bin` 和 permutation，便于重新恢复。
8. 以 inverse permutation 恢复 NumPy 顺序，再按下游契约决定保存 triu 或 tril。
9. 验证 shape、有限值、非空、round-trip、cell order 和独立指标后，才更新绘图/评估 manifest。

## 13. 常见故障

| 症状 | 主要原因 | 处理 |
|---|---|---|
| contact map 像被打散 | R/Python feature order 方向错误 | 用 `np.argsort(feature_order)` 从 raw bin 重建 |
| 输出 shape 对但 PCC 很差 | 错误 permutation 或 cell order | 做 feature round-trip 和 cell ID 对齐 |
| `Impute_SZ` NPZ 极小 | 结果为空或转换失败 | 不使用；检查 R object 和 raw bin 大小 |
| R 读入矩阵错位 | binary 不是 `<f8` Fortran order | 重新按 features×cells 写出 |
| 重跑仍读取旧结果 | 旧 NPZ 触发 sbatch skip | 新实验使用新输出目录，或确认后移开旧文件 |
| 真实数据无法构造 expected | 当前模拟流程依赖 GT expected | 使用方法允许的可观测 expected 或不传，不能泄漏 GT |

## 14. 权威路径

- HiC 准备/R/提交：`3_HiCImpute/1_HiCImputeData/scripts/`
- FLAMINGO 准备/R/提交：`3_HiCImpute/2_FLAMINGOData/v3_scripts/`
- FLAMINGO corrected converter：`paperplots/2_imputedContactHeatmap/nature-style-plot/convert_hicimpute_impute_all.py`
- 当前输入清单：`paperplots/2_imputedContactHeatmap/hicimpute_heatmap_input_manifest.tsv` 和 `flamingo_heatmap_input_manifest.tsv`

