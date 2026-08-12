# scHiC-Diff 插补复现与下游使用手册

本文档记录 scHiC-Diff 在本项目中的 h5ad 输入、扩散模型配置、Slurm 训练/推理、结果收集和下游使用。当前方法根目录保存结果与配置快照，但生产提交脚本位于源码树和 FLAMINGO 数据目录，不能只查看本目录下当前为空的 `scripts/`。

## 1. 当前基准结果

### 1.1 HiCImputeData

统一指标和绘图读取：

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/1_HiCImputeData/output/npz_lower_tri/{data_name}_scHiCDiff_imputed.npz
```

- shape：`(100,1830)`。
- 当前下游 manifest 将其定义为 canonical `tril(k=-1)`。
- 文件来自已验证的 Batch 10 `v5_scdiff_1mbsucess` 结果收集。

原始 `denoise_recon_inv.npz` 保持其输入 h5ad 的 feature 顺序。新数据收集时必须根据 `var_names` 做坐标映射后再写 canonical NPZ，不能假定所有历史 h5ad 都是同一三角顺序。

### 1.2 FLAMINGOData

当前七条件直接读取每个训练目录的：

```text
{data_name}_scdiff2_v5fast_batched_test_bs1500_testbs9999/denoise_recon_inv.npz
```

完整根目录由数据条件决定：

- W=0.7：`.../3_500cells_datasets/training_results_v5fast_batched_test_bs1500_testbs9999/`
- 其余六条件：`.../5_paramsweep_datasets/training_results_v5fast_batched_test_bs1500_testbs9999/`

shape 为 `(1500,124750)`，feature 顺序与 FLAMINGO h5ad `var_names=chrFLAMINGO_i_j` 一致，即 NumPy row-major `triu(k=1)`。

## 2. 方法在本项目中的含义

scHiC-Diff 是条件扩散去噪模型。当前配置：

1. 从 observed contact 向量构造输入和条件。
2. 训练时按 `mask_strategy=none_zero` 对非零元素 mask，并按配置采样部分零元素。
3. 模型以 `parameterization=x0` 直接预测干净信号。
4. 推理执行扩散去噪并保存归一化及反归一化结果。
5. 下游只使用反归一化 contact 空间的 `denoise_recon_inv.npz`。

`denoise_recon.npz` 位于归一化空间，不能与原始 GT 直接计算 MAE；`denoise_target.npz` 也不是插补结果。

## 3. 环境和源码版本

Python：

```text
/public/home/hpc254701055/micromamba/envs/scdiff2/bin/python
```

环境清单和一键部署脚本位于 `7_scHiCDiff/env/`，用于在新服务器上重建 `scdiff2`：

```text
env/scdiff2_environment.yml   # conda 层 (python 3.9 + cudatoolkit 11.6 + cuml 23.08)
env/scdiff2_requirements.txt  # pip 层 (torch 1.12.1+cu116 + pytorch-lightning 1.9.0 等 172 包)
env/deploy_scdiff2.sh         # 一键部署脚本 (含镜像配置、torch index、验证)
```

在新服务器上部署：

```bash
cd env && bash deploy_scdiff2.sh
# 无 GPU 节点: NO_GPU=1 bash deploy_scdiff2.sh
```

当前两个关键源码树：

```text
HiC Batch 10:
/public/home/hpc254701055/2_projects/10_schicdiff/v5_scdiff_1mbsucess/3_DiffusionModel/scHiC-Diff-master

FLAMINGO current:
/public/home/hpc254701055/2_projects/10_schicdiff/v5_scdiff_fast_batched_test/3_DiffusionModel/scHiC-Diff-master
```

两个版本的 test output 行为不同。FLAMINGO 使用 batched-test 版本，能在 test batch 分批时先拼接再保存；不要换回旧版后仍假定只会生成一个完整结果文件。

## 4. h5ad 输入契约

### 4.1 共同要求

- `X`：模型 observed 输入，非负、cells×features。
- `layers['counts']`：若存在，应与模型实际输入一致。
- `layers['gt']`：模拟数据 GT，仅用于评估，不应作为模型输入。
- `obs`：至少保存稳定 cell ID；行顺序必须保持到 prediction。
- `var_names`：必须唯一并能映射到明确 `(chrom,bin1,bin2)`。

最终 `denoise_recon_inv.npz` 的行和列顺序跟随 h5ad。所有方法横向比较前必须按 ID/feature pair 对齐，不能只比较 shape。

### 4.2 HiCImputeData

- 12 个 h5ad，100 cells、1830 features、61 bins。
- 当前副本位于 `7_scHiCDiff/1_HiCImputeData/input/{dataset}_sim.h5ad`。
- 训练源码使用的数据根为 `1_scHiC/3_DiffusionModel/scHiC-Diff-master/data/SimuData`。
- GT 位于 `5_baseline/0_gtData/1_Gt_HiCImputeData/{dataset}_true.npz`。

### 4.3 FLAMINGOData

- 七个 h5ad，1500 cells、124750 features、500 beads。
- `X/layers['counts']` 是 observed，`layers['gt']` 是 GT。
- `var_names` 为 `chrFLAMINGO_i_j`，当前按 NumPy triu 顺序生成。
- 当前运行在训练前把 `fixed_heldout_indices.npz` 指定的 observed 位置置零，并同步更新 `X` 和 `counts`；GT layer 不改变。

## 5. 关键模型超参数

HiC Batch 10 配置快照：

```text
7_scHiCDiff/1_HiCImputeData/output/configs/batch10_v5_1schic_like_bs128_lr2e4/
```

当前主要参数：

| 参数 | 值 |
|---|---:|
| batch size | 128（HiC）；1500（FLAMINGO） |
| test batch size | 9999（FLAMINGO） |
| learning rate | `2e-4` |
| seed | 10 |
| timesteps | 1000 |
| denoise sample steps | 1000 |
| max epochs | 1000（HiC）；500（FLAMINGO） |
| train/valid split | 0.8/0.2 |
| mask strategy | `none_zero` |
| `mask_none_zero` | 0.8 |
| `zero_to_none_zero` | 0.1 |
| loss | L2, `recon_masked` |
| transformer depth | 6 |
| embedding dimension | 512 |
| heads | 8，head dim 64 |
| condition | cross-attention, 1 token |
| normalize / return_raw | true / true |

Batch 10 v5 还包含 early stopping：`val/loss_MSE_ema`、`min_delta=1e-4`、`patience=25`。ModelCheckpoint 监控的是 `val/loss_ema`，两者不是同一指标。

新数据的 feature 数或稀疏度变化后，batch size、mask ratio 和模型宽度都需要重新评估。不要因为 1500 个 cells 能设 batch 1500，就在更大数据上照搬。

## 6. HiCImputeData 运行

当前权威提交脚本不在 baseline 方法目录，而在：

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/3_DiffusionModel/scHiC-Diff-master/run/SimuData/hpc_run/hpc_v5_1schic_like_bs128_lr2e4_train_hicimpute_scdiff2.sh
```

提交：

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/3_DiffusionModel/scHiC-Diff-master
sbatch run/SimuData/hpc_run/hpc_v5_1schic_like_bs128_lr2e4_train_hicimpute_scdiff2.sh
```

资源为 1 GPU、10 CPU、80G、array `0-11%9`。脚本从 `v5_scdiff_1mbsucess` 运行 `main.py`，但结果写到指定 HiC result root。

训练完成后应从每个 `{dataset}/denoise_recon_inv.npz` 按 h5ad feature pair 收集到新的 canonical 输出目录。当前 baseline 根目录的 `scripts/` 为空，因此不能假定存在自动 collector；新批次应保留一份明确的 Slurm collector 和 conversion report。

## 7. FLAMINGOData 七条件运行

### 7.1 W=0.7

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/3_500cells_datasets
sbatch run_impute_v5fast_batched_test_v3_500cells_bs1500_testbs9999_gpu2q.sbatch
```

### 7.2 其余六条件

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets
sbatch run_impute_v5fast_batched_test_paramsweep_bs1500_testbs9999_gpu2q.sbatch
```

两个任务都使用 `gpu2Q/gpuq`、1 GPU、20 CPU；W=0.7 内存 120G，其余条件 50G。运行前脚本会在 GPU job 内生成 heldout-masked h5ad。若输入很大，建议把这一预处理拆到 CPU Slurm job，并用 dependency 后提交 GPU 训练。

## 8. 输出选择和收集

每个结果目录可能包含：

| 文件 | 是否用于正式指标 | 含义 |
|---|---|---|
| `raw_x.npz` | 否 | observed 输入 |
| `denoise_recon.npz` | 否 | 归一化空间 prediction |
| `denoise_recon_inv.npz` | **是** | 反归一化 contact prediction |
| `denoise_target.npz` | 否 | 模型 target/归一化空间 |

收集规则：

1. 确认文件由完整 test 阶段生成，而不是某个 test batch 的局部覆盖。
2. 按 h5ad `obs_names` 核对行数和 cell 顺序。
3. 按 `var_names` 解析 feature pair。
4. HiC 输出转成下游 canonical `tril`；FLAMINGO 保持 canonical `triu`。
5. 保留原始文件、配置快照、seed、checkpoint 信息和转换报告。

## 9. 输出验证

必须在 CPU 节点检查：

- shape 分别为 `(100,1830)` 和 `(1500,124750)`；
- SciPy sparse 文件可完整读取；
- 无 NaN/Inf，负值处理符合模型定义；
- 行顺序与 h5ad obs 一致；
- feature pair round-trip 后坐标一致；
- `denoise_recon_inv` 与 Raw 不完全相同，也不是全零；
- heldout 位置不是由 GT 写回输入；
- 训练日志、配置和输出目录属于同一 seed/run tag。

## 10. PCC、MAE 和 held-out

统一入口：

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/1_pccAndMae_all/recalc_all_metrics.py
```

HiCImputeData 逐 cell 计算全部特征 PCC/MAE；dropout mask 为 `(Raw==0)&(True!=0)`。

FLAMINGOData 对 prediction/GT 做非负 `log1p`，并计算 GT-positive all、Raw-positive observed 和 `GT-positive & not observed` held-out。当前 scHiC-Diff 路径由 `recalc_eval_common.py` 根据 W=0.7 或 paramsweep root 动态解析。

重新训练后必须先重跑统一指标，再用新的 `FLAMINGOData_unified_PCC_MAE.csv` 或 `HiCImputeData_unified_PCC_MAE.csv` 标图，不能把旧 PCC 写在新 contact map 上。

## 11. Contact map 绘图

当前 HiC 单图和 FLAMINGO grid 都显示 cell index 0：

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/2_imputedContactHeatmap
sbatch submit_render_all_hicimpute_method_heatmaps.sbatch
sbatch submit_render_all_hicimpute_method_heatmaps_gr.sbatch
sbatch nature-style-plot/submit_flamingo_heatmap_grid.sbatch
```

普通/GR 单图只给 scHiC-Diff 保留右侧 colorbar；所有方法都不显示方法标题。FLAMINGO grid 归一化为 contacts per 10,000 并使用 GT 共享色标。图形异常时先检查 feature/cell 对齐，不应先通过换色图掩盖结构问题。

## 12. 新数据适配清单

1. 建立 h5ad，明确 `X/counts/gt` 的职责，GT 不进入模型输入。
2. 保存稳定的 obs ID 和可解析的 var pair；拒绝重复名称。
3. 在 CPU Slurm 上生成 heldout mask，并保存 seed 和 `(cell,feature)` 坐标。
4. 根据 cell 数、feature 数和显存重新选择 train/test batch size。
5. 根据稀疏度做 mask ratio pilot；记录模型代码 commit 和完整 YAML。
6. 在 GPU Slurm 上先跑一个数据集，检查 train、validation、checkpoint、test 和四类 NPZ。
7. 只选择 `denoise_recon_inv.npz`，按明确坐标转换到目标 triangle order。
8. 验证 cell/feature order、shape、有限值、heldout 和独立指标。
9. 更新统一 registry 与绘图 manifest 后再全量重算和绘图。

## 13. 常见故障

| 症状 | 原因 | 处理 |
|---|---|---|
| MAE 极大但归一化指标正常 | 误用了 `denoise_recon.npz` | 改用 `denoise_recon_inv.npz` |
| 结果只有部分 cells | 旧版每 test batch 覆盖保存 | 使用 batched-test 版本或确保 test 一批容纳全部 cells |
| 输出存在但 feature 错位 | 假定 h5ad 是固定三角顺序 | 解析 var pair 后做 canonical 重排 |
| train 很快但结果不可信 | 输入/GT 泄漏或 heldout 未同步 | 核对 X、counts、gt 和 heldout mask |
| 早停和 best checkpoint 不一致 | monitor 指标不同 | 同时检查 `val/loss_MSE_ema` 和 `val/loss_ema` |
| 新版本复现失败 | 源码树或 Lightning 行为变化 | 使用保存的项目/Lightning YAML 和明确源码版本 |

## 14. 当前与历史版本

- **当前 HiC**：Batch 10 `v5_scdiff_1mbsucess`，有 early stopping，最终 consolidated NPZ 用于下游。
- **历史 HiC Batch 6**：原始 `1_scHiC` 版本，仅用于稳定性对照，不作为当前收集来源。
- **当前 FLAMINGO**：`v5_scdiff_fast_batched_test`，batch 1500/test batch 9999。
- **历史 retention 100-cell sweep**：`7_scHiCDiff/2_FLAMINGO/20_FLAMINGOData_baselineBs128Cell300_sweep_h5ad` 下任务用于旧 retention 实验，不是当前七条件生产入口。

## 15. 权威文件

- HiC 配置快照：`7_scHiCDiff/1_HiCImputeData/output/configs/batch10_v5_1schic_like_bs128_lr2e4/`
- HiC 提交：源码树 `run/SimuData/hpc_run/hpc_v5_1schic_like_bs128_lr2e4_train_hicimpute_scdiff2.sh`
- FLAMINGO W=0.7 提交：`3_500cells_datasets/run_impute_v5fast_batched_test_v3_500cells_bs1500_testbs9999_gpu2q.sbatch`
- FLAMINGO 其余条件：`5_paramsweep_datasets/run_impute_v5fast_batched_test_paramsweep_bs1500_testbs9999_gpu2q.sbatch`
- 当前绘图输入：`paperplots/2_imputedContactHeatmap/hicimpute_heatmap_input_manifest.tsv` 和 `flamingo_heatmap_input_manifest.tsv`

