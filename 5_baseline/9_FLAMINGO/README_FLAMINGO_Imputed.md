# Tensor-FLAMINGO 插补复现与下游使用手册

本文档记录本项目中 Tensor-FLAMINGO（FLAMINGO t-SVD/LRTC）的两套流程。HiCImputeData 在 power-law distance（PD）空间补全；FLAMINGOData 在 raw contact 空间直接补全。两套流程的变换、ADMM `selection` 和最终文件格式不同，不能混用。

## 1. 当前基准实际读取什么

### 1.1 HiCImputeData

当前统一指标和绘图读取：

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/1_HiCImputeData/output_distance/contact_from_pd/npz_lower_tri/{data_name}_flamingo_lower_tri.npz
```

- shape：`(100,1830)`。
- 数据空间：已从 PD 反变换回 contact/IF。
- 特征顺序：NumPy `tril(k=-1)`。

另有 `output_distance_best/contact_from_pd/npz_lower_tri/`，来自推荐的 `selection=best` PD 流程。当前 benchmark manifest 尚未切换到该目录；若正式替换，必须同步重算指标和图片。

### 1.2 FLAMINGOData 当前临时结果

当前绘图 manifest 仍读取：

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData/v3ContactOutput/{subdir}/completed_tensor.npy
```

每个 tensor shape 为 `(1500,500,500)`。这批旧文件具有特殊历史编码：canonical `triu` feature 序列被放在 tensor 的 `tril` 坐标中，所以 manifest 标记为：

```text
tri_tensor_tril_encoded_triu
```

它不是普通的对称 contact tensor。当前 heatmap loader 会先从 `tril` 取向量，再按 `triu` 语义重建。

### 1.3 FLAMINGOData corrected triu 流程

推荐的新流程是：

```text
2_FLAMINGOData/run_v3_h5ad_contact_lrtc_triu_cpu20_array.sbatch
2_FLAMINGOData/v3ContactInput_triu/
2_FLAMINGOData/v3ContactOutput_triu/
```

只有在七个 `completed_tensor.npy` 全部完成、评估通过且矩阵坐标验证正确后，才能用它替换 manifest。替换时还必须修改统一 evaluator：旧 `load_tensor_flamingo_lower_tri()` 特意抽取 legacy lower 编码；corrected 对称 tensor 应直接抽取 NumPy `triu(k=1)` 与 h5ad 对齐。只改路径是不够的。

## 2. 方法原理和两种输入空间

Tensor-FLAMINGO 把单细胞 contact map 堆叠为三阶 tensor，并用 t-SVD nuclear norm 与 ADMM 做 low-rank tensor completion。

### 2.1 HiCImputeData：PD 空间

```text
observed contact/IF
  -> PD = IF^(-0.25)，零值保持零
  -> 对称 PD tensor
  -> t-SVD LRTC
  -> contact = PD^(-4)，零值保持零
  -> canonical tril NPZ
```

当前指数：

```python
ALPHA = 0.25
PD = IF ** (-ALPHA)
IF = PD ** (-1.0 / ALPHA)
```

PD 接近 0 时反变换会爆炸。例如 `PD=0.02` 对应 contact 约 `6.25e6`。因此不能把小 PD 简单 floor 到正数；无效/过小值应根据明确阈值置为无 contact，并检查转换后的分布。

#### 2.1.1 已有的具体 PD 阈值方案

项目中已有一套可执行的小 PD 过滤方案，实现在：

```text
9_FLAMINGO/5_lee_SuperTAD_pileline/scripts/lee_flamingo_pipeline.py
```

默认参数为 `clip_factor=2.0`、`clip_min_pd=None`、`keep_observed_values=True`。对一个 `cell_type` 的全部细胞共同计算阈值，而不是逐细胞计算。设：

```text
C_obs_max = max(IF_observed[IF_observed > 0])
f         = clip_factor，默认 2.0
C_limit   = f * C_obs_max
PD_min    = C_limit^(-ALPHA) = C_limit^(-0.25)
```

completed PD 转 contact 时执行：

```python
tiny = (completed_pd > 0) & (completed_pd < PD_min)
completed_pd[tiny] = 0.0

completed_contact = np.zeros_like(completed_pd)
valid = np.isfinite(completed_pd) & (completed_pd > 0)
completed_contact[valid] = completed_pd[valid] ** (-4.0)
```

该规则等价于：若某个小正 PD 会产生 `IF > 2 * C_obs_max`，则把它视为 t-SVD 反演伪影并将 contact 置为 0。这里不是把过大的 contact 截到 `C_limit`，也不是把 PD 抬高到 `PD_min`；实际代码会直接把过小 PD 置零。这样可避免大量缺失位点被人为赋成同一个很大的 contact。

例如 `C_obs_max=10` 时，默认 `C_limit=20`，`PD_min=20^(-0.25)≈0.473`。`PD=0.4` 会反变换为 `39.063`，超过上限，因此置零；`PD=0.5` 会反变换为 `16`，因此保留。

参数优先级和关闭方式如下：

- `clip_min_pd` 设置为正数时，它优先于 `clip_factor`，所有 `PD < clip_min_pd` 的值直接置零。虽然参数名中有 `min`，其当前实现语义是过滤阈值，不是数值 floor。
- `clip_min_pd=None` 且 `clip_factor` 为有限值时，使用上面的观测最大值推导阈值；默认 `clip_factor=2.0`。
- `clip_min_pd=None` 且 `clip_factor=np.inf` 时，不执行小 PD 阈值过滤。
- 阈值过滤和 `PD^-4` 完成后，先强制矩阵非负、对称且对角线为 0；当 `keep_observed_values=True` 时，再把原始 observed contact 的正值位置恢复为原值。阈值只决定补全位置的异常值处理，不应改写真实观测值。

必须注意当前接入状态：这套阈值逻辑目前只在上述 Lee pipeline 中实现。`1_HiCImputeData/scripts/convert_pd_to_contact.py` 仍会把每个有限正 PD 直接按 `PD^-4` 转换，没有 `clip_factor` 或 `clip_min_pd` 参数。因此，当前 benchmark 使用的 `output_distance/contact_from_pd/` 文件不能仅依据本节就宣称已经做过该阈值过滤；如需启用，必须先把相同逻辑接入 converter，在新目录重生成结果，再比较 held-out PCC/MAE、过滤比例和 contact 分布后更新 manifest。

### 2.2 FLAMINGOData：contact 空间

```text
h5ad layers['counts'] raw contact
  -> 500x500 对称 contact txt
  -> t-SVD LRTC
  -> completed contact tensor
```

该流程不做 PD 转换，`log1p` 只用于统一指标，不是模型输入变换。

## 3. 环境、runner 和 HPC

当前环境：

```text
/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python
```

共享 runner：

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/run_flamingo_pyfftw_completion.py
```

runner 从 `{input_root}/{dataset}/{input_subdir}/RawCount_Cell_*.txt` 读取每细胞矩阵，并写 `completed_tensor.npy`。

当前资源：

| 数据族 | CPU | 内存 | 时限 | array |
|---|---:|---:|---:|---:|
| HiCImputeData PD | 8 | 16G | 12h | `0-14` |
| FLAMINGOData contact | 20 | 50G | 96h | `0-6` |

全量文本矩阵写出、FFT/t-SVD、tensor 转换和指标都属于高负载任务，必须使用 `cpuQ/cpuq`，不能在登录节点运行。

## 4. ADMM 超参数

共同参数：

```text
mu = 1e-4
max_mu = 1e10
rho = 1.1
max_iter = 500
tol = 1e-4
keep_observed = true
```

关键差异：

| 流程 | `selection` | 原因 |
|---|---|---|
| HiCImputeData PD 推荐 | `best` | 选择 residual 最低的中间迭代，避免后期 PD 退化 |
| HiCImputeData 历史 current manifest | `final` 产物目录 | 主要复现当前 benchmark；可能接近保留 observed 的 no-op |
| FLAMINGOData contact | `final` | 当前 contact-space 实验使用最终迭代 |

PD 空间使用 `selection=final` 时，大量 missing entry 可能回到 PD=0，反变换后仍为 contact=0；指标可能主要反映 `keep_observed` 保留的原始 contact，而非真实补全。新数据优先比较 `best` 与 held-out 指标。

## 5. HiCImputeData 输入契约

输入 manifest：

```text
1_HiCImputeData/input_distance/manifest.tsv
```

它包含 12 个 T1/T2/T3 数据集和 3 个合并 ALL 数据集，共 15 个 array task。单个类型数据：

- observed/true NPZ 均为 cells×1830；
- 61 beads；
- canonical `tril`；
- observed 和 truth 的 cell 顺序必须一致。

准备脚本把 contact 转为 PD 后写：

```text
input_distance/{dataset}/distance_matrices/RawCount_Cell_<NNN>.txt
input_distance/{dataset}/observed_distance_tensor.npy
input_distance/{dataset}/truth_distance_tensor.npy
input_distance/{dataset}/input_file_index.csv
input_distance/{dataset}/metadata.json
```

GT distance tensor只用于评估，不能传入 LRTC runner 作为 observed。

## 6. HiCImputeData 推荐运行

权威脚本：

```text
1_HiCImputeData/scripts/hicimpute_distance_lrtc.py
1_HiCImputeData/scripts/convert_pd_to_contact.py
1_HiCImputeData/run_hicimpute_distance_lrtc_best_cpu_array.sbatch
```

提交 `selection=best`：

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/1_HiCImputeData
jid=$(sbatch --parsable run_hicimpute_distance_lrtc_best_cpu_array.sbatch)
sbatch --dependency=afterok:${jid} combine_hicimpute_distance_lrtc_best_metrics_cpu.sbatch
```

完成后还需在 CPU 节点把 best PD tensor 转回 contact。可提交：

```bash
BASE=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/1_HiCImputeData
sbatch -A pi_limin_r -p cpuQ -q cpuq -c 4 --mem=32G -t 04:00:00 \
  --wrap="/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python ${BASE}/scripts/convert_pd_to_contact.py --input-root ${BASE}/input_distance --output-root ${BASE}/output_distance_best --contact-root ${BASE}/output_distance_best/contact_from_pd --no-txt"
```

当前 converter 的新文件名可能包含 `_flamingo_contact_lower_tri.npz`，而统一 manifest 使用标准名 `_flamingo_lower_tri.npz`。正式接入时应明确标准化命名并更新 manifest，不能仅复制旧文件名而不记录来源。

历史 `run_hicimpute_distance_lrtc_cpu_array.sbatch` 使用 `selection=final`，用于复现当前 `output_distance`，不作为新数据首选。

## 7. FLAMINGOData 输入契约

输入 h5ad：

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets/{data_name}_scdiff2.h5ad
```

- 七个条件，1500 cells、500 beads、124750 features。
- `layers['counts']` 是 observed，`layers['gt']` 是 GT。
- `var_names=chrFLAMINGO_i_j` 给出 canonical NumPy triu pair。
- cell 顺序保持 h5ad obs，当前应为 T1 1..500、T2 1..500、T3 1..500。

`scripts/v3_h5ad_contact_lrtc.py prep` 会保存 observed/truth feature NPZ、tensor、cell index 和 metadata，并写 1500 个 `RawCount_Cell_####.txt`。`.complete` 会导致重复运行跳过，因此更换 feature-order 实现时必须使用新的 input root，不能复用旧 marker。

## 8. FLAMINGOData corrected 运行

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData
sbatch run_v3_h5ad_contact_lrtc_triu_cpu20_array.sbatch
```

该脚本使用：

```text
INPUT_PARENT=v3ContactInput_triu
OUTPUT_PARENT=v3ContactOutput_triu
feature_order=numpy_row_major_triu
selection=final
keep_observed=true
```

完成标准不是 Slurm task 离开队列，而是每个数据集都有：

```text
completed_tensor.npy
process_time.tsv（含 end_epoch/elapsed_seconds）
v3_h5ad_contact_lrtc_cell_level_metrics.csv
v3_h5ad_contact_lrtc_summary_metrics.csv
```

七个 tensor 均通过坐标、对称性、shape 和指标验证后，再提交适配 `v3ContactOutput_triu` 的 merge job并替换下游路径。

## 9. legacy 与 corrected tensor 的解码差异

### legacy 当前绘图文件

```python
vector = tensor[cell][np.tril_indices(500, k=-1)]
matrix = interpret_vector_as_triu_then_mirror(vector)
```

### corrected 对称 tensor

```python
matrix = tensor[cell]
vector = matrix[np.triu_indices(500, k=1)]
```

迁移 corrected 文件时必须同时修改：

- `flamingo_heatmap_input_manifest.tsv` 的路径与 `feature_order`；
- `nature-style-plot/flamingo_manifest_data.py` 的 tensor loader 行为；
- `paperplots/recalc_eval_common.py` 的 dense tensor feature extraction；
- unified PCC/MAE CSV 和最终 contact map。

否则可能出现图看起来对称但 PCC 使用错误 feature permutation 的情况。

## 10. 输出验证

HiCImputeData contact-from-PD：

- shape `(100,1830)`，canonical `tril`；
- PD 和 contact 都有限、非负；
- observed 位置由 `keep_observed` 保留；
- missing contact 分布无由极小 PD 引起的巨大离群值；
- best 结果的 held-out 指标优于或至少不同于 Raw/no-op。

FLAMINGOData corrected：

- 七个 `(1500,500,500)` tensor；
- 对称、主对角线处理一致、有限、非负；
- observed 位置保持；
- triu 向量与 h5ad feature pair 一一对应；
- 无全零 cell；
- `process_time.tsv` 有结束时间；
- 不再需要 legacy `tril_encoded_triu` 解码。

## 11. PCC、MAE 与 held-out

统一入口为 `paperplots/1_pccAndMae_all/recalc_all_metrics.py`。

HiCImputeData：在 contact 空间逐 cell 计算全部 feature PCC/MAE；dropout=`(Raw==0)&(True!=0)`。不能直接拿 PD tensor与 contact GT 计算这些指标。

FLAMINGOData：prediction/GT 截断非负后做 `log1p`，计算：

```python
all      = GT > 0
observed = Raw > 0
heldout  = (GT > 0) & ~(Raw > 0)
```

corrected tensor 接入前必须先修正 dense tensor extraction；当前 evaluator 的 lower extraction是为 legacy 编码保留的。

## 12. Contact map 绘图

当前 HiC 图展示 cell index 0，并按 `tril` 重建 61×61：

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/2_imputedContactHeatmap
sbatch submit_render_all_hicimpute_method_heatmaps.sbatch
sbatch submit_render_all_hicimpute_method_heatmaps_gr.sbatch
```

FLAMINGO 7×9 图：

```bash
sbatch nature-style-plot/submit_flamingo_heatmap_grid.sbatch
```

grid 会把每张矩阵归一化到总 contact 10,000，并使用 GT-derived shared color range。legacy tensor 必须由专用 decoder 处理；corrected tensor 应直接作为对称矩阵，不得再次 triangle-reorder。

## 13. 新数据适配清单

1. 明确使用 contact 还是 PD 空间，并记录理由。
2. 固定 cell ID 与 feature pair order；从 manifest 驱动所有转换。
3. contact->PD 时零值保持零，记录 alpha；PD->contact 时防止小正值爆炸。
4. 在 CPU Slurm 上准备少量 cell，检查对称矩阵和 `RawCount_Cell` 命名。
5. 小规模比较 `selection=best/final`、held-out 指标和 missing-value 分布。
6. 保持 observed 还是允许模型更新 observed，应作为显式 `keep_observed` 决策。
7. 全量任务写入新 input/output root，避免复用 `.complete` 或旧 tensor。
8. 验证 tensor 坐标语义后，生成 canonical NPZ 或声明 tensor decoder。
9. 同步更新 evaluator、manifest、PCC CSV 和图片。

## 14. 常见故障

| 症状 | 原因 | 处理 |
|---|---|---|
| PD 转 contact 后出现巨值 | completed PD 有很小正值 | 检查 selection/residual，并按明确阈值置零而非 floor |
| PCC 接近 Raw、missing 仍全零 | PD `selection=final` 退化为 no-op | 使用 `selection=best` 并看 held-out 指标 |
| corrected 图仍错位 | 只换路径，仍使用 legacy decoder | 同步修改 heatmap 和 evaluator extraction |
| task 运行很久无日志 | 正在写大量文本或做 FFT/t-SVD | 查 `sstat`/文件增长；不要在登录节点重复启动 |
| `.complete` 后输入没更新 | 复用了旧 input root | 新 feature-order 流程使用新目录或显式 `--force` |
| tensor shape 对但下游 PCC 错 | triangle sequence 与坐标语义不一致 | 用已知 `(i,j)` round-trip 验证，不靠 shape |

## 15. 权威文件

- HiC PD 准备/评估：`9_FLAMINGO/1_HiCImputeData/scripts/hicimpute_distance_lrtc.py`
- HiC PD->contact：`9_FLAMINGO/1_HiCImputeData/scripts/convert_pd_to_contact.py`
- 已有 PD 小值阈值实现：`9_FLAMINGO/5_lee_SuperTAD_pileline/scripts/lee_flamingo_pipeline.py` 的 `post_cell_type()`
- HiC 推荐提交：`run_hicimpute_distance_lrtc_best_cpu_array.sbatch`
- FLAMINGO contact 准备/评估：`9_FLAMINGO/2_FLAMINGOData/scripts/v3_h5ad_contact_lrtc.py`
- FLAMINGO corrected 提交：`run_v3_h5ad_contact_lrtc_triu_cpu20_array.sbatch`
- 共享 LRTC runner：`4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/run_flamingo_pyfftw_completion.py`
- 当前绘图输入：`paperplots/2_imputedContactHeatmap/hicimpute_heatmap_input_manifest.tsv` 和 `flamingo_heatmap_input_manifest.tsv`
