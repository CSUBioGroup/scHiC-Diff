# scHiCluster 插补复现与下游使用手册

本文档记录 scHiCluster 在 HiCImputeData 和 FLAMINGOData 上的当前插补流程。重点是单细胞文本输入、`hicluster impute-cell` 参数、每细胞中间文件的收集，以及两个数据族不同的三角特征顺序。

## 1. 当前基准结果

| 数据族 | 当前结果模板 | shape | 真实特征顺序 |
|---|---|---:|---|
| HiCImputeData | `/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/1_HiCImputeData/output/2_lower_tri_npz/{data_name}_scHiCluster_imputed.npz` | `(100,1830)` | NumPy `tril(k=-1)` |
| FLAMINGOData | `/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/2_FLAMINGOData/v3_outputData/2_lower_tri_npz/{data_name}_scHiCluster_imputed.npz` | `(1500,124750)` | NumPy `triu(k=1)` |

FLAMINGOData 的目录名 `2_lower_tri_npz` 是历史名称，文件内容是 `triu`。下游必须按 manifest 的 `feature_order=triu` 解释，不能按目录名推断。

## 2. 方法与本项目实现

当前流程对每个 cell 独立运行：

```text
hicluster impute-cell
  -> 邻域卷积/平滑
  -> random walk with restart
  -> sqrtVC normalization
  -> 每细胞 HDF5 或 NPZ
  -> 按 cell ID 排序并收集成 cells × features NPZ
```

HiCImputeData 使用 HDF5 中间结果；FLAMINGOData 使用 NPZ 中间结果。两者的平滑参数相同，但 collector 的 bin 裁剪和三角顺序不同。

## 3. 环境和资源

当前环境：

```text
/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python
/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/hicluster
```

当前 CPU 任务都使用 `cpuQ/cpuq`。每个数据集内部并行 20 个 cell process，并把 BLAS 线程限制为 1，避免 20 个进程各自再开多线程。

| 阶段 | CPU | 内存 | 时限 |
|---|---:|---:|---:|
| HiCImputeData impute | 20 | 80G | 4h |
| HiCImputeData collect/eval | 20 | 80G | 4h |
| FLAMINGO prepare | 20 | 120G | 6h |
| FLAMINGO impute | 20 | 80G | 48h |
| FLAMINGO collect/eval | 20 | 160G | 12h |

这些步骤会加载或生成大量矩阵，禁止在登录节点直接运行 Python。

## 4. 输入契约

### 4.1 `impute-cell` 文本格式

每个 cell 一个文件：

```text
cell_<id>_<chrom>.txt
```

三列、无表头：

```text
row_bin  col_bin  contact
```

只写非零 contact。cell ID 从 1 开始，最终 collector 按数值 ID 排序。输入目录还必须有 chrom sizes 文件。

### 4.2 HiCImputeData

- `chr19`，resolution 1，真实 61 bins，100 cells。
- 当前历史 chrom sizes 第一行为 `chr19 61`，scHiCluster 因此创建 62×62 中间矩阵。
- collector 只取左上 61×61，再从 HDF5 上三角的转置坐标提取 canonical `tril`。
- chrom sizes 中历史性的 `chr1 0` 行不是新数据所需规范；新数据应只保留真实染色体并验证 bin 数。

### 4.3 FLAMINGOData

- 来源 h5ad 的 `layers['counts']`，`layers['gt']` 仅用于评估。
- 500 bins，1500 cells，124750 features。
- 输入 feature 是 NumPy row-major `triu`，准备脚本写 `chr19 499`，从而得到 500 bins。
- cell 行顺序保持 h5ad 顺序；当前七条件应为 T1 1..500、T2 1..500、T3 1..500。

## 5. 当前超参数

```text
resolution = 1
pad = 1
std = 1.0
rp = 0.5
tol = 0.01
window_size = 500
step_size = 500
mode = pad1_std1_rp0.5_sqrtvc
workers = 20
```

含义和迁移注意：

- `pad/std` 控制局部高斯平滑；分辨率改变后不应无条件照搬。
- `rp=0.5` 是 random-walk restart probability。
- `output_dist` 当前等于 bin 数，表示保留完整距离范围。
- `window_size/step_size=500` 在单染色体 500-bin FLAMINGO 上相当于整段处理；更大真实染色体要按内存重新设置。
- 新数据必须把参数和输出 mode 名一起记录，否则 collector 会找不到中间文件。

## 6. HiCImputeData 流程

权威脚本：

```text
1_HiCImputeData/scripts/01_impute_hicimpute.py
1_HiCImputeData/scripts/02_submit_impute_hicimpute.sbatch
1_HiCImputeData/scripts/05_collect_simu_hdf5.py
1_HiCImputeData/scripts/07_submit_collect_eval.sbatch
```

### 6.1 路径拼写警告

当前最终权威目录已更正为 `1_HiCImputeData`，但上述若干脚本仍硬编码历史拼写 `1_HiCImputeDate`。提交新任务前必须把以下路径统一改到新的实验根目录：

```text
SCRIPT_DIR
INPUT_ROOT
OUTPUT_ROOT
DEFAULT_INPUT_ROOT
DEFAULT_OUTPUT_ROOT
Slurm log 路径
```

可先检查：

```bash
rg -n '1_HiCImputeDate' /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/1_HiCImputeData/scripts
```

未改完前不要直接提交现有 sbatch，否则可能读取旧目录或把新结果写回历史目录。

### 6.2 提交顺序

路径修正并检查输入后：

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/1_HiCImputeData
jid=$(sbatch --parsable scripts/02_submit_impute_hicimpute.sbatch)
sbatch --dependency=afterok:${jid} scripts/07_submit_collect_eval.sbatch
```

impute array 为 12 个数据集。collector 读取每个 HDF5 的 `Matrix` CSR group，裁剪 62×62 到 61×61，并保存 canonical `tril`。

## 7. FLAMINGOData 七条件流程

权威脚本位于：

```text
2_FLAMINGOData/v3_scripts/01_prepare_v3_paramsweep_schicluster.py
2_FLAMINGOData/v3_scripts/03_impute_v3_paramsweep.py
2_FLAMINGOData/v3_scripts/05_collect_v3_paramsweep.py
```

### 7.1 六个 paramsweep 条件

W=0.5、0.6、0.8、0.9 和 P=1%、5% 使用：

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/4_scHiCluster/2_FLAMINGOData
prep=$(sbatch --parsable v3_scripts/02_submit_prepare_v3_paramsweep.sbatch)
imp=$(sbatch --parsable --dependency=afterok:${prep} v3_scripts/04_submit_impute_v3_paramsweep.sbatch)
sbatch --dependency=afterok:${imp} v3_scripts/07_submit_collect_eval.sbatch
```

### 7.2 W=0.7 基准条件

W=0.7 文件位于单独的 `3_500cells_datasets` 源目录，因此使用 end-to-end 作业：

```bash
sbatch v3_scripts/08_submit_500cells_full.sbatch
```

两个来源合并后才是当前 manifest 的全部七条件。

### 7.3 FLAMINGO collector 的真实语义

`05_collect_v3_paramsweep.py` 对每个 500×500 imputed matrix 使用：

```python
np.triu_indices(500, k=1)
```

因此最终 `2_lower_tri_npz` 中的文件是 `triu`。脚本内部变量名 `_load_lower` 和目录名都属于历史命名，不能作为语义依据。

## 8. 输出验证

轻量数量检查：

```bash
find 1_HiCImputeData/output/2_lower_tri_npz -maxdepth 1 -name '*_scHiCluster_imputed.npz' | wc -l
find 2_FLAMINGOData/v3_outputData/2_lower_tri_npz -maxdepth 1 -name '*_scHiCluster_imputed.npz' | wc -l
```

预期 12 和 7。随后在 CPU 节点检查：

- shape 正确；
- 无 NaN/Inf 和负值；
- cell ID 连续，不能只有 99/1499 个文件；
- 中间矩阵尺寸符合预期；
- HiC collector 实际裁剪到 61×61；
- HiC 最终按 `tril`、FLAMINGO 最终按 `triu` 做已知坐标 round-trip；
- 输出不应全零，且各 cell 有合理 `nnz`。

## 9. PCC、MAE 和缺失位点

统一指标入口：

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/1_pccAndMae_all/recalc_all_metrics.py
```

HiCImputeData：逐 cell 对全部 1830 特征计算 PCC/MAE，dropout 为 `(Raw==0) & (True!=0)`。

FLAMINGOData：对非负 `log1p` contact 计算 GT-positive all、Raw-positive observed、`GT-positive & not observed` held-out。

注意：`paperplots/recalc_eval_common.py` 当前仍可能指向历史 `1_HiCImputeDate`。在重算统一指标前，应把 registry 与 `hicimpute_heatmap_input_manifest.tsv` 的 corrected 路径核对一致；两个旧/新目录同时存在时，路径存在检查无法发现读错版本。

## 10. Contact map 绘图

HiC 单图由 manifest 标记为 `tril`，FLAMINGO 7×9 图标记为 `triu`。当前展示 cell 均为 index 0：

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/2_imputedContactHeatmap
sbatch submit_render_all_hicimpute_method_heatmaps.sbatch
sbatch submit_render_all_hicimpute_method_heatmaps_gr.sbatch
sbatch nature-style-plot/submit_flamingo_heatmap_grid.sbatch
```

FLAMINGO grid 将每个矩阵归一化到总 contact 10,000，并用 GT 的共享 `vmax`。图形异常时，先排查三角顺序和 cell order，再调整色图。

## 11. 新数据适配清单

1. 明确每条染色体的 `n_bins`、resolution 和 chrom sizes 的端点约定。
2. 为每个 cell 写三列非零 triplet，并保证 cell ID 连续且自然排序。
3. 根据数据规模调整 window/step；不要让每个进程同时加载超大整染色体矩阵。
4. 在独立输出根目录做 2-3 个 cell 的 Slurm pilot。
5. 固定 `pad/std/rp/tol` 后，把 mode 名同时交给 collector。
6. 明确最终下游需要 `tril` 还是 `triu`，用矩阵坐标提取，不靠目录名。
7. 检查完整 cell 数、shape、有限值、非负值、`nnz` 和 cell order。
8. 更新统一评估 registry 和两个绘图 manifest 后再生成正式结果。

## 12. 常见故障

| 症状 | 原因 | 处理 |
|---|---|---|
| sbatch 写到旧目录 | `1_HiCImputeDate` 仍被硬编码 | 提交前统一修正所有路径变量 |
| HiC HDF5 是 62×62 | 当前 chrom size 61 的端点约定 | 保持 collector 裁剪到 61；新数据优先校正 chrom size |
| collector 找不到文件 | mode 参数或 output format 不一致 | impute/collect 使用相同 `pad/std/rp` 与扩展名 |
| FLAMINGO contact map 被打散 | 把目录名当成 `tril` | 按真实 `triu` 重建 |
| 少量 cell 缺失 | 单 cell impute 失败但未检查 array log | 查 `FAIL cell_*`，补跑后再 collect |
| CPU 利用率异常或内存暴涨 | 进程池与 BLAS 双重并行 | 保持 BLAS 线程为 1，workers 不超过分配 CPU |

## 13. 权威文件

- HiC impute/collect：`4_scHiCluster/1_HiCImputeData/scripts/`
- FLAMINGO prepare/impute/collect：`4_scHiCluster/2_FLAMINGOData/v3_scripts/`
- 当前绘图输入：`paperplots/2_imputedContactHeatmap/hicimpute_heatmap_input_manifest.tsv` 和 `flamingo_heatmap_input_manifest.tsv`

