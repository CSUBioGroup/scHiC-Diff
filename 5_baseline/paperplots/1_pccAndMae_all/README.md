# 插补结果 PCC、MAE 与 SCC 计算

本目录统一计算 HiCImputeData 和 FLAMINGOData 的插补评价指标。两类数据
共享同一个 Python 入口和同一套方法路径配置，最终结果按数据类型分别保存。

以下命令均以 `paperplots/1_pccAndMae_all/` 为当前工作目录。

## 目录内容

```text
1_pccAndMae_all/
├── calculate_imputation_metrics.py
├── imputation_metric_config.py
├── submit_metric_array.sbatch
├── submit_metric_control.sbatch
├── imputation_metric_tasks.csv
├── 1_HiCImputedData/
│   ├── HiCImputeData_PCC_MAE_SCC_metrics.csv
│   ├── per_cell_metrics/
│   └── README.md
├── 2_FLAMINGOData/
│   ├── FLAMINGOData_PCC_MAE_SCC_metrics.csv
│   ├── per_cell_metrics/
│   └── README.md
└── logs/
```

- `calculate_imputation_metrics.py`：唯一指标计算入口，提供
  `prepare-manifest`、`run-task` 和 `aggregate` 三个子命令。
- `imputation_metric_config.py`：两类数据、七种方法、原始矩阵路径、特征顺序
  和矩阵读取规则的唯一配置文件。
- `imputation_metric_tasks.csv`：由配置文件生成的 133 个计算任务；包含
  49 个 FLAMINGOData 任务和 84 个 HiCImputeData 任务。
- `per_cell_metrics/`：每个“方法 × 数据条件”的逐细胞指标 JSON。最终 CSV
  由这些 JSON 汇总得到，不能只修改 CSV 而不更新逐细胞结果。
- `logs/`：Slurm 标准输出和错误日志。

三个核心文件的关系为：

```text
imputation_metric_config.py              # 人工维护的路径与格式配置
        │ prepare-manifest
        ▼
imputation_metric_tasks.csv              # 自动生成的 Slurm 任务快照
        │ run-task / Slurm array
        ▼
1_HiCImputedData/per_cell_metrics/       # 84 个逐细胞 JSON
2_FLAMINGOData/per_cell_metrics/         # 49 个逐细胞 JSON
        │ aggregate
        ▼
HiCImputeData_PCC_MAE_SCC_metrics.csv    # 84 条正式汇总
FLAMINGOData_PCC_MAE_SCC_metrics.csv     # 49 条正式汇总
```

`imputation_metric_config.py` 是唯一需要人工维护的配置源。
`imputation_metric_tasks.csv` 是 prepare 阶段生成的派生文件，不应手工修改；
配置变化后应重新运行 prepare。

## 方法和数据覆盖

七种方法为：

```text
scVI-3D, HiCImpute, scHiCluster, Higashi_nbr0, Higashi_nbr5,
Tensor-FLAMINGO, scHiC-Diff
```

HiCImputeData 覆盖 `T1/T2/T3 × 1k/2k/4k/7k` 共 12 个条件，最终表含
`7 × 12 = 84` 条记录。FLAMINGOData 覆盖 W 扫描 0.5–0.9，以及
`W=0.7` 下的 `P=1%`、`P=5%`，最终表含 `7 × 7 = 49` 条记录。

Tensor-FLAMINGO 的 HiCImputeData 正式指标使用
`9_FLAMINGO/1_HiCImputeData/output_distance_best/contact_from_pd/npz_lower_tri/`。
旧的 `output_distance/contact_from_pd/npz_lower_tri/` 在 11/12 个条件下产生
零方差预测，heldout PCC/SCC 会变为 NaN，因此不再作为正式评价输入。

## 指标定义

每个细胞分别计算 PCC、MAE 和 SCC，再跨细胞计算均值与总体标准差
（`numpy.nanstd`，`ddof=0`）。三个评价子集为：

| 后缀 | 特征范围 |
|---|---|
| `all` | 完整三角特征向量，包含 GT 为 0 的位置 |
| `obs` | observed/input 值大于 0 的位置 |
| `held` | GT 大于 0 且 observed/input 不大于 0 的位置 |

PCC 是 Pearson correlation coefficient，SCC 是 Spearman rank
correlation coefficient，MAE 是 mean absolute error。字段名采用
`{metric}_{subset}_{mean|std}`，例如 `pcc_held_mean`。

逐细胞指标无法计算时记录为 NaN；aggregate 使用 `numpy.nanmean` 和
`numpy.nanstd` 忽略这些 NaN。每个 JSON 同时记录各指标的 NaN 数量，正式
使用结果前应检查异常数量，而不应只查看最终均值。

HiCImputeData 与 FLAMINGOData 均使用保存的原始数值直接计算，不做 `log`、
`log1p`、截负值或额外归一化；mask 始终由原始 GT 和 observed 定义。三角特征
规则及 FLAMINGOData 的 `denoise_recon_inv.npz` 输入说明见各数据子目录 README。

## Slurm 重算流程

不要在登录节点加载矩阵或运行指标计算。先生成并验证任务清单：

```bash
mkdir -p logs
sbatch submit_metric_control.sbatch prepare
```

确认 prepare 作业成功且 `imputation_metric_tasks.csv` 为 134 行（含表头）后，
提交 133 个数组任务。`%10` 用于限制并发，可按集群资源调整：

```bash
sbatch --array=0-132%10 submit_metric_array.sbatch
```

全部数组任务成功后汇总：

```bash
sbatch submit_metric_control.sbatch aggregate
```

也可以使用 `afterok` 依赖自动门控 aggregate：

```bash
array_job=$(sbatch --parsable --array=0-132%10 submit_metric_array.sbatch)
sbatch --dependency=afterok:${array_job} submit_metric_control.sbatch aggregate
```

计算环境固定为 `micromamba/envs/2_schic-scvi-3d`，Slurm 使用
`cpuQ/cpuq`。如方法输出路径、文件名或特征顺序发生变化，只修改
`imputation_metric_config.py`，然后从 prepare 开始完整重跑。

## 局部重算

主程序支持 `--method`、`--dataset-family` 和 `--dataset` 过滤；正式结果目录
仍共用固定文件名。因此，局部重算时应始终保留完整的 133 行任务清单，只提交
需要更新的 task ID。

例如只重算 FLAMINGOData 的全部七种方法：

```bash
array_job=$(sbatch --parsable --array=0-48%7 submit_metric_array.sbatch)
sbatch --dependency=afterok:${array_job} submit_metric_control.sbatch aggregate FLAMINGOData
```

这会重写 49 行 `FLAMINGOData_PCC_MAE_SCC_metrics.csv`，不会触碰
HiCImputeData 的 84 行 CSV。`aggregate FLAMINGOData` 会拒绝包含非 `raw`
变换的 FLAMINGOData JSON，防止旧的 `log1p` 结果与新结果混合。

如果仅更新一个方法，并且该数据族的其余逐细胞 JSON 已完整存在，可以先从
完整清单中筛出 task ID。例如重算 scVI-3D：

```bash
method="scVI-3D"
task_ids=$(awk -F, -v method="${method}" \
  'NR > 1 && $2 == "FLAMINGOData" && $4 == method {ids = ids (ids ? "," : "") $1} END {print ids}' \
  imputation_metric_tasks.csv)
array_job=$(sbatch --parsable --array="${task_ids}%10" submit_metric_array.sbatch)
sbatch --dependency=afterok:${array_job} submit_metric_control.sbatch aggregate FLAMINGOData
```

上例适用于 FLAMINGOData；HiCImputeData 则将最后参数改为
`HiCImputeData`。aggregate 若发现所选数据族中任何方法/条件的 JSON 缺失会
直接失败，不会生成不完整结果。

如果需要一张只包含单个方法、且与正式 CSV 完全隔离的独立汇总表，当前脚本
尚未提供独立 `--output-dir`；应先扩展输出目录参数，不能复用正式 aggregate
输出位置。

## 与绘图目录的关系

contact-map 绘图使用各自目录中的轻量指标副本：

```text
../2_imputedContactHeatmap/nature-style-plot/1_HiCImputedData/HiCImputeData_PCC_MAE_metrics.tsv
../2_imputedContactHeatmap/nature-style-plot/2_FLAMINGOData/FLAMINGOData_PCC_MAE_metrics.tsv
```

重算后如需更新热图 PCC 标注，应从本目录最终 CSV 中按
`method + data_name/dataset` 复制 PCC/MAE 字段到对应 TSV。绘图脚本不会
在绘图时重新计算 PCC，也不会自动覆盖指标副本。
