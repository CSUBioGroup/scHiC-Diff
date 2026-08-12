[English](README.md) | **中文**

# 插补运行时间图

本目录用于汇总两套模拟数据上各插补方法的端到端运行时间，并生成正文候选图、单数据集图、
补充分析图及数值汇总表。所有代码路径均相对于本目录解析，不依赖其他服务器上的绝对路径。

## 目录结构

```text
plot_imputation_runtime.py          # 数据读取、聚合和全部图表入口
imputation_runtime_style.py         # 共享字体、配色、坐标轴和运行时间图元
submit_imputation_runtime.sbatch    # cpuQ/cpuq Slurm 入口
data/                               # 原始运行时间记录
figures/                            # PDF 和 600 dpi PNG
results/                            # 汇总 CSV 和 LaTeX 表
logs/                               # Slurm 标准输出和错误日志
```

## 输入数据

```text
data/TensorFLAMINGO_simulations_500x500_runtime.csv
data/HiCImpute_simulations_61x61_runtime.csv
```

这两份 CSV 分别来源于原文件 `all_methods_FLAMINGO_v3_impute_time.csv` 和
`all_methods_HiCImputeData_impute_time.csv`。本次整理只规范化文件名和存放位置，没有修改
任何运行时间、硬件、配置或日志来源记录。

图中使用的两个数据条件名称为：

- `Tensor-FLAMINGO simulations 500x500`
- `HiCImpute simulations 61x61`

CSV 中的 `Higashi_nbr0` 和 `Higashi_nbr5` 仅在图和汇总表中显示为
`Higashi-nbr0` 和 `Higashi-nbr5`，原始数据字段保持不变。
`Tensor-FLAMINGO` 的硬件显示统一为 `CPU x20`；该规范化只作用于图和汇总表，原始
CSV 中的硬件字段不改写。

## 统计口径

- 每个方法以 `impute_time_seconds` 的中位数作为柱或点的位置。
- 深色散点表示每次独立运行；汇总表同时给出样本数、最小值、最大值和均值。
- 横坐标使用秒为单位的对数尺度，顶部标注秒、分钟、小时和天的参照节点。
- 右侧独立读数栏给出中位耗时及主要硬件，避免坐标轴外文字扩大最终图宽。
- 对明确记录为并行 batch 总耗时的方法，平均单数据集图使用总耗时除以数据集数，并以
  `*` 在汇总表中标记；正文柱状图仍展示实际端到端 batch wall time。

## 运行

按照 HPC 规则，完整绘图通过 Slurm CPU 节点执行：

```bash
cd paperplots/0_ImputeTime
sbatch submit_imputation_runtime.sbatch
```

默认同时输出矢量 PDF 和 600 dpi PNG。绘图作业完成后检查：

```bash
squeue -j <JOB_ID>
tail -n 50 logs/imputation_runtime_<JOB_ID>.out
tail -n 50 logs/imputation_runtime_<JOB_ID>.err
```

## 图形输出

`figures/` 中生成以下 PDF/PNG：

```text
imputation_runtime_combined
imputation_runtime_lollipop
imputation_runtime_tensorflamingo_simulations_500x500
imputation_runtime_hicimpute_simulations_61x61
imputation_runtime_hicimpute_simulations_61x61_mean_per_dataset
imputation_runtime_hicimpute_simulations_61x61_scaling
imputation_runtime_summary_table
```

其中 `combined` 是上下排列的双面板运行时间柱状图；`lollipop` 使用相同数据和对数坐标，
以中位数、范围和单次运行点展示分布；两个数据条件分别提供单图。平均单数据集耗时和输入
规模缩放图只针对 HiCImpute simulations 61x61，因为该数据包含可比较的 T1/T2/T3 和
1k/2k/4k/7k 条件。

## 数值输出

`results/` 中生成：

```text
tensorflamingo_simulations_500x500_runtime_summary.csv
hicimpute_simulations_61x61_runtime_summary.csv
imputation_runtime_all_summary.csv
imputation_runtime_summary.tex
```

这些表和图使用同一套加载、硬件归类、batch 判断及时间聚合函数，避免图中数值与补充表
使用不同统计口径。
