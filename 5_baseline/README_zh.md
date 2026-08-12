[English](README.md) | **中文**

# 对比方法与论文绘图流程

本目录包含 scHiC-Diff 项目中用于基线评价的可复现流程。

## 目录组织

- [`paperplots/`](paperplots/) 保存论文绘图和评价代码，用于生成论文图、指标汇总、聚类对比、contact map、运行时间图和案例研究分析。
- 本层的各编号方法目录保存项目中运行相应对比方法所需的插补流程、适配代码、预处理脚本、提交脚本和结果整理代码。
- [`0_gtData/`](0_gtData/) 保存各基准流程共用的真值和参考数据准备目录结构。

## 插补方法目录

| 目录 | 方法或用途 |
|---|---|
| [`1_scVI-3D/`](1_scVI-3D/) | scVI-3D 插补流程 |
| [`3_HiCImpute/`](3_HiCImpute/) | HiCImpute 插补流程 |
| [`4_scHiCluster/`](4_scHiCluster/) | scHiCluster 插补流程 |
| [`5_scHiCTools/`](5_scHiCTools/) | scHiCTools 相关对比流程 |
| [`6_Higashi/`](6_Higashi/) | Higashi 插补流程，包括不同邻居参数变体 |
| [`7_scHiCDiff/`](7_scHiCDiff/) | scHiC-Diff 基准推理和结果处理流程 |
| [`8_ScUnicorn/`](8_ScUnicorn/) | ScUnicorn 相关对比流程 |
| [`9_FLAMINGO/`](9_FLAMINGO/) | Tensor-FLAMINGO 插补流程 |

各方法目录保留了进行一致性比较所需的项目专用数据转换、运行方式和输出契约。大型输入及输出矩阵不会上传至 Git；保留的空目录用于表示预期的文件系统结构。

复现某种方法的插补结果时，请先查看该方法目录中的 README。生成论文图或进行跨方法比较时，请进入 [`paperplots/`](paperplots/)，再查看对应编号绘图目录中的双语 README。
