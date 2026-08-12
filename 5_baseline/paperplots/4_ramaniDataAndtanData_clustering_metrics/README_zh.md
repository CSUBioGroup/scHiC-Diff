[English](README.md) | **中文**

# Ramani 与 Tan 聚类评价

本目录包含 Ramani ML1+ML3 和 Tan 数据的正式聚类评价与论文绘图流程。流程读取相邻输入目录中已经完成的 Raw 和插补矩阵，不重新运行任何插补方法。

## 目录结构

```text
inputRamaniData/       # Ramani Raw 输入
imputedRamaniData/     # Ramani 各插补方法输出
inputTanData/          # Tan Raw 输入
imputedTanData/        # Tan 各插补方法输出
nature-plot/           # 指标计算、绘图和正式结果
```

流程重建不同方法的 Ramani embedding，在已选维度和敏感性分析维度上评价 K-means ARI，生成仅用于展示的二维 UMAP 坐标，对 Tan 片段计算 ARI 和对齐后的混淆矩阵，并通过唯一正式绘图入口生成正文图和独立图。

输入契约、已选维度、脚本、Slurm 提交方式、输出和解释边界见[详细流程说明](nature-plot/README.md)。
