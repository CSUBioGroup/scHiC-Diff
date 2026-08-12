[English](README.md) | **中文**

# 插补 Contact Map 论文图

本目录包含两个模拟 scHi-C 基准的 contact map 论文绘图流程。不同数据集的代码、指标表、清单、日志和输出分别保存在 `nature-style-plot/` 下。

## 子项目

- [`nature-style-plot/1_HiCImputedData/`](nature-style-plot/1_HiCImputedData/)：HiCImputeData 正文网格图、7k 补充网格图和各方法单独面板。
- [`nature-style-plot/2_FLAMINGOData/`](nature-style-plot/2_FLAMINGOData/)：FLAMINGOData 7 x 9 对比网格图。
- [`nature-style-plot/gr_style.py`](nature-style-plot/gr_style.py)：共享的论文绘图样式与输出工具。

数据集目录布局见[详细绘图说明](nature-style-plot/README.md)。不同数据集的 CSV 和图应继续保存在对应子目录中。
