[English](README.md) | **中文**

# 发育阶段与 Loop 案例研究

本目录包含发育阶段聚类和长程 loop 案例研究的论文流程。

## 子项目

### 最终论文图

[`nature-plot-new/`](nature-plot-new/) 是最终正文图和补充图的独立绘图包。它读取经过 SHA-256 验证的已复制绘图输入，不重新运行 UMAP、silhouette、插补、loop calling、APA 或 held-out support 计算。

正文图包括发育阶段 UMAP、contact map 与 loop summit、held-out normalized APA，以及 held-out raw-supported fraction。补充输出提供完整方法网格和诊断图。复现命令、数据来源、输出和解释边界见[论文图说明](nature-plot-new/README.md)。

### Loop Calling 与 APA

[`2_callLoop_apa/`](2_callLoop_apa/) 将 Map2 位点流程（`chr1:65-67 Mb`、20 kb、100 bins）拆分为 loop 对比和 APA 两条管线。流程支持不同方法共用细胞抽样，并可通过配置加入新的插补方法。

目录结构、配置模板、依赖和执行命令见 [loop/APA 说明](2_callLoop_apa/README.md)。
