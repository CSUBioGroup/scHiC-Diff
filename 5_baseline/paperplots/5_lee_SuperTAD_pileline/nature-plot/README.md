# Lee 数据 TAD 方法比较

nature-plot 是 Lee PDGFRA 区域 TAD 方法比较的独立可执行项目。所有命令
都必须以本目录为当前工作目录运行，项目代码和输出元数据只使用相对路径。

## 正式输入版本

- 实验输入及 Target pseudo-bulk：../input_leeData
- 各方法插补矩阵：../imputedData

四种细胞类型统一使用以下正式版本：

| 图中方法 | 正式输入目录 |
|---|---|
| HiCImpute | ../imputedData/HiCImpute_fig1_current |
| scVI-3D | ../imputedData/scVI-3D_candidate_per_cell |
| T-FLAMINGO | ../imputedData/FLAMINGO_fixed_contact |

其余正式方法见 LeeData_TAD_method_comparison_input_paths.csv。候选插补版本
评价已完成，其派生代码和输出不再保留。

## 根目录文件

| 文件 | 用途 |
|---|---|
| LeeData_TAD_method_comparison_input_paths.csv | Target 与八种正式方法的相对输入路径 |
| TAD_method_comparison_config.py | 输入清单、细胞类型、基因组窗口和输出路径 |
| step01_calculate_PCC_trials.py | 计算 PCC trial 并选择代表 trial |
| step02_call_SuperTAD_domains.py | 对 Target 和代表矩阵调用 SuperTAD |
| step03_plot_TAD_method_comparison.py | 生成完整比较图、独立子图和直接绘图数据 |
| run_all_TAD_method_comparison_steps.py | 顺序运行三个步骤、验证并记录日志 |
| README.md | 项目结构、规则和复现命令 |

## 目录含义

| 目录 | 内容 |
|---|---|
| intermediate_data/ | 只保存正式绘图和 SuperTAD 所需的代表矩阵 |
| SuperTAD/ | bin/SuperTAD 程序及 domains/ 下的 Target/代表 TAD TSV |
| figures/ | 最终 PDF、PNG、TIFF，不混放 CSV 或 JSON |
| results/ | PCC、代表 trial、绘图数据、运行信息和复现检查 |
| logs/ | 每次总流水线产生的时间戳日志 |
| tests/ | 数值、路径、TAD 坐标、绘图和流水线测试 |
| tools/ | 不参与主流程的文件清单工具 |
| docs/ | 目录设计、实施计划及项目迁移记录 |

### intermediate_data/

intermediate_data/representative_matrices/<方法>/ 对每种细胞类型只保留一个
代表 pseudo-bulk NPZ，不保存全部 trial 矩阵。

### SuperTAD/

- SuperTAD/bin/SuperTAD：本项目使用的程序。
- SuperTAD/domains/target/：四个 Target TSV。
- SuperTAD/domains/representatives/<方法>/：每个方法和细胞类型的代表 TSV。

### figures/

figures/TAD_method_comparison/ 保存九个最终导出：

- TAD_method_comparison.{pdf,png,tif}
- panel_A_TAD_heatmaps.{pdf,png,tif}
- panel_B_PCC_violin_plots.{pdf,png,tif}

### results/

| 路径 | 含义 |
|---|---|
| PCC_trials_by_method/<方法>/<方法>_PCC_trials.csv | 四种细胞类型的全部 PCC trial |
| PCC_trials_by_method/<方法>/<方法>_PCC_calculation_information.json | 输入、抽样数和相对输出路径 |
| selected_representative_trials.csv | 32 个代表 trial、PCC、抽样细胞和矩阵路径 |
| PCC_method_comparison_summary.csv | 子图 B 的统计量、效应量和 bootstrap 区间 |
| TAD_boundary_plot_check.csv | 子图 A 上下半区的矩阵、TAD、trial 和层级检查 |
| TAD_method_comparison_run_information.json | 输入版本、参数、TAD 规则和相对路径 |
| final_experiment_summary.md | 当前正式结果与复现摘要 |
| reproducibility_verification/ | 迁移哈希、删除记录和完整性检查 |

### logs/

总入口写入 logs/TAD_method_comparison_YYYYMMDD_HHMMSS.log。项目工作目录
在日志中记录为点号，不保存本机绝对路径。

## 环境和命令

~~~bash
mamba activate 10_snaphic_env
~~~

验证当前结果，不重新计算：

~~~bash
python run_all_TAD_method_comparison_steps.py --verify-only
~~~

从正式输入完整重算：

~~~bash
python run_all_TAD_method_comparison_steps.py --force
~~~

也可以按编号单独执行：

~~~bash
python step01_calculate_PCC_trials.py --force
python step02_call_SuperTAD_domains.py --force
python step03_plot_TAD_method_comparison.py --force
~~~

已有正式输出默认受保护，只有显式使用 --force 才允许替换。

## 科学与绘图规则

子图 A：

- 上三角使用 Target pseudo-bulk 和 Target SuperTAD。
- 下三角使用同一方法、同一代表 trial 的矩阵和 SuperTAD。
- SuperTAD TSV 从 1-based inclusive 转为 NumPy 的 0-based inclusive。
- 与窗口相交的 domain 只平移端点，不用 max/min 强制裁剪。
- Target 和全部方法统一选择 deepest non-singleton domain。
- Target 与 Method TAD 使用不同线型；上下三角间只有一条灰色虚线。
- 每个小图右上角标注 Target，左下角 PCC 保留三位小数；二者统一为
  6.2 pt、正常字重。
- 方法名仅在每列底部显示一次，并使用完整名称 Higashi-nbr0/5；不再显示
  底部 TAD 线型图例和 lower/upper 注释。
- 图形专用方法顺序为 Raw、scHiCluster、HiCImpute、Higashi-nbr0、
  Higashi-nbr5、scVI-3D、T-FLAMINGO、scHiC-Diff；这不改变 PCC、
  代表 trial 或 SuperTAD 的上游计算顺序。
- Astro、Endo、ODC、OPC 标签分别使用 #3F6FAE、#D6A33B、#8A6BB1、
  #3C927D；完整图中的 A/B 子图字母统一为 12 pt 粗体，独立 Panel A/B
  导出不显示子图字母。
- Panel A 顶部统一标注 `chr4:55.09–55.17 Mb`；文字位于中间两列中心，
  距图顶 4.6 mm。两侧横线长度均为实际文字宽度的 60%，不超过 2/3，
  与文字间隔 1.5 mm。
- 正文完整图的 A–B 显式间距固定为 5 mm；独立子图尺寸不受影响。

子图 B 默认使用 unpaired bootstrap。TAD boundary F1 和 domain Jaccard
不属于本项目保留的评价指标。

## 完整性验证

--verify-only 检查：

- 8 个方法 × 4 种细胞类型 × 100 个 PCC trial，共 3,200 行；
- 32 个代表矩阵；
- 4 个 Target TSV 和 32 个代表 SuperTAD TSV；
- 9 个非空图像导出；
- 直接结果文件完整；
- 正式 Python、CSV、JSON 和 Markdown 中没有本机绝对项目路径。
