# 2_callLoop_apa

这套整理包把当前项目里和 `Map2 locus (chr1: 65-67 Mb, 20 kb, 100 bins)` 相关的两条流程拆开了，方便你后续直接替换成新插补方法的数据继续跑：

1. `loop_compare/`
   - 从 `raw / scHiCluster / scHiC-Diff / 新方法` 的 `(n_cells, upper_triangle_features)` `.npz` 输入出发
   - 为每个 `n = 3, 5, 10, 100, 200, 476` 生成共享子集
   - 调 loop calling
   - 画出 `noLoop_Comparison_*.png` 和带 loop 圈的对比图

2. `apa/`
   - 读取 loop calling 结果
   - 转成 Juicer 需要的 `.bedpe`
   - 对 all / top50 / top100 loops 跑 APA
   - 画出 `rawHiC_depth85` 风格的单张 APA 图

## 目录结构

```text
2_callLoop_apa/
├── ORIGINAL_CODE_MAP.md
├── README.md
├── apa/
│   ├── configs/
│   └── scripts/
├── data/
│   ├── hic_references/
│   └── inputs/
├── examples/
│   └── final_figures/
├── loop_compare/
│   ├── configs/
│   └── scripts/
└── outputs/
```

## 当前案例里的关键约定

- locus: `chr1: 65,000,000-67,000,000`
- resolution: `20,000`
- bins: `100`
- upper triangle feature count: `5050`
- cell type: `earlyNeurons`

`selected_cells.npz` 的生成逻辑现在被显式整理成“共享抽样索引”：

- 先基于参考方法的总细胞数和固定 `seed=42` 为每个 `n` 生成一个 `selected_indices_*.npy`
- 再让所有方法都读同一份索引
- 这样 raw / scHiCluster / scHiC-Diff / 新方法能保证完全使用同一批细胞

这比原来“每个脚本各自用同一个 seed 重抽一次”更直观，也更方便排查。

## 快速使用

### 1) 生成 loop comparison

```bash
python3 loop_compare/scripts/run_loop_compare_case.py \
  --config loop_compare/configs/current_case_loop_compare.json
```

默认会产出：

- `outputs/loop_compare/final/noLoop_Comparison_Raw_Cluster_Diff_earlyNeurons.png`
- `outputs/loop_compare/final/Comparison_Raw_Cluster_Diff_earlyNeurons.png`

### 2) 基于 476-cell loop 结果跑 APA

```bash
python3 apa/scripts/run_apa_case.py \
  --config apa/configs/current_case_rawHiC_depth85.json
```

默认会产出：

- `outputs/apa/rawHiC_depth85/final/APA_scHiCDiff_All_Loops.png`
- `outputs/apa/rawHiC_depth85/final/APA_scHiCDiff_Top50_Loops.png`
- `outputs/apa/rawHiC_depth85/final/APA_scHiCDiff_Top100_Loops.png`
- `outputs/apa/rawHiC_depth85/final/APA_scHiCluster_All_Loops.png`
- `outputs/apa/rawHiC_depth85/final/APA_scHiCluster_Top50_Loops.png`
- `outputs/apa/rawHiC_depth85/final/APA_scHiCluster_Top100_Loops.png`

## 替换成新插补方法时怎么改

### 只做 loop comparison

复制 `loop_compare/configs/template_loop_compare.json`，新增一个方法块：

```json
{
  "name": "MyMethod",
  "slug": "mymethod",
  "input_npz": "../../data/inputs/mymethod/earlyNeurons_mymethod.npz",
  "output_dir": "../../outputs/loop_compare/output_mymethod"
}
```

然后把它加进 `methods` 列表即可。

### 做 APA

如果 APA 想包含新方法，复制 `apa/configs/template_apa_case.json`，增加一个方法块：

```json
{
  "name": "MyMethod",
  "slug": "mymethod",
  "sample_name": "earlyNeurons_476cells",
  "loop_result_dir": "../../outputs/loop_compare/output_mymethod",
  "plot_titles": {
    "all": "MyMethod_All_Loops",
    "top50": "MyMethod_Top50_Loops",
    "top100": "MyMethod_Top100_Loops"
  }
}
```

## 依赖

- Python 3
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `statsmodels`
- Java
- `juicer_tools.2.20.00.jar`

## 说明

- 这套脚本保留了原流程的核心参数和输出命名。
- APA 的参考 `.hic` 在当前案例里是固定外部输入，不再硬编码在脚本内部，而是放到配置文件里。
- 当前 `rawHiC_depth85` 配置中的 reference `.hic` 已对齐能复现旧 `W-200kb/rawHiC_depth85` 结果的 `3_raw/earlyNeurons_476cells.hic`。
- `examples/final_figures/` 下放的是当前项目已经做出的最终图，便于对照。
