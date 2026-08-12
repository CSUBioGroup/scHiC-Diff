# Lee PDGFRA SuperTAD Pipeline

## 概述

本管线在 Lee et al. 人类皮层 scHi-C 数据集上评估插补方法，聚焦 PDGFRA 基因位点 (chr4:55095461-55164412, hg19, 10kb 分辨率)。

实验流程：对每种细胞类型，用全部细胞构建 target pseudo-bulk map，然后做 100 次重复试验，每次随机抽 30 个细胞，将插补后的细胞聚合为 pseudo-bulk map，计算与 target 的 PCC。同时用 SuperTAD 识别 TAD 边界。

支持多种插补方法并行评估，每种方法的结果独立存放在以方法名命名的子目录中。

## 目录结构

```
lee_SuperTAD_pileline/
├── README.md
├── config.py                    # 共享配置
├── prepare_data.py              # Stage 1: 数据准备（已执行）
├── run_trials.py                # Stage 2: 采样 + PCC（需 --method）
├── run_supertad.py              # Stage 3: SuperTAD（需 --method）
├── plot_results.py              # Stage 4: 绘图（需 --method）
├── bin/SuperTAD                 # SuperTAD 二进制 (arm64)
├── input_lee/                   # 待插补原始数据（共享）
│   ├── per_cell_npz/            # 2098 NPZ (49×49)
│   ├── per_cell_bedpe/          # 2098 bedpe
│   └── metadata.json
├── target/                      # Target 矩阵（共享）
│   ├── {CellType}_target.npz
│   └── {CellType}_target.txt
├── imputed/                     # 插补结果（按方法名分子目录）
│   ├── scHiC-Diff/
│   │   └── {CellType}_cell_{idx:04d}.npz
│   └── NewMethod/
│       └── {CellType}_cell_{idx:04d}.npz
├── trials/                      # 试验结果（按方法名分子目录）
│   └── {method}/
│       ├── pcc_results.csv
│       └── matrices/
├── supertad/                    # SuperTAD 结果
│   ├── target/                  # Target TAD（共享）
│   └── {method}/                # 各方法 Trial TAD
│       ├── trials/
│       └── summary.json
└── figures/                     # 图表（按方法名分子目录）
    └── {method}/
        ├── pcc_distribution.png
        ├── heatmap_comparison.png
        ├── tad_overlay_target.png
        └── tad_counts.png
```

## 数据格式

### input_lee/per_cell_npz/（插补输入，共享）

每个文件是一个细胞的 49×49 接触矩阵（scipy sparse CSR, NPZ）。

- 文件名: `{CellType}_cell_{idx:04d}.npz`
- 细胞类型: Astro (449), Endo (202), ODC (1244), OPC (203)
- 矩阵: 49×49, 对称, 上三角 k=1 无对角线
- 区域: chr4:54,890,000-55,380,000 (10kb, 49 bins)
- PDGFRA 子区域: bins 20-28 (8×8)

### imputed/{method}/（插补方法需生成）

插补方法对 `input_lee/per_cell_npz/` 中每个细胞插补，结果保存到 `imputed/{方法名}/`。

**格式要求:**
- 文件名: `{CellType}_cell_{idx:04d}.npz`（与 input_lee 一一对应）
- 格式: scipy sparse CSR, NPZ
- 矩阵: 49×49, 对称（M + M.T）
- 值: float, count space
- 对角线: 可有可无（PCC 计算时置零）

```python
from scipy.sparse import load_npz, save_npz, csr_matrix
import numpy as np

raw = load_npz("input_lee/per_cell_npz/Astro_cell_0000.npz").toarray()
imputed = your_method(raw)        # 你的插补方法
imputed = imputed + imputed.T     # 确保对称
save_npz("imputed/YourMethod/Astro_cell_0000.npz", csr_matrix(imputed))
```

### target/（已准备，共享）

从 per-cell .cool 文件聚合的 target pseudo-bulk map。

## 执行流程

### 前提

- Python 3.8+, 安装 numpy scipy pandas matplotlib seaborn
- `bin/SuperTAD` 已包含 (arm64 macOS; Linux 需重新编译)
- `input_lee/` 和 `target/` 已就绪

### 步骤 1: 运行插补

将插补结果保存到 `imputed/{方法名}/`:

```bash
mkdir -p imputed/YourMethod
# 运行你的插补方法，输出到 imputed/YourMethod/
# 文件名: {CellType}_cell_{idx:04d}.npz
```

验证:
```bash
for ct in Astro Endo ODC OPC; do
    echo "$ct: $(ls imputed/YourMethod/${ct}_cell_*.npz 2>/dev/null | wc -l) files"
done
```

### 步骤 2-4: 评价 + 绘图

```bash
python run_trials.py --method YourMethod      # 100次采样 + PCC
python run_supertad.py --method YourMethod    # SuperTAD TAD边界
python plot_results.py --method YourMethod    # 生成图表
```

输出到 `trials/YourMethod/`, `supertad/YourMethod/`, `figures/YourMethod/`。

### 多方法对比

对每种方法重复上述步骤，结果分别存放在各自子目录中:

```bash
# 方法1
python run_trials.py --method scHiC-Diff
python run_supertad.py --method scHiC-Diff
python plot_results.py --method scHiC-Diff

# 方法2
python run_trials.py --method HiCImpute
python run_supertad.py --method HiCImpute
python plot_results.py --method HiCImpute
```

图表分别在 `figures/scHiC-Diff/` 和 `figures/HiCImpute/` 下。

## 关键参数 (config.py)

| 参数 | 值 | 说明 |
|------|-----|------|
| N_TRIALS | 100 | 重复试验次数 |
| N_SAMPLE | 30 | 每次采样细胞数 |
| BASE_SEED | 42 | 随机种子 (trial_id 用 seed=42+id) |
| N_BINS | 49 | 矩阵维度 |
| RESOLUTION | 10000 | 10kb |
| PDGFRA_SUB_BINS | (20, 28) | PDGFRA 8×8 子区域 |
| SUPERTAD_HEIGHT | 3 | SuperTAD multi 层级 |

## 细胞类型

| 类型 | 细胞数 | 论文细胞数 |
|------|--------|----------|
| Astro | 449 | 449 |
| Endo | 202 | 205 |
| ODC | 1244 | 1245 |
| OPC | 203 | 203 |

## 注意事项

1. `imputed/{方法名}/` 中文件名必须与 `input_lee/per_cell_npz/` 一一对应
2. 插补结果在 count space（非 log 变换）
3. 矩阵必须对称
4. SuperTAD 二进制是 arm64 macOS；Linux 需重新编译并替换 `bin/SuperTAD`
5. `--method` 参数为必选，指定插补方法名称
