# Dropout recovery and SZ/DO supplementary diagnostics

This directory produces the revised simulated-data figures. The main-text
figure contains only dropout-position PCC and MAE for HiCImputeData and
FLAMINGOData. Structural-zero (SZ) versus dropout (DO) results are retained
only as a HiCImputeData supplementary diagnostic, because real scHi-C data do
not provide observable SZ/DO labels and FLAMINGOData has no `GT == 0` class.

All commands are run from `paperplots/3_dropoutBar_confuseMap/`.

## Evaluation scope

The continuous and classification diagnostics share GT, observed matrices,
imputed outputs, cells, feature order, and raw contact scale, but answer
different questions:

```text
分类全集  = observed == 0
heldout DO = observed == 0 and GT > 0
true SZ     = observed == 0 and GT == 0

heldout DO is a strict subset of the classification universe
```

- The main-text PCC/MAE bars evaluate continuous recovery only at true DO,
  namely held-out positive contacts.
- F1, confusion matrices, ROC, and PR curves are supplementary-only SZ/DO
  diagnostics on all observed-zero HiCImputeData contacts.

不把 PCC/MAE 扩展到全部 observed-zero contacts。否则大量 true SZ 会主导 MAE，
并使 PCC 同时混合数值恢复和类别区分两个问题。

## 正式代码

```text
nature_plot_style.py
plot_dropout_pcc_mae_bar.py
calculate_sz_do_crossfit_metrics.py
plot_sz_do_metric_lines.py
plot_sz_do_confusion_matrices.py
plot_main_dropout_figure.py
calculate_sz_do_roc_pr.py
plot_sz_do_roc_pr_curves.py
submit_hicimputed_dropout_bar.sbatch
submit_flamingo_dropout_bar.sbatch
submit_hicimputed_sz_do_crossfit.sbatch
submit_hicimputed_sz_do_roc_pr.sbatch
submit_main_dropout_figure.sbatch
```

`plot_dropout_pcc_mae_bar.py` 读取 `../1_pccAndMae_all/` 的权威 raw-scale
PCC/MAE/SCC 汇总表，不在绘图阶段重新计算指标。

`calculate_sz_do_crossfit_metrics.py` recalculates the HiCImputeData SZ/DO
operating-point metrics from the current formal imputed matrices. The protocol
is fixed as follows:

- 候选位置仅为 `observed == 0`，observed positives 不参与分类。
- SZ 是正类；`prediction < threshold` 为 predicted SZ。
- 阈值最大化 MCC；并列时先最大化 balanced accuracy，再选择较小阈值。
- 阈值只应用于未参与选择的留出折，五折 OOF 预测合并后计算正式指标。

- 按 T1/T2/T3 各 100 cells 分层；每折留出每类 20 cells。
- 校准时合并其余 240 cells，并为每个 `method x depth x fold` 选择阈值。

scHiC-Diff itself emits continuous imputed contact values and has no
structural-zero classifier. The rule `prediction < threshold => predicted SZ`
is a post-processing diagnostic. Thresholds are method- and depth-specific,
not a universal value such as 0.5.

`calculate_sz_do_roc_pr.py` scans every distinct imputed value from the
pooled five-fold held-out score partitions for every method, cell type, and
depth. It writes the exact threshold scan, ROC-AUC, and PR-AUC (average
precision). The ranking score is `-imputed_contact`, so ROC/PR comparison is
not affected by different monotonic output scales across methods.

## 运行

连续值 PCC/MAE 柱状图：

```bash
sbatch submit_hicimputed_dropout_bar.sbatch
sbatch submit_flamingo_dropout_bar.sbatch
sbatch submit_main_dropout_figure.sbatch
```

HiCImputeData supplementary SZ/DO operating-point metrics, full ROC/PR scans,
and supporting figures:

```bash
sbatch submit_hicimputed_sz_do_crossfit.sbatch
sbatch submit_hicimputed_sz_do_roc_pr.sbatch
```

仅重算指标或仅重绘：

```bash
MODE=metrics sbatch submit_hicimputed_sz_do_crossfit.sbatch
MODE=curves sbatch submit_hicimputed_sz_do_crossfit.sbatch
MODE=plots sbatch submit_hicimputed_sz_do_crossfit.sbatch
MODE=main sbatch submit_hicimputed_sz_do_crossfit.sbatch
```

## 正式输出

```text
1_HiCImputedData/
├── HiCImputeData_SZ_DO_cell_folds.tsv
├── HiCImputeData_SZ_DO_5fold_thresholds.tsv
├── HiCImputeData_SZ_DO_5fold_fold_metrics.tsv
├── HiCImputeData_SZ_DO_5fold_OOF_metrics.tsv
├── HiCImputeData_SZ_DO_5fold_OOF_threshold_scan.tsv.gz
├── HiCImputeData_SZ_DO_5fold_OOF_curve_plot_points.tsv
├── HiCImputeData_SZ_DO_5fold_OOF_ROC_PR_AUC.tsv
├── figures/dropout_bar/
├── figures/sz_do_roc_pr/
├── figures/sz_do_metric_lines/
├── figures/sz_do_confusion_matrix/
└── logs/

2_FLAMINGOData/
├── FLAMINGOData_SZ_DO_eligibility_audit.tsv
├── figures/dropout_bar/
└── logs/

figures/main_dropout_figure/
└── Simulated_dropout_PCC_MAE_main_figure.pdf/png
```

四张五折表分别包含 300 条 cell-fold 分配、140 条校准阈值、420 条留出折指标和
84 条合并 OOF 指标。折线图直接读取 84 条 OOF 指标。混淆矩阵作为补充图，按
cell type 合并为 T1、T2、T3 三张图；每张图四行对应 1K/2K/4K/7K，七列对应
七种插补方法。

矩阵行依次为 true SZ、true DO，列依次为 predicted SZ、predicted DO，并按
真实类别逐行归一化。左上角等于 SZ recall，右下角等于 DO recall；后者在以 SZ
为正类时等价于 specificity（真负率）。图中 SZ F1 与折线图相同条件下的 SZ F1
必须一致。

### Supplementary operating-point metrics and confusion matrices

折线图和混淆矩阵不存在评价口径差异。两者均直接读取
`1_HiCImputedData/HiCImputeData_SZ_DO_5fold_OOF_metrics.tsv`，使用相同的五折
cell-wise OOF 预测和相同阈值。对于以 SZ 为正类的行归一化混淆矩阵：

```text
                     predicted SZ       predicted DO
true SZ              SZ recall          1 - SZ recall
true DO              1 - DO recall      DO recall (= specificity)
```

因此混淆矩阵左上角就是折线图中的 SZ recall，右下角是 DO recall（specificity）。
SZ precision 定义为 `TP / (TP + FP)`，不能直接由行归一化矩阵中的单个格子读取；
SZ F1 则是 SZ precision 和 SZ recall 的调和平均。某个方法可以通过把大量
observed-zero contacts 都判为 SZ 获得很高的 SZ recall，但同时产生大量
false-positive SZ，导致 SZ precision、DO recall、SZ F1 和 MCC 较低。单独比较
SZ recall 不能代表总体分类能力。

scHiC-Diff 在 12 个 `cell type x sequencing depth` 条件上的宏平均结果为：

```text
SZ precision                  = 0.994
SZ recall                     = 0.899
SZ F1                         = 0.944
DO recall (= specificity)     = 0.993
MCC                = 0.906
Balanced accuracy  = 0.946
```

这些结果表明 scHiC-Diff 使用了较保守但更均衡的判定：它识别约 90% 的 true SZ，
同时几乎不把 true DO 错判为 SZ。其 SZ recall 不是最高，但宏平均 SZ F1、
SZ precision、DO recall、MCC 和 balanced accuracy 均表现出更好的总体 SZ/DO
区分能力。

例如 T1-1K 条件下：

```text
scHiC-Diff       SZ precision=1.000  SZ recall=0.890  SZ F1=0.942  DO recall=1.000
HiCImpute        SZ precision=0.176  SZ recall=1.000  SZ F1=0.300  DO recall=0.188
Tensor-FLAMINGO  SZ precision=0.153  SZ recall=0.984  SZ F1=0.265  DO recall=0.055
```

HiCImpute 和 Tensor-FLAMINGO 在该条件下接近 1 的 SZ recall 主要来自过度预测 SZ；
对应混淆矩阵中分别有 81% 和 94% 的 true DO 被错判为 SZ。scHiC-Diff 的矩阵为
`[[0.89, 0.11], [0.00, 1.00]]`，说明其优势来自同时控制 false-negative SZ 和
false-positive SZ，而不是单独最大化 SZ recall。

scHiC-Diff 并非在每个单独条件下均为最优。例如 T1-4K 中 HiCImpute 的
`SZ F1=0.984`，高于 scHiC-Diff 的 `SZ F1=0.943`。因此论文中应表述为“总体上取得最佳
或最稳健的 SZ precision-recall/SZ-DO 权衡”，不能表述为“所有条件下均最优”。图中
scHiC-Diff 的红色边框和粗体只用于标识本文方法，不属于性能证据。

### Revised manuscript placement and captions

`plot_main_dropout_figure.py` creates the main-text figure at
`figures/main_dropout_figure/Simulated_dropout_PCC_MAE_main_figure.pdf`.
Panel A is HiCImputeData and Panel B is FLAMINGOData; both panels show only
dropout-position PCC and MAE. The figure is natively rendered in Matplotlib,
uses a shared seven-method legend, and contains no SZ/DO classification panel.

> **Figure X | Dropout recovery on two simulated scHi-C benchmarks.** PCC
> (higher is better) and MAE (lower is better) are calculated only at true
> dropout contacts (`observed = 0` and `GT > 0`) on the raw contact scale.
> Panel A shows HiCImputeData across cell types and sequencing depths; Panel B
> shows FLAMINGOData across the W and P simulation sweeps. Bars and error bars
> show cell-wise means and standard deviations.

The SZ/DO figures, including the fixed-MCC operating-point lines, confusion
matrices, full ROC/PR curves, and ROC-AUC/AP heatmap, belong in the
Supplementary Materials. Recommended methods text and reviewer-response
language are in `STRUCTURAL_ZERO_REVISION.md`.

## 为什么只有 HiCImputeData 能绘制 SZ/DO

下采样后出现零不等于存在 structural zero。两类标签必须同时根据观测矩阵和
同一位置的 GT 定义：

```text
candidate = observed == 0
true SZ   = observed == 0 and GT == 0
true DO   = observed == 0 and GT > 0
```

### HiCImputeData：GT 中显式构造了 structural zeros

HiCImputeData 使用 HiCImpute 包的 `scHiC_simulate()` 从三维坐标生成数据。模拟器
先由 bead pair 距离和协变量计算期望 contact 强度 `lambda`，然后执行以下步骤：

1. 取 `lambda` 低于 `gamma` 分位数的位置作为低强度候选；本项目使用
   `gamma=0.2`。
2. 从这些候选中随机选择一半作为 true-zero 候选，并把相应 GT contact 明确设为
   0。
3. `eta=0.8` 使其中大部分 structural zeros 在所有 cells 间共享，其余位置按 cell
   随机决定是否保持为零。
4. 对该含零的 `truecount` 进行 Poisson 抽样，再按测序深度下采样得到 observed。
   下采样会保留原有 true zeros，同时把部分 `GT > 0` contact 变成新的观测零。

因此 HiCImputeData 的 observed-zero 候选中同时存在两类：原本在 GT 中就是零的
SZ，以及因 Poisson 抽样或下采样丢失的 DO。例如 `K562_T1_1k` 共包含：

```text
GT entries               = 183000
GT == 0 (true SZ)         = 16509
observed-zero candidates  = 111495
true DO (GT > 0)          = 94986
```

本地模拟调用和测序深度下采样记录位于
`../../../1_Dataset/1-HiCImpute_Simulation_Data/Documents/Double check the simulated K562 data in HiCImpute.R`；
true-zero 构造见 HiCImpute 官方源码 `R/scHiC_simulate.R`。

### FLAMINGOData：GT 全部为正，只能产生 DO

FLAMINGOData 虽然也经过下采样，但其 per-cell GT 是由每个非对角 bead pair 的正
距离通过 `log1p(d^(-1/alpha))` 转换得到的稠密正 contact 矩阵。有限正距离经过该
变换后始终大于 0，因此 GT 中没有可作为 structural zero 的位置。随后执行的
Poisson thinning 只把一部分正 contact 从 observed 中删除：这些位置满足
`observed == 0 and GT > 0`，全部是 DO，而不是 SZ。

H5AD 中 `X` 和 `layers['counts']` 保存下采样后的稀疏 observed，`layers['gt']` 保存
未下采样的 per-cell GT。七个正式 H5AD 的形状均为 `1500 x 124750`，且
`layers['gt']` 均保存完整的 `187125000` 个非零 entries。W=0.5 数据的精确审计为：

```text
candidate contacts = 186186121
true SZ (GT == 0)  = 0
true DO (GT > 0)   = 186186121
```

七个条件的完整计数保存在
`2_FLAMINGOData/FLAMINGOData_SZ_DO_eligibility_audit.tsv`。

`heldout_masked` H5AD 额外遮蔽的是原 observed 中的正 contact，`layers['gt']` 仍为
正值，因此这些 heldout 位置也属于 DO，不会产生 SZ。

所以 FLAMINGOData 在当前 `GT == 0` 定义下只有一个真实类别，F1、Precision、
Recall、MCC 和 2 x 2 confusion matrix 均不能作为 SZ/DO 二分类指标计算或绘制。
FLAMINGOData 只保留 true-DO 位置上的 PCC/MAE 连续值恢复评价。不能复用
HiCImputeData 流程强行绘图。

相关生成和 H5AD 封装代码位于：

- `../../../1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/1_RawData/3_fixed_flamnigoGen/2_generate_v2_4090_style.py`
- `../../../1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/prepare_hierarchical_h5ad.py`

如需对 FLAMINGOData 开展分类评价，必须先由模拟器提供独立的 binary
structural-zero ground-truth mask；或者预先定义并论证 `GT <= tau` 的
effective-SZ 标签。后者属于新的低 contact 分类任务，不能与 HiCImputeData 的
`GT == 0` SZ/DO 结果合并解释。
