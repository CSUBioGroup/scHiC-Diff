# HiCImputeData dropout recovery and supplementary SZ/DO diagnostics

本目录只保存 HiCImputeData 的正式指标表、图片和 Slurm 日志。代码和提交入口
统一位于上级 `3_dropoutBar_confuseMap/` 根目录。

## PCC/MAE 柱状图

柱状图直接读取：

```text
../../1_pccAndMae_all/1_HiCImputedData/HiCImputeData_PCC_MAE_SCC_metrics.csv
```

该权威表包含七种方法在 `T1/T2/T3 x 1k/2k/4k/7k` 共 12 个条件下的
84 条 raw-contact 指标。正式柱状图使用：

```text
heldout DO = observed == 0 and GT > 0
PCC        = pcc_held_mean +/- pcc_held_std
MAE        = mae_held_mean +/- mae_held_std
```

绘图不会重新计算 PCC 或 MAE，也不会使用 SZ/DO 分类阈值。

## Supplementary five-fold SZ/DO diagnostic

分类候选范围为全部 `observed == 0` 位置：

```text
true SZ = observed == 0 and GT == 0
true DO = observed == 0 and GT > 0
```

使用 5 折 cell-wise cross-fitting。每折在其余 cells 上为每个
`method x sequencing depth` 选择最大 MCC 阈值，T1/T2/T3 共享该阈值；留出
fold 的 GT 不参与阈值选择。SZ 是正类，五折留出预测合并成 OOF 结果。

正式表：

```text
HiCImputeData_SZ_DO_cell_folds.tsv              300 条 cell-fold 分配
HiCImputeData_SZ_DO_5fold_thresholds.tsv        140 条校准阈值
HiCImputeData_SZ_DO_5fold_fold_metrics.tsv      420 条留出折指标
HiCImputeData_SZ_DO_5fold_OOF_metrics.tsv        84 条合并 OOF 指标
```

The fixed-MCC F1, precision, recall, specificity, and confusion matrices are
supplementary-only. scHiC-Diff emits continuous values; SZ calling is the
external rule `prediction < threshold`. The threshold is calibrated separately
for every method, depth, and fold, not fixed at 0.5.

`calculate_sz_do_roc_pr.py` additionally scans every unique prediction value
on the five-fold score partitions and writes complete ROC/PR threshold scans,
ROC-AUC, and PR-AUC (average precision). The compressed exact scan is
`HiCImputeData_SZ_DO_5fold_OOF_threshold_scan.tsv.gz`.

## 运行

从上级 `3_dropoutBar_confuseMap/` 提交：

```bash
sbatch submit_hicimputed_dropout_bar.sbatch
sbatch submit_hicimputed_sz_do_crossfit.sbatch
sbatch submit_hicimputed_sz_do_roc_pr.sbatch
```

只重绘已经存在的五折结果：

```bash
MODE=plots sbatch submit_hicimputed_sz_do_crossfit.sbatch
MODE=curves sbatch submit_hicimputed_sz_do_crossfit.sbatch
```

## 图片与日志

```text
figures/dropout_bar/             heldout DO 的 PCC/MAE 柱状图
figures/sz_do_metric_lines/      OOF F1/Precision/Recall/Specificity 折线图
figures/sz_do_confusion_matrix/  T1/T2/T3 三张 4-depth x 7-method 补充图
figures/sz_do_roc_pr/            full-scan ROC/PR and AUC supplementary figures
logs/                            Slurm 标准输出和错误日志
```

The revised main-text figure is shared across both simulated families and is
written to `../figures/main_dropout_figure/`. It contains dropout PCC/MAE only.

Tensor-FLAMINGO 使用共享配置中的正式 `output_distance_best/contact_from_pd`
结果。所有方法均使用 `1_pccAndMae_all/imputation_metric_config.py` 解析的当前
正式插补路径。
