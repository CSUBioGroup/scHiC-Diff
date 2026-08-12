# FLAMINGOData dropout recovery 结果

本目录保存 FLAMINGOData 的正式连续值指标图片和 Slurm 日志。代码与提交入口
统一位于上级 `3_dropoutBar_confuseMap/` 根目录。

## PCC/MAE 柱状图

柱状图读取权威 raw-scale 指标：

```text
../../1_pccAndMae_all/2_FLAMINGOData/FLAMINGOData_PCC_MAE_SCC_metrics.csv
```

该表包含七种方法在 W sweep 和 P sweep 共七个唯一条件下的 49 条记录。PCC、
MAE 和 SCC 使用原始 GT 与原始插补值，不做 `log`、`log1p`、截负值或额外
归一化。柱状图使用 true DO 范围：

```text
heldout DO = observed == 0 and GT > 0
```

从上级目录运行：

```bash
sbatch submit_flamingo_dropout_bar.sbatch
```

结果保存到 `figures/dropout_bar/`，日志保存到 `logs/`。

## SZ/DO 分类不适用

逐条件审计表：

```text
FLAMINGOData_SZ_DO_eligibility_audit.tsv
```

FLAMINGOData 每个条件包含 T1/T2/T3 各 500 cells，但它的 per-cell GT 是由
3D bead pairs 的正距离通过 `log1p(d^(-1/alpha))` 生成的全稠密正 contact
矩阵。七个 H5AD 的 `layers['gt']` 均具有 `187125000 = 1500 x 124750` 个
非零 entries，没有可作为 structural zero 的 `GT == 0` 位置。

审计利用 H5AD CSR 元数据计算：`observed_zero = total_entries - observed_nnz`。
由于 `csr_matrix(gt_dense)` 不保存零值，而每个条件均满足
`gt_nnz == total_entries`，所以七个条件的 `true_sz_count` 都严格为 0。真实
pilot 对 W=0.5 全部 observed-zero contacts 的矩阵级计数也给出相同结果：

```text
candidate contacts = 186186121
true SZ (GT == 0)  = 0
true DO (GT > 0)   = 186186121
```

因此 F1、Precision、Recall、MCC 和 2 x 2 confusion matrix 在当前标签定义下
不可计算，不能生成与 HiCImputeData 同义的折线图或混淆矩阵。若要建立新的
分类任务，需要以下二者之一：

1. 模拟器提供独立的 binary structural-zero ground-truth mask。
2. 在评价前预先定义并论证 `GT <= tau` 为 effective SZ，且 tau 不能使用测试
   labels 调整。

第二种方案改变了 SZ 的定义，结果必须明确标注为 low-contact/effective-SZ
classification，不能与 `GT == 0` 的 HiCImputeData SZ/DO 指标直接合并解释。
