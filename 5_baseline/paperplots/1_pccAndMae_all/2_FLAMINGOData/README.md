# FLAMINGOData 插补指标

本目录保存 FLAMINGO 参数扫描数据的正式汇总指标和逐细胞指标。计算入口位于
上级目录的 `calculate_imputation_metrics.py`。

## 文件

```text
2_FLAMINGOData/
├── FLAMINGOData_PCC_MAE_SCC_metrics.csv
├── per_cell_metrics/
└── README.md
```

- `FLAMINGOData_PCC_MAE_SCC_metrics.csv`：七种方法在七个参数条件下的
  PCC、MAE 和 SCC 汇总，共 49 条记录。
- `per_cell_metrics/`：49 个 JSON；每个 JSON 保存一个方法和数据条件下
  1,500 个细胞的全部逐细胞指标。

## 数值处理

每个细胞包含 500 个 bead 的 124,750 个非对角特征。PCC、MAE 和 SCC 均在
保存的原始尺度上直接计算：GT 使用 H5AD 的 `layers['gt']`，prediction 使用
各方法的原始插补输出。不做 `log`、`log1p`、截负值或按细胞归一化；因此结果与
HiCImputeData 的评价尺度一致。最终 CSV 的 `transform` 字段应为 `raw`。

observed 和 heldout mask 同样由原始数值定义：`observed > 0` 与
`GT > 0 and observed <= 0`。这套指标处理不等同于 contact-map 绘图的面板显示
归一化。

## 特征顺序和数据版本

评价空间统一为 NumPy row-major `triu(k=1)`：

- GT、observed 和 scHiC-Diff 的七个条件（包括 W=0.7）统一使用
  `5_paramsweep_datasets`；不使用同名的 `3_500cells_datasets` 副本，以确保
  指标与 7x9 heatmap 的 Input/GT 来源一致。
- scVI-3D 使用 `v3_outputData_earlystop_bs1500/npz_upper_tri` 的早停版本。
- HiCImpute 使用已修正 R column-major 排列的 `npz_triu_corrected`。
- scHiCluster 和 Higashi 的历史目录名虽然含 `lower_tri`，实际特征顺序是
  canonical `triu(k=1)`。
- Tensor-FLAMINGO 是唯一特殊编码：canonical triu 特征序列存储在 legacy
  tensor 的 tril 坐标，读取时从 tril 坐标提取并按 canonical triu 语义评价。
- scHiC-Diff 使用与 FLAMINGO H5AD 特征一致的 `triu(k=1)`，并且正式评价
  `denoise_recon_inv.npz`，即逆变换后的插补结果。

原先基于 `log1p(max(value, 0))` 的历史汇总不再是正式结果，不能与本表混用。
