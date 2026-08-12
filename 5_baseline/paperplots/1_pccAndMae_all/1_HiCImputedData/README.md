# HiCImputeData 插补指标

本目录保存 HiCImputeData 的正式汇总指标和逐细胞指标。计算入口位于上级
目录的 `calculate_imputation_metrics.py`。

## 文件

```text
1_HiCImputedData/
├── HiCImputeData_PCC_MAE_SCC_metrics.csv
├── per_cell_metrics/
└── README.md
```

- `HiCImputeData_PCC_MAE_SCC_metrics.csv`：七种方法在 12 个数据条件下的
  PCC、MAE 和 SCC 汇总，共 84 条记录。
- `per_cell_metrics/`：84 个 JSON；每个 JSON 保存一个方法和数据条件下
  100 个细胞的全部逐细胞指标。

## 数值和特征顺序

GT、observed 和七种插补输出均按 61 个 bead 的非对角三角特征表示，每个
细胞 1,830 个特征。当前 HiCImputeData 输入采用 NumPy row-major
`tril(k=-1)` 顺序；计算使用原始 contact 数值，不做 log、log1p、总接触量
归一化或绘图色阶归一化。

最终 CSV 的主键为 `method + data_name`，其中 `data_name` 覆盖
`K562_T1/T2/T3_1k/2k/4k/7k`。`transform` 字段应为 `raw`。

Tensor-FLAMINGO 正式采用 `selection=best` 的 contact-space 输出：

```text
9_FLAMINGO/1_HiCImputeData/output_distance_best/contact_from_pd/npz_lower_tri/
```

旧 `output_distance/contact_from_pd/npz_lower_tri/` 在 11/12 个条件的 heldout
区域产生零方差预测，导致 PCC/SCC 无法计算，因此不再用于正式指标、热图或
SZ/DO 分析。当前 CSV 同时包含 all、observed、heldout 的 PCC、MAE 和 SCC，
是 HiCImputeData 的唯一正式汇总表。
