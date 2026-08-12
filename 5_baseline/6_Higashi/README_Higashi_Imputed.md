# Higashi 插补复现与下游使用手册

本文档同时覆盖 `Higashi_nbr0` 和 `Higashi_nbr5`。两者属于同一方法、共享输入与基础训练流程，不分别建立 README。下游将它们作为两个基准变体比较。

## 1. 当前基准结果

| 数据族 | 当前结果模板 | shape | 特征顺序 |
|---|---|---:|---|
| HiCImputeData nbr0 | `/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/1_HiCImputeData/output/npz_lower_tri/{data_name}_higashi_nbr_0_lower_tri.npz` | `(100,1830)` | NumPy `tril(k=-1)` |
| HiCImputeData nbr5 | 同目录 `{data_name}_higashi_nbr_5_lower_tri.npz` | `(100,1830)` | NumPy `tril(k=-1)` |
| FLAMINGOData nbr0 | `/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/2_FLAMINGOData/v3_epoch1000_outputData/npz_lower_tri/{data_name}_higashi_nbr_0_lower_tri.npz` | `(1500,124750)` | NumPy `triu(k=1)` |
| FLAMINGOData nbr5 | 同目录 `{data_name}_higashi_nbr_5_lower_tri.npz` | `(1500,124750)` | NumPy `triu(k=1)` |

FLAMINGOData 的文件名含 `lower_tri`，但 collector 实际抽取 `triu`。以 manifest 和 collector 代码为准。

## 2. nbr0 与 nbr5 的关系

| 项目 | `nbr0` | `nbr5` |
|---|---|---|
| `neighbor_num` | 0 | 5 |
| 基础 embedding | 需要 | 需要 |
| no-neighbor imputation | 训练并输出 | 先训练并通常也产生中间 nbr0 输出 |
| neighbor-enhanced stage | 不运行 | 额外运行 `train_for_imputation_with_nbr()` 和 `impute_with_nbr()` |
| 最终 HDF5 名 | `*_nbr_0_impute.hdf5` | `*_nbr_5_impute.hdf5` |
| 适用性 | 不从邻近细胞借信息 | 用 5 个 embedding 邻居平滑，可能提高稳定性，也可能混合亚群 |

不要把 `nbr5` 理解为只在 `nbr0` 输出上做简单平均。它包含额外的 neighbor-aware 训练与插补阶段。对新数据建议两者都运行，用独立 GT/held-out 指标选择；不能预设 `nbr5` 一定更好。

## 3. 方法流程

本项目采用 classic Higashi `higashi_v1` 输入：

```text
observed cells × triangle features
  -> 过滤正 contact 和最小基因组距离
  -> data.npy: [cell_id, chrom_id, bin1, bin2]
  -> weight.npy: contact value
  -> config.JSON + chrom sizes + labels
  -> create_matrix / generate_chrom_start_end
  -> prep_model
  -> train_for_embeddings
  -> train_for_imputation_nbr_0 / impute_no_nbr
  -> 可选 train_for_imputation_with_nbr / impute_with_nbr
  -> HDF5
  -> 按目标数据族的 triangle order 收集 NPZ
```

最终 NPZ 行顺序与输入 cell ID 的 0-based 顺序一致。

## 4. 环境与 HPC

主要环境：

```text
Higashi GPU: /public/home/hpc254701055/micromamba/envs/6_higashi/bin/python
HiCImputeData 历史任务: /public/home/hpc254701055/micromamba/envs/hic-impute/bin/python
FLAMINGO prepare/collect: /public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python
```

训练必须使用 GPU Slurm 队列。当前资源范围为 1 GPU、4-20 CPU、64-128G、最长 7 天；FLAMINGO `updates=1000` 使用 `gpu4Q/gpuq`、10 CPU、120G。prepare、collect、评估也不能在登录节点加载全量矩阵。

## 5. 输入契约

### 5.1 共同文件

每个 `{dataset}_nbrK/` 目录包含：

```text
config.JSON
data/<chrom>.chrom.sizes
data/label_info.pickle
temp/data.npy
temp/weight.npy
temp/chrom_start_end.npy
```

`data.npy` 的 cell ID 从 0 开始；`weight.npy` 与其逐行对应。contact 必须有限且大于 0，`bin1 < bin2`。

### 5.2 HiCImputeData

- 数据根：`/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/3_DiffusionModel/scHiC-Diff-master/data/SimuData/1_HiCImputeData`
- `sim/*.h5ad` 是模型输入；`gt/*.npz` 仅用于验证/评估。
- 61 bins、100 cells、1830 features，输入向量为 canonical `tril`。
- `min_delta=1`，相邻 bin 可进入训练。
- `vector_to_higashi_rows()` 将 tril 坐标统一成 `bin1=min(i,j)`、`bin2=max(i,j)`。

### 5.3 FLAMINGOData

- 来源七个 h5ad 的 `layers['counts']`，GT 在 `layers['gt']`。
- 500 bins、1500 cells、124750 features，h5ad feature 是 NumPy row-major `triu`。
- 当前 `min_delta=2`，排除主对角线和紧邻 bin。
- cell 顺序保持 h5ad 行顺序；应为 T1 1..500、T2 1..500、T3 1..500。

## 6. 当前 config 超参数

| 参数 | 当前值 |
|---|---:|
| `dimensions` | 64 |
| `loss_mode` | `zinb` |
| `resolution` / `resolution_cell` | 1,000,000 |
| `local_transfer_range` | 1 |
| `embedding_epoch` | 60 |
| `no_nbr_epoch` | 45 |
| `with_nbr_epoch` | 30 |
| `correct_be_impute` | false |
| `precompute_weighted_nbr` | true |
| `structured` | false |
| `cpu_num_torch` | 最多 4 |

Higashi 的 chrom size 当前写成 `n_bins * resolution`，并通过 `chrom_start_end.npy=[[0,n_bins]]` 限定 bin 区间。迁移时要一起验证，不能只改 JSON 的 resolution。

### 6.1 `training-updates` 不是 epoch

runner 会 patch：

```text
update_num_per_training_epoch
update_num_per_eval_epoch
```

当前设置：

| 数据族 | training updates/epoch | eval updates/epoch | config epochs |
|---|---:|---:|---|
| HiCImputeData | 200 | 10 | 60/45/30 |
| FLAMINGOData 当前 manifest | 1000 | 10 | 60/45/30 |
| FLAMINGOData 历史基线 | 200 | 10 | 60/45/30 |

`updates=1000` 不等于 `embedding_epoch=1000`。当前 FLAMINGO 图和统一指标使用独立的 `v3_epoch1000_*` 结果；不要与 `v3_outputData` 的 updates=200 文件混合。

## 7. runner 中的兼容补丁

`run_higashi_one.py` 对 classic Higashi 做了本项目必要的运行时 patch：

- feature generation 对过滤后空 feature block 做容错；
- 限制 TruncatedSVD component 数，避免小矩阵维度错误；
- patch 每个 epoch 的 train/eval update 数；
- HiCImputeData runner 还包含 dense toy chromosome 的 negative-sampling fallback。

因此不要绕过 runner 直接调用 Higashi 类。升级 Higashi 包后必须重新验证这些 monkey patch 的函数签名。

## 8. HiCImputeData 流程

### 8.1 新数据生成 manifest 和两个 config

当前输入已准备好时无需重复。新数据可在 CPU 节点运行：

```bash
BASE=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/1_HiCImputeData
NEW_DATA_ROOT=/absolute/path/to/new_data
NEW_INPUT_ROOT=/absolute/path/to/new_higashi_input
prep=$(sbatch --parsable -A pi_limin_r -p cpuQ -q cpuq -c 20 --mem=80G -t 04:00:00 \
  --export=ALL,BASE="${BASE}",NEW_DATA_ROOT="${NEW_DATA_ROOT}" \
  --wrap='/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python "${BASE}/scripts/hicimpute_higashi_pipeline.py" write-manifest --data-root "${NEW_DATA_ROOT}" --manifest "${BASE}/manifest_new.tsv"')
sbatch --dependency=afterok:${prep} -A pi_limin_r -p cpuQ -q cpuq -c 20 --mem=80G -t 04:00:00 \
  --export=ALL,BASE="${BASE}",NEW_INPUT_ROOT="${NEW_INPUT_ROOT}" \
  --wrap='/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python "${BASE}/scripts/hicimpute_higashi_pipeline.py" build-all --manifest "${BASE}/manifest_new.tsv" --input-root "${NEW_INPUT_ROOT}" --neighbors 0 5'
```

新实验应使用新的 manifest/input/output 根，避免覆盖当前 config 和 checkpoint。

### 8.2 训练和收集

当前主 array：

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/1_HiCImputeData
sbatch run_higashi_hicimpute_array.sbatch
```

每个 task 顺序运行 nbr0 和 nbr5，再调用 `hicimpute_higashi_pipeline.py convert-output`。部分历史慢任务使用 `run_higashi_hicimpute_slow_cpu20_gpu2q.sbatch` 或 `run_higashi_hicimpute_t3_4k7k_gpu4q_cpu10.sbatch` 恢复；这些是故障恢复入口，不应作为新数据默认流程。

## 9. FLAMINGOData updates=1000 流程

当前七条件被拆为两组 14 个 dataset×neighbor 任务：

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/2_FLAMINGOData
sbatch v3_scripts/submit_higashi_epoch1000_subset_gpu4q.sbatch
sbatch v3_scripts/submit_higashi_epoch1000_remaining_gpu4q.sbatch
```

- subset：W=0.5 和 W=0.7，各含 nbr0/nbr5。
- remaining：W=0.6、0.8、0.9、P=1%、P=5%，各含 nbr0/nbr5。
- config 不存在时，作业会先调用 `v3_prepare_higashi.py`。
- 每个 task 收集到 `v3_epoch1000_outputData/npz_lower_tri` 并生成独立 log1p metrics CSV。

历史 `submit_higashi.sbatch` 使用 updates=200 和 `v3_inputData/v3_outputData`，可用于对照，不是当前绘图结果来源。

## 10. HDF5 到 NPZ 的收集

HiCImputeData collector 读取 HDF5 后抽取：

```python
np.tril_indices(61, k=-1)
```

FLAMINGOData collector 读取 HDF5、对坐标存储补对称、截断负值和非有限值后抽取：

```python
np.triu_indices(500, k=1)
```

如果 HDF5 缺少某个 cell key，当前 collector 会留下全零行。正式使用前必须检查每个 cell 的 `nnz`，不能只检查整体 shape。

## 11. 输出验证

轻量检查数量：

```bash
find 1_HiCImputeData/output/npz_lower_tri -maxdepth 1 -name '*_higashi_nbr_*_lower_tri.npz' | wc -l
find 2_FLAMINGOData/v3_epoch1000_outputData/npz_lower_tri -maxdepth 1 -name '*_higashi_nbr_*_lower_tri.npz' | wc -l
```

预期分别为 24 和 14。再通过 CPU Slurm 检查：

- shape、有限值、非负性、`nnz`；
- 任何全零 cell 行；
- cell order 与 h5ad/GT；
- HiC 的 `tril` 和 FLAMINGO 的 `triu` 坐标 round-trip；
- nbr0/nbr5 文件确实来自对应 HDF5，而不是把 nbr0 误命名成 nbr5；
- config 中 `neighbor_num`、`impute_with_nbr` 与文件名一致。

## 12. PCC、MAE 与 held-out

统一计算位于 `paperplots/1_pccAndMae_all/recalc_all_metrics.py`。

- HiCImputeData：逐 cell 全特征 PCC/MAE；dropout=`(Raw==0)&(True!=0)`。
- FLAMINGOData：非负 `log1p` 后计算 GT-positive all、Raw-positive observed 和 `GT-positive & not observed` held-out。

统一 registry 当前指向 HiC `output/npz_lower_tri` 和 FLAMINGO `v3_epoch1000_outputData/npz_lower_tri`。更新结果后必须先更新 registry/manifest，再通过 Slurm 重算，不能沿用旧 CSV 给新图标注 PCC。

## 13. Contact map 绘图

当前 HiC 和 FLAMINGO 图都展示 cell index 0。提交入口：

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/2_imputedContactHeatmap
sbatch submit_render_all_hicimpute_method_heatmaps.sbatch
sbatch submit_render_all_hicimpute_method_heatmaps_gr.sbatch
sbatch nature-style-plot/submit_flamingo_heatmap_grid.sbatch
```

FLAMINGO grid 会将每张对称矩阵归一化到总 contact 10,000，并使用 GT 共享色标。若 nbr5 图比 nbr0 更模糊，可能是邻居混合的真实结果，也可能是 cell/feature 顺序错误；先做顺序验证再解释生物学差异。

## 14. 新数据适配清单

1. 明确 cell ID、cell type、染色体、bin 和 feature pair 顺序。
2. 从 observed 数据生成 `data.npy/weight.npy`，GT 只能用于评估。
3. 根据分辨率决定 `min_delta`，并记录过滤掉的 contact 数量。
4. 为 nbr0/nbr5 分别生成 config，检查 `neighbor_num` 和 `impute_with_nbr`。
5. 先在 Slurm 上小规模验证 create_matrix、embedding、nbr0 和 nbr5 全流程。
6. 根据耗时/显存决定 training updates、CPU 和 GPU；不要混合不同 updates 的输出目录。
7. 收集后检查全零行、shape、三角顺序和 cell order。
8. 用 held-out/GT 指标选择 nbr 配置，再更新统一 manifest 和正式图片。

## 15. 常见故障

| 症状 | 原因 | 处理 |
|---|---|---|
| negative sampling 为空 | 模拟小染色体过密 | 必须通过项目 runner 使用 fallback |
| SVD 报 component 错误 | 过滤后 feature block 太小/空 | 使用 patched feature generation |
| nbr5 文件不存在 | config 未启用 `impute_with_nbr` | 检查 `neighbor_num=5` 和运行日志 |
| nbr5 task 很慢 | 额外 neighbor-aware 训练，CPU 配置不足 | 使用独立高资源 Slurm 作业，不要在登录节点恢复 |
| 输出 shape 正确但有全零 cell | HDF5 缺少 cell key | 检查 key 数并补跑对应任务 |
| FLAMINGO 图错位 | 按文件名误当成 tril | 按 `triu` 重建 |
| 误把 updates=1000 当 epoch | 参数概念混淆 | 同时记录 config epochs 与每 epoch update 数 |

## 16. 权威文件

- HiC 准备/转换：`6_Higashi/1_HiCImputeData/scripts/hicimpute_higashi_pipeline.py`
- HiC runner：`6_Higashi/1_HiCImputeData/scripts/run_higashi_one.py`
- FLAMINGO 准备/收集：`6_Higashi/2_FLAMINGOData/v3_scripts/v3_prepare_higashi.py`、`v3_collect_higashi.py`
- FLAMINGO current submit：`submit_higashi_epoch1000_subset_gpu4q.sbatch`、`submit_higashi_epoch1000_remaining_gpu4q.sbatch`
- 当前绘图输入：`paperplots/2_imputedContactHeatmap/hicimpute_heatmap_input_manifest.tsv` 和 `flamingo_heatmap_input_manifest.tsv`
