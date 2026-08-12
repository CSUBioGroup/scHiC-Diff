# Higashi FLAMINGO v3 Imputation Notes

本文档记录当前已经成功的 Higashi FLAMINGO v3 插补流程。后续用 Codex/opencode 迁移到新数据集时，优先按这里的输入数据契约、预处理规则、运行命令和验证清单执行。

## 1. 当前目录

当前 `2_FLAMINGOData` 只保留 v3 正确插补相关内容：

```text
2_FLAMINGOData/
  v3_inputData/      # Higashi 输入目录，每个 dataset x neighbor 一个目录
  v3_outputData/     # 统一收集后的 NPZ 和两套评估结果
  v3_scripts/        # v3 预处理、运行、收集、评估脚本和成功日志
```

当前成功结果：

- 7 个 FLAMINGO v3 paramsweep 数据集。
- 每个数据集运行 `neighbor=0` 和 `neighbor=5`，共 14 个任务。
- `v3_inputData/*/config.JSON` 共 14 个。
- `v3_outputData/npz_lower_tri/*_lower_tri.npz` 共 14 个，shape 应为 `(1500, 124750)`。
- `v3_scripts/logs/impute_20048981_*.log` 共 14 个成功日志，每个日志应包含 `[OK] Higashi run finished`。

另有独立的 `updates=1000` 超参实验，输出不覆盖上述 `updates=200` 基线：

```text
2_FLAMINGOData/
  v3_epoch1000_inputData/
  v3_epoch1000_outputData/
  v3_scripts/logs_epoch1000/
```

截至 2026-07-03 20:09 CST：

- 已完成并比较：`W0p5` 和 `W0p7` 的 `neighbor=0/5`，共 4 个任务，GPU array job `20064012`，比较 job `20064016`。
- 已提交剩余 5 个数据集的 `neighbor=0/5`，共 10 个任务，GPU array job `20064175`，全量比较 job `20064185` 依赖 `afterok:20064175`。
- 剩余任务使用 `gpu4Q`、`pi_limin_r`、`cpus-per-task=10`、`--training-updates 1000 --eval-updates 10`。
- 全量完成后，`v3_epoch1000_outputData/npz_lower_tri` 和 `v3_epoch1000_outputData/metrics/log1p_tasks` 都应各有 14 个文件。
- 全量对比输出将写入 `v3_epoch1000_outputData/metrics/higashi_epoch1000_all_vs_epoch200_summary.csv`。

## 2. 输入 h5ad 数据契约

当前脚本假设输入 h5ad 位于：

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets
```

文件命名：

```text
v3_hybrid_<tag>_scdiff2.h5ad
```

当前成功运行的数据集 stem：

```text
v3_hybrid_W0p5_500cells_level0
v3_hybrid_W0p6_500cells_level0
v3_hybrid_W0p7_500cells_level0
v3_hybrid_W0p7_500cells_level0_r0p01
v3_hybrid_W0p7_500cells_level0_r0p05
v3_hybrid_W0p8_500cells_level0
v3_hybrid_W0p9_500cells_level0
```

每个 h5ad 必须满足：

- `layers/counts`：观测到的 noisy contacts，作为 Higashi 输入。
- `layers/gt`：ground truth，只用于评估，不参与 Higashi 训练和插补。
- `obs/_index`：cell 索引。
- `var/_index`：feature 索引。
- 当前 v3 固定为 `N_BINS=500`，`N_FEATURES=124750`，单染色体名 `chrFLAMINGO`。

feature 顺序是最关键的约定：

- 每个 feature 表示 500 x 500 contact matrix 的上三角坐标 `(i, j)`，要求 `i < j`。
- 顺序为 `np.triu_indices(500, k=1)` 的 row-major 顺序。
- `var_names` 应匹配类似 `chrFLAMINGO_i_j` 的含义。
- `v3_prepare_higashi.py`、`v3_collect_higashi.py`、`v3_evaluate.py` 都依赖同一套 feature 顺序。更换数据集时不能只换 h5ad，而不检查 feature 顺序。

## 3. 关键脚本职责

### `v3_scripts/v3_common.py`

共享常量和 h5ad 读取逻辑：

- `N_BINS = 500`
- `N_FEATURES = 124750`
- `CHROM_NAME = "chrFLAMINGO"`
- `DEFAULT_DATA_DIR` 指向当前 FLAMINGO v3 h5ad 目录。
- `discover_datasets()` 自动发现 `v3_hybrid_*_scdiff2.h5ad`。
- `load_layer(path, "counts")` 从 h5ad CSR layer 读取输入；如果 layer 不存在会 fallback 到 `X`。
- `feature_to_bins(N_BINS)` 生成 feature index 到 `(i, j)` 的映射。

### `v3_scripts/v3_prepare_higashi.py`

把 h5ad `layers/counts` 转成 Higashi classic `higashi_v1` 输入格式。

每个 `stem` 和 `neighbor` 会生成：

```text
v3_inputData/<stem>_nbr<0|5>/
  config.JSON
  data/flamingo.chrom.sizes
  data/label_info.pickle
  temp/data.npy
  temp/weight.npy
  temp/chrom_start_end.npy
  temp/raw/
```

预处理规则：

- 只读取 `layers/counts`，不读取 `layers/gt`。
- 对每个 cell 的非零 contacts 做过滤：
  - contact 值 finite；
  - contact 值 `> 0`；
  - `j_idx - i_idx >= min_delta`，当前默认 `min_delta=2`。
- `data.npy` 每行是 `[cell_id, 0, i_idx, j_idx]`。
- `weight.npy` 是对应 contact count，`float32`。
- `chrom_start_end.npy` 是 `[[0, 500]]`。
- `flamingo.chrom.sizes` 写入 `chrFLAMINGO    500000000`，因为 resolution 是 `1,000,000`。

当前成功配置中的主要 Higashi 参数：

```text
embedding_name = flamingo_higashi
resolution = 1000000
dimensions = 64
loss_mode = zinb
cpu_num = 20
cpu_num_torch = min(cpu_num, 4)
gpu_num = 1
embedding_epoch = 60
no_nbr_epoch = 45
with_nbr_epoch = 30
impute_no_nbr = true
impute_with_nbr = true only when neighbor_num > 0
neighbor_num = 0 or 5
precompute_weighted_nbr = true
input_format = higashi_v1
chrom_list = ["chrFLAMINGO"]
impute_list = ["chrFLAMINGO"]
```

### `v3_scripts/run_higashi_one.py`

运行单个 `config.JSON` 的 Higashi 训练和插补。

它做了两个补丁：

- patch `higashi.Process.generate_feats_one`，使空 feature block 不会导致 SVD 报错。
- patch `higashi.Higashi_wrapper.Higashi.fetch_info_from_config`，覆盖每个 epoch 的训练和验证 update 数。

当前成功 sbatch 实际使用：

```text
--training-updates 200
--eval-updates 10
```

不要只看 `run_higashi_one.py` 里的 argparse 默认值；成功批任务以 `submit_higashi.sbatch` 中的参数为准。当前 `updates=200` 是已确认成功的基线配置；`updates=1000` 是后续独立超参实验，输入输出目录单独保存。

运行阶段：

1. `generate_chrom_start_end`
2. `create_matrix`
3. `prep_model`
4. `train_for_embeddings`
5. `train_for_imputation_nbr_0`
6. `impute_no_nbr`
7. 如果 `impute_with_nbr=true`，继续 `train_for_imputation_with_nbr` 和 `impute_with_nbr`

`neighbor=5` 的目录会同时产生 no-nbr 和 with-nbr hdf5；最终评估时使用文件名中对应 neighbor 的结果。

### `v3_scripts/v3_collect_higashi.py`

把 Higashi 的 hdf5 插补结果转换为统一评估用的 sparse CSR NPZ。

输入：

```text
v3_inputData/<stem>_nbrK/temp/chrFLAMINGO_flamingo_higashi_nbr_K_impute.hdf5
```

输出：

```text
v3_outputData/npz_lower_tri/<stem>_higashi_nbr_K_lower_tri.npz
```

转换规则：

- 每个 cell 转成 `(n_cells, n_features)`。
- feature 顺序仍为 `np.triu_indices(N_BINS, k=1)`。
- NaN、Inf、负值都置为 0。
- 当前 v3 输出 shape 必须是 `(1500, 124750)`。

### `v3_scripts/v3_evaluate.py`

计算 log1p 评估指标。

输出：

```text
v3_outputData/metrics/scHiCluster_FLAMINGO_v3_paramsweep_quality_metrics.csv
```

注意：这个文件名里有 `scHiCluster`，但当前文件中保存的是 Higashi v3 log1p 指标，这是历史命名遗留。

评估规则：

- GT 来自 h5ad `layers/gt`。
- observed mask 来自 h5ad `layers/counts > 0`。
- heldout mask 来自同一数据目录下的 `fixed_heldout_indices.npz`，数组名为 `heldout`，列为 `[cell_index, feature_index]`。
- 对 prediction 和 GT 都执行 `log1p(max(x, 0))` 后，计算 cell-wise PCC、Spearman、MAE。
- 分别统计 `all`、`observed`、`heldout` 三类 mask。

### `v3_scripts/13_cal_FLAMINGO_Baseline_metrics.py`

计算 raw GT 与 raw prediction 的 cell-wise PCC 和 MAE。

输出：

```text
v3_outputData/metrics/higashi_FLAMINGO_v3_paramsweep_raw_gt_PCC_MAE.csv
```

评估规则：

- GT 来自 h5ad `layers/gt`。
- prediction 来自 `v3_outputData/npz_lower_tri/*_lower_tri.npz`。
- 不做 log1p，直接在 raw scale 上计算 PCC 和 MAE。

因此当前有两套评估：

- raw GT vs raw prediction：`higashi_FLAMINGO_v3_paramsweep_raw_gt_PCC_MAE.csv`
- log1p GT vs log1p prediction：`scHiCluster_FLAMINGO_v3_paramsweep_quality_metrics.csv`

高 PCC 但 raw MAE 大于 2 并不必然说明插补失败。PCC 主要反映趋势相关，raw MAE 对绝对尺度偏差敏感；需要同时看 log1p MAE、heldout PCC、输出 shape 和日志成功标记。

## 4. 批量运行当前 v3

主提交脚本：

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi
sbatch 2_FLAMINGOData/v3_scripts/submit_higashi.sbatch
```

`submit_higashi.sbatch` 是 array job：

- `#SBATCH --array=0-13`
- 7 个 dataset x 2 个 neighbor。
- 偶数 task 对应 `neighbor=0`。
- 奇数 task 对应 `neighbor=5`。
- 使用 GPU 环境 `/public/home/hpc254701055/micromamba/envs/6_higashi/bin/python` 跑 Higashi。
- 使用 Python 3.8 环境 `/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python` 做 prepare、collect、log1p evaluate。

每个 task 的顺序：

1. 如果 `config.JSON` 不存在，运行 `v3_prepare_higashi.py`。
2. 如果对应 Higashi impute hdf5 不存在，运行 `run_higashi_one.py --training-updates 200 --eval-updates 10`。
3. 总是运行 `v3_collect_higashi.py`，更新 NPZ。
4. 总是运行 `v3_evaluate.py --append`，追加 log1p 指标。

如果日志里出现 `[skip] prepare` 或 `[skip] Higashi`，说明相应中间文件已经存在。本次耗时不能代表从零预处理或从零训练的耗时。

raw 指标单独提交：

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi
sbatch 2_FLAMINGOData/v3_scripts/submit_eval_higashi_v3_metrics_cpu.sbatch
```

## 5. `updates=1000` 独立实验

这个实验用于比较 `--training-updates 1000` 与基线 `--training-updates 200` 的插补差异。它仍保持：

- `embedding_epoch = 60`
- `no_nbr_epoch = 45`
- `with_nbr_epoch = 30`
- `--eval-updates 10`

因此这里的 `1000` 不是把 `embedding_epoch` 改成 1000，而是把 Higashi 每个训练 epoch 的 update 数改为 1000。

### 已完成子集

先完成的 4 个任务：

```text
v3_hybrid_W0p5_500cells_level0 nbr0/nbr5
v3_hybrid_W0p7_500cells_level0 nbr0/nbr5
```

提交脚本：

```bash
sbatch 2_FLAMINGOData/v3_scripts/submit_higashi_epoch1000_subset_gpu4q.sbatch
```

已完成 job：

```text
GPU array: 20064012
CPU compare: 20064016
```

子集比较输出：

```text
v3_epoch1000_outputData/metrics/higashi_epoch1000_vs_epoch200_summary.csv
v3_epoch1000_outputData/metrics/higashi_epoch1000_subset_log1p_metrics.csv
v3_epoch1000_outputData/metrics/higashi_epoch1000_subset_raw_gt_PCC_MAE.csv
```

子集结果摘要：`updates=1000` 对 `W0p7 nbr0/nbr5` 明显更好；对 `W0p5` 是混合结果，部分 PCC 变好但 MAE 或 heldout 指标有变差。

### 剩余 5 个数据集

剩余任务覆盖：

```text
v3_hybrid_W0p6_500cells_level0 nbr0/nbr5
v3_hybrid_W0p7_500cells_level0_r0p01 nbr0/nbr5
v3_hybrid_W0p7_500cells_level0_r0p05 nbr0/nbr5
v3_hybrid_W0p8_500cells_level0 nbr0/nbr5
v3_hybrid_W0p9_500cells_level0 nbr0/nbr5
```

提交脚本：

```bash
sbatch 2_FLAMINGOData/v3_scripts/submit_higashi_epoch1000_remaining_gpu4q.sbatch
```

当前提交信息：

```text
GPU array: 20064175
CPU all-compare: 20064185, dependency afterok:20064175
```

查看进度：

```bash
squeue -j 20064175,20064185
```

全量比较脚本：

```bash
sbatch --dependency=afterok:20064175 \
  2_FLAMINGOData/v3_scripts/submit_compare_higashi_epoch1000_all_cpu.sbatch
```

全量完成后应检查：

```bash
find 2_FLAMINGOData/v3_epoch1000_outputData/npz_lower_tri -name '*_lower_tri.npz' | wc -l
find 2_FLAMINGOData/v3_epoch1000_outputData/metrics/log1p_tasks -name '*_log1p_metrics.csv' | wc -l
```

两个数字都应为 `14`。

全量比较输出：

```text
v3_epoch1000_outputData/metrics/higashi_epoch1000_all_vs_epoch200_summary.csv
v3_epoch1000_outputData/metrics/higashi_epoch1000_all_log1p_metrics.csv
v3_epoch1000_outputData/metrics/higashi_epoch1000_all_raw_gt_PCC_MAE.csv
```

## 6. 单个数据集手动运行示例

下面示例以 `v3_hybrid_W0p7_500cells_level0` 和 `neighbor=5` 为例。实际环境建议与 sbatch 一致。

```bash
ROOT=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi
SCRIPT_DIR=${ROOT}/2_FLAMINGOData/v3_scripts
DATA_DIR=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets
INPUT_ROOT=${ROOT}/2_FLAMINGOData/v3_inputData
OUTPUT_ROOT=${ROOT}/2_FLAMINGOData/v3_outputData
STEM=v3_hybrid_W0p7_500cells_level0
NEIGHBOR=5
PYTHON=/public/home/hpc254701055/micromamba/envs/6_higashi/bin/python
PY38=/public/home/hpc254701055/micromamba/envs/3_schicluster_python38/bin/python
```

Prepare:

```bash
${PY38} ${SCRIPT_DIR}/v3_prepare_higashi.py \
  --data-dir ${DATA_DIR} \
  --input-root ${INPUT_ROOT} \
  --datasets ${STEM} \
  --neighbors ${NEIGHBOR} \
  --cpu-num 20 \
  --gpu-num 1 \
  --overwrite
```

Train and impute:

```bash
${PYTHON} ${SCRIPT_DIR}/run_higashi_one.py \
  --config ${INPUT_ROOT}/${STEM}_nbr${NEIGHBOR}/config.JSON \
  --training-updates 200 \
  --eval-updates 10
```

Collect:

```bash
${PY38} ${SCRIPT_DIR}/v3_collect_higashi.py \
  --input-root ${INPUT_ROOT} \
  --output-root ${OUTPUT_ROOT} \
  --datasets ${STEM} \
  --neighbor ${NEIGHBOR} \
  --overwrite
```

Log1p evaluate:

```bash
${PY38} ${SCRIPT_DIR}/v3_evaluate.py \
  --method "Higashi ${NEIGHBOR} nbr" \
  --config-tag "${NEIGHBOR}nbr" \
  --data-dir ${DATA_DIR} \
  --pred-dir ${OUTPUT_ROOT}/npz_lower_tri \
  --pred-pattern "{stem}_higashi_nbr_${NEIGHBOR}_lower_tri.npz" \
  --output-csv ${OUTPUT_ROOT}/metrics/scHiCluster_FLAMINGO_v3_paramsweep_quality_metrics.csv \
  --datasets ${STEM} \
  --workers 20 \
  --append
```

Raw evaluate for all available lower-triangle NPZ:

```bash
python ${SCRIPT_DIR}/13_cal_FLAMINGO_Baseline_metrics.py \
  --higashi-v3-pred-dir ${OUTPUT_ROOT}/npz_lower_tri \
  --higashi-v3-gt-dir ${DATA_DIR} \
  --higashi-v3-output-csv ${OUTPUT_ROOT}/metrics/higashi_FLAMINGO_v3_paramsweep_raw_gt_PCC_MAE.csv
```

## 7. 迁移到新数据集时怎么改

### 情况 A：新数据仍是 FLAMINGO-like 500 beads h5ad

如果新数据仍是：

- 单染色体；
- 500 bins；
- 124750 个上三角 feature；
- feature 顺序与 `np.triu_indices(500, k=1)` 一致；
- 有 `layers/counts`；
- 有 `layers/gt` 用于评估；

则主要修改：

1. 把新 h5ad 放到一个新 `DATA_DIR`。
2. 保持文件名为 `<stem>_scdiff2.h5ad`，或同步修改 `v3_common.discover_datasets()` 的 glob。
3. 更新 `submit_higashi.sbatch` 中的 `DATA_DIR` 和 `DATASETS=(...)`。
4. 建议为新数据设置新的 `INPUT_ROOT` 和 `OUTPUT_ROOT`，不要覆盖当前成功结果。
5. 如果没有 `fixed_heldout_indices.npz`，`v3_evaluate.py` 仍可计算 `all` 和 `observed`，但 `heldout` 会为空或 NaN。

### 情况 B：bins 数、feature 数或染色体名变化

必须同步修改脚本，不能只替换数据：

1. 在 `v3_common.py` 修改：
   - `N_BINS`
   - `N_FEATURES`
   - `CHROM_NAME`
   - `DEFAULT_DATA_DIR`
2. 确认 `feature_to_bins()` 的顺序仍匹配新 h5ad 的 `var/_index`。如果新数据不是上三角 row-major，就要重写 feature index 到 `(i, j)` 的映射。
3. 在 `v3_prepare_higashi.py` 确认：
   - `chrom_start_end.npy` 的范围正确；
   - `flamingo.chrom.sizes` 文件名和内容适合新染色体；
   - `min_delta=2` 是否仍合理。
4. 在 `v3_collect_higashi.py` 确认读取的 hdf5 路径和 `CHROM_NAME` 一致。
5. 在 `v3_evaluate.py` 和 raw evaluator 中确认 GT shape 与 prediction shape 完全一致。

### 情况 C：没有 GT

Higashi 插补本身不需要 GT。没有 GT 时：

- 仍可以运行 prepare、train、impute、collect。
- 跳过 `v3_evaluate.py` 和 `13_cal_FLAMINGO_Baseline_metrics.py`，或改写评估逻辑。
- 不要把 `layers/counts` 同时当输入和 GT 评价，否则指标会失真。

## 8. 验证清单

当前 v3 成功标准：

```bash
find 2_FLAMINGOData/v3_inputData -name config.JSON | wc -l
find 2_FLAMINGOData/v3_outputData/npz_lower_tri -name '*_lower_tri.npz' | wc -l
rg -l '\[OK\] Higashi run finished' 2_FLAMINGOData/v3_scripts/logs | wc -l
```

当前三个数字都应该是 `14`。

检查日志：

```bash
rg -n 'Traceback|Error|Exception|Killed|CUDA out of memory' 2_FLAMINGOData/v3_scripts/logs
rg -n '\[OK\] Higashi run finished|Finished at' 2_FLAMINGOData/v3_scripts/logs
```

检查输出 shape：

```bash
python - <<'PY'
from pathlib import Path
from scipy.sparse import load_npz

for p in sorted(Path("2_FLAMINGOData/v3_outputData/npz_lower_tri").glob("*_lower_tri.npz")):
    mat = load_npz(p)
    print(p.name, mat.shape, mat.nnz)
PY
```

当前 FLAMINGO v3 每个 shape 应为 `(1500, 124750)`。迁移新数据时，应替换为新数据自己的 `(n_cells, n_features)`。

检查 metrics 行数：

```bash
python - <<'PY'
import pandas as pd
for p in [
    "2_FLAMINGOData/v3_outputData/metrics/scHiCluster_FLAMINGO_v3_paramsweep_quality_metrics.csv",
    "2_FLAMINGOData/v3_outputData/metrics/higashi_FLAMINGO_v3_paramsweep_raw_gt_PCC_MAE.csv",
]:
    df = pd.read_csv(p)
    print(p, len(df), df[["method", "config_tag"]].drop_duplicates().to_string(index=False))
PY
```

当前每个 CSV 应有 14 行。

## 9. 常见问题和注意事项

- 不要混淆两套评估：`v3_evaluate.py` 是 log1p 后的 all/observed/heldout 指标；`13_cal_FLAMINGO_Baseline_metrics.py` 默认是 raw GT 和 raw prediction。
- raw MAE 偏高但 PCC 高，通常表示绝对尺度有偏差但整体变化趋势一致；需要结合 log1p MAE、heldout PCC 和输出 shape 判断。
- `cpu_num=20` 不等于 PyTorch 训练使用 20 线程；当前配置中 `cpu_num_torch` 被限制到最多 4。
- 基线成功运行使用 `--training-updates 200 --eval-updates 10`；`updates=1000` 只在 `v3_epoch1000_*` 目录中做独立超参实验，不能覆盖基线输出。
- 看到日志中 `finish imputing, used ~90 s` 只代表某个 impute 阶段耗时，不代表完整任务总耗时；完整任务还包括 create_matrix、embedding 训练、imputation 训练、collect 和 evaluate。
- `neighbor=5` 会先生成 `nbr_0` impute，再生成 `nbr_5` impute；最终 `v3_collect_higashi.py --neighbor 5` 只收集 `nbr_5` 文件。
- 迁移新数据时，最容易出错的是 feature 顺序。只要 h5ad feature 顺序和 `feature_to_bins()` 不一致，即使脚本跑通，评估也会错位。
- 当前 FLAMINGO v3 是单染色体 `chrFLAMINGO`。多染色体真实 scHi-C 数据不能直接套用本目录脚本，需要扩展 chrom list、chrom sizes、feature 映射和 collect 逻辑。
- 不要为了看结果直接读取巨大输入或输出全文；优先用 shape、nnz、日志关键字和 metrics CSV 抽查。

## 10. 推荐的新数据集迁移步骤

1. 先用小脚本检查新 h5ad：
   - `layers/counts` 是否存在；
   - `layers/gt` 是否存在；
   - shape 是否符合预期；
   - `var/_index` 是否和 feature 映射一致。
2. 复制 `v3_scripts` 到新实验目录，保留当前成功目录不动。
3. 修改 `v3_common.py` 的 `DEFAULT_DATA_DIR`、`N_BINS`、`N_FEATURES`、`CHROM_NAME`。
4. 修改 `submit_higashi.sbatch` 的 `DATA_DIR`、`INPUT_ROOT`、`OUTPUT_ROOT`、`DATASETS` 和 array 范围。
5. 先跑 1 个 dataset x 1 个 neighbor，确认：
   - prepare 生成 `config.JSON/data.npy/weight.npy`；
   - Higashi 日志出现 `[OK] Higashi run finished`；
   - collect 后 NPZ shape 正确；
   - 如果有 GT，log1p 和 raw 评估都能产生 1 行结果。
6. 再提交完整 array job。
7. 完整运行后用第 7 节的验证清单检查数量、日志、shape 和 metrics 行数。
