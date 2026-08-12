# Higashi Ramani Pre-Imputation

This directory contains two Ramani Higashi configurations for the unified ARI/cluster plot:

- `Higashi (0 nbr)`: `neighbor_num=0`
- `Higashi (5 nbr)`: `neighbor_num=5`

Both start from the same Ramani ML1+ML3 1 Mb matrices and export separate plotting outputs.

## Input

Source matrices:

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/2-Ramani-GSE84920-ML1-ML3/upper_npz/1000000bp/ML1_ML3/chr*.npz
```

Final row order is fixed by:

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/2-Ramani-GSE84920-ML1-ML3/upper_npz/1000000bp/ML1_ML3_cell_list.txt
```

`scripts/prepare_ramani_higashi_inputs.py` first creates `input/raw_626_chrom_npz`, then writes one Higashi `config.JSON` per chromosome and neighbor setting.

## Preprocessing Contract

The Ramani pipeline follows the same Higashi v1 prebuilt-numpy contract used by the successful FLAMINGO v3 run, adjusted to real multi-chromosome Ramani inputs:

```text
input/<chrom>_nbr<0|5>/
  config.JSON
  data/ramani.chrom.sizes
  data/label_info.pickle
  temp/data.npy
  temp/weight.npy
  temp/chrom_start_end.npy
  temp/raw/
```

Per chromosome, `data.npy` has rows `[cell_id, 0, i_idx, j_idx]`, where chromosome index is always `0` inside that one-chromosome Higashi job. `weight.npy` stores the matching observed contact values as `float32`, and `chrom_start_end.npy` is `[[0, n_bins]]`.

Important defaults:

- Feature order is `np.triu_indices(n_bins, k=1)`, matching the source `chr*.npz` upper-triangle columns.
- Contacts are kept only when finite, positive, and `j_idx - i_idx >= 2` (`--min-delta 2`).
- `config.JSON` uses `input_format="higashi_v1"`, because Higashi itself only branches on `higashi_v1` / `higashi_v2`; the prebuilt arrays are consumed directly by `create_matrix()`.
- Higashi hyperparameters mirror the successful FLAMINGO setup unless explicitly overridden: `dimensions=64`, `loss_mode=zinb`, `embedding_epoch=60`, `no_nbr_epoch=45`, `with_nbr_epoch=30`, `cpu_num_torch=min(cpu_num, 4)`.

## Run

Prepare only:

```bash
/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python scripts/prepare_ramani_higashi_inputs.py --neighbor-num 0
/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python scripts/prepare_ramani_higashi_inputs.py --neighbor-num 5
```

Run arrays:

```bash
sbatch run_ramani_higashi_nbr0.sbatch
sbatch run_ramani_higashi_nbr5.sbatch
```

The sbatch defaults use:

```text
--training-updates 1000
--eval-updates 10
```

Here `updates=1000` means `run_higashi_one.py --training-updates 1000`, which patches Higashi's `update_num_per_training_epoch` to `1000`. It does **not** change `embedding_epoch`, `no_nbr_epoch`, or `with_nbr_epoch` to 1000. To override for a run, submit with environment variables, for example:

```bash
TRAINING_UPDATES=200 EVAL_UPDATES=10 sbatch run_ramani_higashi_nbr0.sbatch
```

These sbatch files call the existing patched Higashi runner:

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/1_HiCImputeData/scripts/run_higashi_one.py
```

Collect after training:

```bash
/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python scripts/collect_ramani_higashi_outputs.py --neighbor-num 0 --manifest input/ramani_higashi_nbr0_manifest.tsv --make-embedding
/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python scripts/collect_ramani_higashi_outputs.py --neighbor-num 5 --manifest input/ramani_higashi_nbr5_manifest.tsv --make-embedding
```

If Higashi writes hdf5 files to a different path, pass `--hdf5-pattern`.

The collect script writes one sparse upper-triangle matrix per chromosome, replaces NaN/Inf/negative imputed values with zero, and can build a single concatenated embedding for plotting.

## Output Contract

```text
output/higashi_nbr0/chrom_npz/chr*.npz
output/higashi_nbr0/ramani_embedding.npz
output/higashi_nbr5/chrom_npz/chr*.npz
output/higashi_nbr5/ramani_embedding.npz
```

Each output must have 626 rows aligned to `ML1_ML3_cell_list.txt`.

## ARI & UMAP 计算

插补完成后，从 `output/higashi_nbr0/ramani_embedding.npz` 或 `output/higashi_nbr5/ramani_embedding.npz` 出发计算 ARI 和 UMAP 坐标。

### ramani_embedding.npz 的用途

由 collect 脚本调用 `save_embedding_from_chrom_npz()` 生成：
- shape：(626, per_chrom_dim × 23) = (626, 115)
- 由 `TruncatedSVD(n_components=5, random_state=100)` 对 `np.log1p(chrom_npz)` 降维后拼接
- collect 默认 `--per-chrom-dim 5 --seed 100`，与这里的复现代码一致

### 从 embedding.npz 复现 ARI

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import umap

embedding = np.load("output/higashi_nbr0/ramani_embedding.npz", allow_pickle=True)["data"]

labels = [c.split("_")[0] for c in cell_list]
labels_true = LabelEncoder().fit_transform(labels)

kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=0)
for cd in range(1, 11):
    reducer = umap.UMAP(n_components=cd, random_state=500)
    emb = reducer.fit_transform(embedding)
    labels_pred = kmeans.fit_predict(emb)
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    print(f"dim={cd}: ARI={ari:.4f}")
```

### 从 embedding.npz 生成 UMAP 图坐标

```python
reducer = umap.UMAP(n_components=2, random_state=500)
umap_coords = reducer.fit_transform(embedding)  # (626, 2)
```

### 从 chrom_npz 重新计算（完全复现）

```python
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

features = []
for chrom in chroms:
    x = sparse.load_npz(f"output/higashi_nbr0/chrom_npz/{chrom}.npz").tocsr().toarray()
    x = np.log1p(x)
    svd = TruncatedSVD(n_components=5, random_state=100)
    features.append(svd.fit_transform(x))
X = np.hstack(features)  # (626, 115)
# 后续 UMAP + KMeans 同上（参数见下表）
```

### 关键参数速查表

| 步骤 | 参数 |
|------|------|
| per-chrom SVD | `TruncatedSVD(n_components=5, random_state=100)` |
| log1p | 必须在 SVD 前做 |
| merge UMAP | `umap.UMAP(n_components=dim, random_state=500)`（遍历 dim 1-10） |
| KMeans | `n_clusters=4, init="k-means++", n_init=10, random_state=0` |
| 标签 | `ML1_ML3_cell_list.txt` 中每个细胞名 `_` 前的 cell type 前缀 |
