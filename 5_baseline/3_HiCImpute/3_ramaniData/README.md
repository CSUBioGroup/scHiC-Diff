# HiCImpute Ramani Pre-Imputation

This directory contains the Ramani ML1+ML3 pre-imputation workflow for the unified ARI/cluster plot.

## Input

Use the 1 Mb Ramani combined source matrices:

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/2-Ramani-GSE84920-ML1-ML3/upper_npz/1000000bp/ML1_ML3/chr*.npz
```

The raw matrices have 630 rows. The plotting workflow requires 626 rows in:

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/2-Ramani-GSE84920-ML1-ML3/upper_npz/1000000bp/ML1_ML3_cell_list.txt
```

`scripts/ramani_hicimpute_io.py prepare` filters the legacy four cells by zero-based indices `[414, 577, 556, 16]`, matching the old removal list `HAP1_189`, `GM12878_29`, `K562_60`, `HeLa_9`.

## Run

Prepare HiCImpute binary inputs:

```bash
/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python scripts/ramani_hicimpute_io.py prepare
```

Run per chromosome:

```bash
sbatch run_ramani_hicimpute_array.sbatch
```

The sbatch uses `RSCRIPT=${RSCRIPT:-Rscript}`. The `hic-impute` Python env exists, but `Rscript` was not found inside that env during inspection; load or point `RSCRIPT` to an R installation containing the `HiCImpute` R package.

Collect outputs after the array finishes:

```bash
/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python scripts/ramani_hicimpute_io.py collect --make-embedding
```

## Output Contract

Final plotting inputs are written under:

```text
output/chrom_npz/chr1.npz ... chr22.npz, chrX.npz
output/ramani_embedding.npz
output/ramani_method_manifest_row.csv
```

Every chromosome matrix must have 626 rows in `ML1_ML3_cell_list.txt` order.

## ARI & UMAP 计算

插补完成后，从 `output/ramani_embedding.npz` 或 `output/chrom_npz/` 出发计算 ARI 和 UMAP 坐标。

### ramani_embedding.npz 的用途

由 collect 脚本调用 `save_embedding_from_chrom_npz()` 生成：
- shape：(626, per_chrom_dim × 23) = (626, 115)（per_chrom_dim=5）
- 由 `TruncatedSVD(n_components=5, random_state=100)` 对 `np.log1p(chrom_npz)` 降维后拼接
- key `"data"` 存储 embedding 矩阵（float32）

### 从 embedding.npz 复现 ARI

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import umap

embedding = np.load("output/ramani_embedding.npz", allow_pickle=True)["data"]  # (626, 115)

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
    x = sparse.load_npz(f"output/chrom_npz/{chrom}.npz").tocsr().toarray()
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
