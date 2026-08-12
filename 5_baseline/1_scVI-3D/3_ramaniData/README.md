# scVI-3D Ramani Pre-Imputation

This directory is the Ramani output bridge for scVI-3D.

## Important Row-Order Note

The new Ramani plot reads labels from `ML1_ML3_cell_list.txt`, whose rows are barcode cell names. Some old scVI-3D Ramani files are organized as grouped method names like `K562_1`, `GM12878_1`, `HAP1_1`, `HeLa_1`. Those old grouped outputs are useful references, but they are not automatically guaranteed to match the barcode order required by the new plotting workflow.

Use `scripts/collect_scvi3d_ramani_outputs.py` only when you have confirmed the method cell order or have generated scVI-3D outputs in the same 626-cell order expected by the plot.

## Existing Reference Paths

Input-style txt files:

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/2-Ramani-GSE84920-ML1-ML3/scVI-3D_input/
```

Old result-like full matrices:

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/1_scVI-3D/2_Ramani/full_npz/
```

## Collect

```bash
sbatch run_collect_scvi3d_ramani.sbatch
```

or:

```bash
/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python scripts/collect_scvi3d_ramani_outputs.py --make-embedding
```

## Output Contract

```text
output/chrom_npz/chr1.npz ... chr22.npz, chrX.npz
output/ramani_embedding.npz
output/ramani_method_manifest_row.csv
```

Before adding this row to the plot manifest, verify row order against `ML1_ML3_cell_list.txt`.

## ARI & UMAP 计算

插补完成后，从 `output/ramani_embedding.npz` 或 `output/chrom_npz/` 出发计算 ARI 和 UMAP 坐标。

### ramani_embedding.npz 的用途

`ramani_embedding.npz` 由 collect 脚本调用 `save_embedding_from_chrom_npz()` 生成：
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

# 标签
labels = [c.split("_")[0] for c in cell_list]  # 取前缀 = cell type
labels_true = LabelEncoder().fit_transform(labels)

# UMAP 降维并遍历 cluster_dim
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
# umap_coords[:, 0] = UMAP1, umap_coords[:, 1] = UMAP2
```

### 从 chrom_npz 重新计算（完全复现）

如果 embedding.npz 参数不一致（log1p=False 或 SVD seed 不同），从 `chrom_npz` 重新开始：

```python
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

features = []
for chrom in chroms:
    x = sparse.load_npz(f"output/chrom_npz/{chrom}.npz").tocsr().toarray()
    x = np.log1p(x)
    svd = TruncatedSVD(n_components=5, random_state=100)
    features.append(svd.fit_transform(x))
X = np.hstack(features)  # (626, 115), 等价于 embedding.npz
# 后续 UMAP + KMeans 同上
```

### 关键参数速查表

| 步骤 | 参数 |
|------|------|
| per-chrom SVD | `TruncatedSVD(n_components=5, random_state=100)` |
| log1p | 必须在 SVD 前做 |
| merge UMAP | `umap.UMAP(n_components=dim, random_state=500)`（遍历 dim 1-10） |
| KMeans | `n_clusters=4, init="k-means++", n_init=10, random_state=0` |
| 标签 | `ML1_ML3_cell_list.txt` 中每个细胞名 `_` 前的 cell type 前缀 |
