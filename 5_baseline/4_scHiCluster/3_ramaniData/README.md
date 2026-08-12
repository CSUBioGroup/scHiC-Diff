# scHiCluster Ramani Imputation

scHiCluster baseline for the Ramani dataset (626 cells, 23 chromosomes, 1 Mb resolution).

## Input

Source matrices (626 cells, already filtered from 630 → 626):

```text
input/raw_626_chrom_npz/chr*.npz
```

Each `chr*.npz` is a CSR sparse matrix of shape `(626, n_features)`, where `n_features = n_bins * (n_bins - 1) / 2` (upper triangle, `np.triu_indices(n_bins, k=1)` order).

Cell order is fixed by:

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/2-Ramani-GSE84920-ML1-ML3/upper_npz/1000000bp/ML1_ML3_cell_list.txt
```

Per-chromosome bin counts:

| chrom | n_bins | n_features | chrom | n_bins | n_features |
|-------|--------|------------|-------|--------|------------|
| chr1  | 250    | 31125      | chr13 | 116    | 6670       |
| chr2  | 244    | 29646      | chr14 | 108    | 5778       |
| chr3  | 199    | 19701      | chr15 | 103    | 5253       |
| chr4  | 192    | 18336      | chr16 | 91     | 4095       |
| chr5  | 181    | 16290      | chr17 | 82     | 3321       |
| chr6  | 172    | 14706      | chr18 | 79     | 3081       |
| chr7  | 160    | 12720      | chr19 | 60     | 1770       |
| chr8  | 147    | 10731      | chr20 | 64     | 2016       |
| chr9  | 142    | 10011      | chr21 | 49     | 1176       |
| chr10 | 136    | 9180       | chr22 | 52     | 1326       |
| chr11 | 136    | 9180       | chrX  | 156    | 12090      |
| chr12 | 134    | 8911       |       |        |            |

## Pipeline

```text
01_prepare_ramani.py    — 626-cell NPZ → per-cell txt (scHiCluster input)
03_impute_ramani.py     — hicluster impute-cell (626 cells × 23 chroms)
05_collect_ramani.py    — per-cell hdf5 → per-chrom NPZ + SVD embedding
ramani_cluster_eval.py  — ARI evaluation (shared with Higashi/scHiC-Diff)
```

### Step 1: Prepare

```bash
python scripts/01_prepare_ramani.py --chroms chr1 --workers 20 --overwrite
```

For each chromosome, reads `(626, n_features)` CSR from `input/raw_626_chrom_npz/{chrom}.npz`, reconstructs the full `n_bins × n_bins` symmetric contact matrix per cell, and writes non-zero upper-triangle contacts as `input/schicluster_input/{chrom}/cell_{id}_{chrom}.txt` (tab-separated: `row\tcol\tvalue`). Also writes `ramani_{chrom}.chrom.sizes` with `{chrom}\t{n_bins - 1}`.

Cell IDs 1..626 follow `ML1_ML3_cell_list.txt` order.

### Step 2: Impute

```bash
python scripts/03_impute_ramani.py --chroms chr1 --workers 20
```

Runs `hicluster impute-cell` per cell per chromosome with fixed parameters:

| Parameter | Value |
|-----------|-------|
| `--pad` | 1 |
| `--std` | 1 |
| `--rp` | 0.5 |
| `--tol` | 0.01 |
| `--res` | 1 |
| `--window_size` | n_bins (whole chromosome) |
| `--step_size` | n_bins |
| `--output_dist` | n_bins |
| `--output_format` | hdf5 |
| `--mode` | `pad1_std1_rp0.5_sqrtvc` |

Output: `output/1_imputed_hdf5/{chrom}/cell_{id}_{chrom}_pad1_std1_rp0.5_sqrtvc.hdf5`

Each hdf5 stores a CSR sparse matrix under `Matrix` group (data/indices/indptr + shape attribute).

### Step 3: Collect

```bash
python scripts/05_collect_ramani.py --chroms chr1 --workers 20 --overwrite
```

For each chromosome:
1. Read per-cell hdf5 (CSR `n_bins × n_bins`)
2. Extract upper-triangle features: `np.triu_indices(n_bins, k=1)` → `(n_features,)` per cell
3. Stack into `(626, n_features)` CSR
4. Save to `output/chrom_npz/{chrom}.npz`

NaN/Inf/negative values are replaced with 0.

### Step 4: Submit (SLURM array)

```bash
sbatch scripts/07_submit_ramani.sbatch
```

Array job 0-22 (23 chromosomes), each task runs prepare → impute → collect for one chromosome. 20 CPUs, 80G mem, 2-day time limit per task.

## ARI Evaluation Pipeline

After all 23 chromosomes are collected, compute ARI via clustering quality. This is the **standard pipeline shared across all methods** (scHiCluster, Higashi, scHiC-Diff, Tensor-FLAMINGO).

### Submit

```bash
sbatch scripts/08_submit_eval.sbatch
```

Uses the shared evaluation script:

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/6_Higashi/3_ramaniData/scripts/ramani_cluster_eval.py
```

Python env: `/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python` (has umap-learn)

### Pipeline Specification

The ARI computation consists of 4 stages. All parameters are fixed for reproducibility:

#### Stage 1: Per-chromosome SVD embedding

For each of the 23 chromosomes (`chr1`..`chr22`, `chrX`):

```python
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

x = sparse.load_npz("output/chrom_npz/chrN.npz").toarray().astype(np.float64)  # (626, n_features)
x = np.log1p(x)  # log1p BEFORE SVD
svd = TruncatedSVD(n_components=5, random_state=100)
emb = svd.fit_transform(x)  # (626, 5)
```

#### Stage 2: Merge

```python
chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
X = np.hstack([emb_chrom for chrom in chroms])  # (626, 115)
```

#### Stage 3: UMAP sweep + KMeans

For `dim` in 1..10:

```python
import umap
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

reducer = umap.UMAP(n_components=dim, random_state=500)
emb = reducer.fit_transform(X)  # (626, dim)

kmeans = KMeans(n_clusters=4, init="k-means++", n_init=100, random_state=100)
labels_pred = kmeans.fit_predict(emb)

ari = metrics.adjusted_rand_score(labels_true, labels_pred)
```

#### Stage 4: Cell type labels

Labels come from `ML1_ML3_cell_list.txt` — each cell name's prefix before `_`:

```python
cells = open("ML1_ML3_cell_list.txt").read().split()
labels = [c.split("_")[0] for c in cells]  # HeLa, HAP1, K562, GM12878
labels_true = LabelEncoder().fit_transform(labels)  # 0,1,2,3
```

Cell counts: HeLa=258, HAP1=214, K562=110, GM12878=44 (total=626).

### Parameter Summary

| Step | Parameter | Value |
|------|-----------|-------|
| log1p | Apply before SVD | `np.log1p(x)` |
| Per-chrom SVD | `TruncatedSVD` | `n_components=5, random_state=100` |
| Merge | hstack all 23 chroms | `(626, 115)` |
| UMAP | `umap.UMAP` | `n_components=dim (1-10), random_state=500` |
| KMeans | `KMeans` | `n_clusters=4, init="k-means++", n_init=100, random_state=100` |
| Labels | Cell type prefix | `ML1_ML3_cell_list.txt` → `cell.split("_")[0]` |
| Metric | ARI | `sklearn.metrics.adjusted_rand_score` |

### Output

```text
output/chrom_npz/chr*.npz                              # 23 per-chrom feature NPZ (626 rows)
output/ramani_embedding.npz                             # (626, 115) SVD embedding
output/scHiCluster_cluster_results/scHiCluster_ari_summary.txt
output/scHiCluster_cluster_results/embeddings/scHiCluster_dim{1..10}_ramani_embedding.npz
output/scHiCluster_cluster_results/plots/scHiCluster_cluster_dim{1..10}.png
```

## Results

| dim | ARI | NMI | Purity | Homo | ErrRate |
|-----|-----|-----|--------|------|---------|
| 1 | 0.2287 | 0.2717 | 0.6741 | 0.2876 | 0.4489 |
| 2 | 0.2056 | 0.2937 | 0.5958 | 0.3049 | 0.4872 |
| 3 | 0.2104 | 0.2963 | 0.5958 | 0.3054 | 0.4696 |
| 4 | 0.2252 | 0.3148 | 0.6054 | 0.3231 | 0.4537 |
| 5 | 0.2149 | 0.2987 | 0.5990 | 0.3081 | 0.4665 |
| 6 | 0.2229 | 0.3100 | 0.6070 | 0.3177 | 0.4569 |
| 7 | 0.2283 | 0.3209 | 0.6086 | 0.3295 | 0.4537 |
| 8 | 0.2190 | 0.3059 | 0.6038 | 0.3152 | 0.4649 |
| 9 | 0.2215 | 0.3119 | 0.6038 | 0.3210 | 0.4601 |
| 10 | 0.2123 | 0.3026 | 0.5974 | 0.3109 | 0.4665 |

Best: dim=1, ARI=0.2287. Mean ARI=0.2189.

### Cross-method comparison

| Method | Best ARI | Mean ARI |
|--------|----------|----------|
| scHiC-Diff (v12, 3000ep) | 0.7759 | 0.7700 |
| Higashi (5 nbr) | 0.5974 | 0.5625 |
| Higashi (0 nbr) | 0.5646 | 0.4458 |
| **scHiCluster** | **0.2287** | **0.2189** |
| Tensor-FLAMINGO | 0.0083 | 0.0058 |

All methods use the same ARI pipeline (log1p → per-chrom SVD dim=5 → hstack → UMAP sweep → KMeans → ARI). scHiCluster uses default parameters (pad=1, std=1, rp=0.5); no parameter tuning is needed or recommended.
