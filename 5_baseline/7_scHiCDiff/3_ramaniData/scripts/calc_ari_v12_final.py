#!/usr/bin/env python3
"""Calculate ARI for scHiC-Diff v1.2 (3000 epoch) with old v1.2 pipeline."""

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import umap

cells = open("/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/2-Ramani-GSE84920-ML1-ML3/upper_npz/1000000bp/ML1_ML3_cell_list.txt").read().split()
labels = [c.split("_")[0] for c in cells]
le = LabelEncoder()
labels_true = le.fit_transform(labels)
n_clusters = len(le.classes_)
chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=0)

chrom_dir = "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/7_scHiCDiff/3_ramaniData/output/chrom_npz"

# === Pipeline 1: SVD dim=5 (scVI-3D old flow) ===
print("=== Pipeline 1: log1p + SVD dim=5 seed=100 + UMAP sweep ===")
features = []
for chrom in chroms:
    x = sparse.load_npz(f"{chrom_dir}/{chrom}.npz").tocsr().toarray()
    x = np.log1p(x)
    svd = TruncatedSVD(n_components=5, random_state=100)
    emb = svd.fit_transform(x)
    features.append(emb)
X = np.hstack(features)
print(f"拼接后: {X.shape}")
best_ari = 0; best_dim = 0
for cd in range(1, 11):
    reducer = umap.UMAP(n_components=cd, random_state=500)
    emb = reducer.fit_transform(X)
    labels_pred = kmeans.fit_predict(emb)
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    print(f"  dim={cd}: ARI={ari:.4f}, NMI={nmi:.4f}")
    if ari > best_ari: best_ari = ari; best_dim = cd
print(f"最佳: dim={best_dim}, ARI={best_ari:.4f}")

# === Pipeline 2: UMAP dim=12 (HiCImpute old flow) ===
print("\n=== Pipeline 2: log1p + per-chrom UMAP dim=12 seed=500 + merge UMAP dim=3 ===")
features2 = []
for chrom in chroms:
    x = sparse.load_npz(f"{chrom_dir}/{chrom}.npz").tocsr().toarray()
    x = np.log1p(x)
    reducer = umap.UMAP(n_components=12, random_state=500)
    emb = reducer.fit_transform(x)
    features2.append(emb)
X2 = np.hstack(features2)
print(f"拼接后: {X2.shape}")
reducer2 = umap.UMAP(n_components=3, random_state=500)
embedding2 = reducer2.fit_transform(X2)
for use_dim in [1, 2, 3]:
    labels_pred = kmeans.fit_predict(embedding2[:, :use_dim])
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    print(f"  use_dim={use_dim}: ARI={ari:.4f}, NMI={nmi:.4f}")