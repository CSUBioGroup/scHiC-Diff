# schicluster_office_code Ramani results

This isolated directory contains the Ramani common-embedding sensitivity
analysis requested under the literal `schicluster_office_code` prefix.

## Workflow

```text
imputed/raw chromosome contacts
  -> first off-diagonal at 1 Mb
  -> per-chromosome TruncatedSVD64 or TruncatedSVD128
  -> norm_sig
  -> chromosome concatenation
  -> final TruncatedSVD64 or TruncatedSVD128
  -> norm_sig
  -> first d components, d = 2, 5, 10, 20, 50
  -> KMeans(k=4, n_init=200, random_state=None)
  -> ARI
```

No log1p, PCA, UMAP, or Z-score is applied. Per-chromosome component counts are
capped when a chromosome has fewer available first-off-diagonal features; the
final embeddings are exactly `626 x 64` and `626 x 128`.

These are common two-stage TruncatedSVD embeddings reconstructed from contact
matrices. They are not each method's native embedding and are not the native
PCA embedding returned by historical `hicluster_gpu()`.

## Main files

- `embeddings/dim_D/CONDITION/total_decomp.npz`: self-describing embedding with
  `data`, `cells`, `cell_types`, `source`, and `dimensions` keys.
- `schicluster_office_code_Ramani_ARI_long.csv`: all 80 ARIs.
- `schicluster_office_code_Ramani_ARI_wide.csv`: compact comparison table.
- `schicluster_office_code_Ramani_KMeans_labels.npz`: all 80 cluster vectors.
- `schicluster_office_code_Ramani_cluster_coordinates.csv`: deterministic
  visualization-only UMAP coordinates generated from exactly the same source
  and prefix used by each main ARI: Raw first50, scHiCluster first5,
  HiCImpute first2, Higashi-nbr0/5 first10, scVI-3D first10,
  Tensor-FLAMINGO first20, and scHiC-Diff first20.
- `schicluster_office_code_Ramani_plot_ARI_long.csv`: 40-row plot source table
  containing configured rows from the full no-log1p SVD64/128 sensitivity table.
- `schicluster_office_code_Ramani_plot_ARI_run_config.json`: records every
  configured embedding source dimension and prefix selected for plotting.
- `schicluster_office_code_Ramani_embedding_validation.csv`: 368 chromosome
  validation records.
- `schicluster_office_code_Ramani_ARI_validation.csv`: 16 embedding/order
  validation records.
- `schicluster_office_code_Ramani_*_run_config.json`: complete parameters.

The historical K-means expression does not set `random_state`; saved cluster
labels therefore define the exact realized results in this directory.
