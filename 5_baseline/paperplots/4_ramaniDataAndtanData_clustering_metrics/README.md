**English** | [中文](README_zh.md)

# Ramani and Tan Clustering Evaluation

This directory contains the official clustering evaluation and paper-figure workflow for the Ramani ML1+ML3 and Tan datasets. It reads completed raw and imputed matrices from adjacent input directories and does not rerun any imputation method.

## Directory Layout

```text
inputRamaniData/       # Ramani raw input
imputedRamaniData/     # Ramani outputs from each imputation method
inputTanData/          # Tan raw input
imputedTanData/        # Tan outputs from each imputation method
nature-plot/           # Metric calculation, plotting, and official results
```

The workflow reconstructs method-specific Ramani embeddings, evaluates K-means ARI over selected and sensitivity-analysis dimensions, generates two-dimensional UMAP coordinates for display, evaluates Tan segments with ARI and aligned confusion matrices, and renders the main and standalone paper figures through one official plotting entry point.

See the [detailed workflow README](nature-plot/README.md) for input contracts, selected dimensions, scripts, Slurm submission, outputs, and interpretation boundaries.
