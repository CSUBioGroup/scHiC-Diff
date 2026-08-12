**English** | [中文](README_zh.md)

# Imputed Contact-Map Figures

This directory contains the publication plotting workflow for contact maps from two simulated scHi-C benchmarks. Dataset-specific code, metric tables, manifests, logs, and outputs are separated under `nature-style-plot/`.

## Projects

- [`nature-style-plot/1_HiCImputedData/`](nature-style-plot/1_HiCImputedData/): HiCImputeData main grid, 7k supplementary grid, and individual method panels.
- [`nature-style-plot/2_FLAMINGOData/`](nature-style-plot/2_FLAMINGOData/): FLAMINGOData 7 x 9 comparison grid.
- [`nature-style-plot/gr_style.py`](nature-style-plot/gr_style.py): shared publication style and output helpers.

See the [detailed plotting README](nature-style-plot/README.md) for the dataset layout. Dataset-specific CSV files and figures should remain in their corresponding subdirectories.
