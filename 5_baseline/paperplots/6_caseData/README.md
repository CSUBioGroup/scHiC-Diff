**English** | [中文](README_zh.md)

# Developmental-Stage and Loop Case Studies

This directory contains the paper workflows for the developmental-stage clustering and long-range loop case studies.

## Projects

### Final Paper Figures

[`nature-plot-new/`](nature-plot-new/) is a standalone plotting bundle for the final main and supplementary figures. It uses copied, SHA-256-verified plotting inputs and does not rerun UMAP, silhouette, imputation, loop calling, APA, or held-out support calculations.

The main figure covers developmental-stage UMAPs, contact maps and loop summits, held-out normalized APA, and held-out raw-supported fractions. Supplementary outputs provide complete method grids and diagnostic plots. See the [figure README](nature-plot-new/README.md) for reproduction commands, provenance, outputs, and interpretation boundaries.

### Loop Calling and APA

[`2_callLoop_apa/`](2_callLoop_apa/) packages the Map2 locus workflow (`chr1:65-67 Mb`, 20 kb, 100 bins) into separate loop-comparison and APA pipelines. It supports shared cell sampling across methods and configuration-based addition of new imputation methods.

See the [loop/APA README](2_callLoop_apa/README.md) for directory structure, configuration templates, required dependencies, and execution commands.
