**English** | [中文](README_zh.md)

# Baseline Methods and Paper-Figure Workflows

This directory contains the reproducible baseline evaluation workflows used in the scHiC-Diff project.

## Directory Organization

- [`paperplots/`](paperplots/) contains the plotting and evaluation code used to generate the paper figures, metric summaries, clustering comparisons, contact maps, runtime figures, and case-study analyses.
- The numbered method directories at this level contain the imputation workflows, adapters, preprocessing scripts, submission scripts, and result-collection code used to run each comparison method in this project.
- [`0_gtData/`](0_gtData/) contains the ground-truth and reference-data preparation structure shared by the benchmark workflows.

## Imputation Method Directories

| Directory | Method or role |
|---|---|
| [`1_scVI-3D/`](1_scVI-3D/) | scVI-3D imputation workflows |
| [`3_HiCImpute/`](3_HiCImpute/) | HiCImpute imputation workflows |
| [`4_scHiCluster/`](4_scHiCluster/) | scHiCluster imputation workflows |
| [`5_scHiCTools/`](5_scHiCTools/) | scHiCTools-related comparison workflows |
| [`6_Higashi/`](6_Higashi/) | Higashi imputation workflows, including neighbor-setting variants |
| [`7_scHiCDiff/`](7_scHiCDiff/) | scHiC-Diff benchmark inference and result-processing workflows |
| [`8_ScUnicorn/`](8_ScUnicorn/) | ScUnicorn-related comparison workflows |
| [`9_FLAMINGO/`](9_FLAMINGO/) | Tensor-FLAMINGO imputation workflows |

Each method directory preserves the project-specific data conversion, execution, and output contracts required for consistent comparisons. Large input and output matrices are intentionally excluded from Git; retained empty directories indicate the expected filesystem layout.

Start with the README inside a method directory when reproducing its imputation results. For figures and cross-method comparisons, open [`paperplots/`](paperplots/) and use the bilingual README in the relevant numbered plotting directory.
