**English** | [中文](README_zh.md)

# PCC, MAE, and SCC Evaluation of Imputed Results

This directory provides a unified workflow for calculating imputation metrics for HiCImputeData and FLAMINGOData. Both dataset families share one Python entry point and the same method-path configuration, while final results are stored separately by dataset family.

All commands below assume `paperplots/1_pccAndMae_all/` as the current working directory.

## Directory Contents

```text
1_pccAndMae_all/
├── calculate_imputation_metrics.py
├── imputation_metric_config.py
├── submit_metric_array.sbatch
├── submit_metric_control.sbatch
├── imputation_metric_tasks.csv
├── 1_HiCImputedData/
│   ├── HiCImputeData_PCC_MAE_SCC_metrics.csv
│   ├── per_cell_metrics/
│   └── README.md
├── 2_FLAMINGOData/
│   ├── FLAMINGOData_PCC_MAE_SCC_metrics.csv
│   ├── per_cell_metrics/
│   └── README.md
└── logs/
```

- `calculate_imputation_metrics.py`: the sole metric-calculation entry point, with the `prepare-manifest`, `run-task`, and `aggregate` subcommands.
- `imputation_metric_config.py`: the single configuration source for two dataset families, seven methods, raw matrix paths, feature order, and matrix-reading rules.
- `imputation_metric_tasks.csv`: 133 calculation tasks generated from the configuration, including 49 FLAMINGOData tasks and 84 HiCImputeData tasks.
- `per_cell_metrics/`: per-cell metric JSON files for each method-by-condition combination. The final CSV files are aggregated from these JSON files; do not modify only the CSV without updating the per-cell results.
- `logs/`: Slurm standard output and error logs.

The three core files are related as follows:

```text
imputation_metric_config.py              # Manually maintained path and format configuration
        │ prepare-manifest
        ▼
imputation_metric_tasks.csv              # Automatically generated Slurm task snapshot
        │ run-task / Slurm array
        ▼
1_HiCImputedData/per_cell_metrics/       # 84 per-cell JSON files
2_FLAMINGOData/per_cell_metrics/         # 49 per-cell JSON files
        │ aggregate
        ▼
HiCImputeData_PCC_MAE_SCC_metrics.csv    # 84 official summary records
FLAMINGOData_PCC_MAE_SCC_metrics.csv     # 49 official summary records
```

`imputation_metric_config.py` is the only configuration source that should be maintained manually. `imputation_metric_tasks.csv` is derived during the prepare stage and must not be edited by hand. Rerun prepare after any configuration change.

## Method and Dataset Coverage

The seven methods are:

```text
scVI-3D, HiCImpute, scHiCluster, Higashi_nbr0, Higashi_nbr5,
Tensor-FLAMINGO, scHiC-Diff
```

HiCImputeData covers 12 conditions from `T1/T2/T3 × 1k/2k/4k/7k`, yielding `7 × 12 = 84` records in the final table. FLAMINGOData covers the W sweep from 0.5 to 0.9 plus `P=1%` and `P=5%` at `W=0.7`, yielding `7 × 7 = 49` records.

Official Tensor-FLAMINGO metrics for HiCImputeData use `9_FLAMINGO/1_HiCImputeData/output_distance_best/contact_from_pd/npz_lower_tri/`. The former `output_distance/contact_from_pd/npz_lower_tri/` produces zero-variance predictions in 11 of 12 conditions, causing held-out PCC/SCC to become NaN, and is therefore no longer an official evaluation input.

## Metric Definitions

PCC, MAE, and SCC are calculated separately for each cell, followed by the cross-cell mean and population standard deviation (`numpy.nanstd`, `ddof=0`). Three evaluation subsets are used:

| Suffix | Feature range |
|---|---|
| `all` | Complete triangular feature vector, including positions where GT is 0 |
| `obs` | Positions where the observed/input value is greater than 0 |
| `held` | Positions where GT is greater than 0 and observed/input is not greater than 0 |

PCC is the Pearson correlation coefficient, SCC is the Spearman rank correlation coefficient, and MAE is mean absolute error. Field names follow `{metric}_{subset}_{mean|std}`, for example `pcc_held_mean`.

Metrics that cannot be calculated for a cell are recorded as NaN. Aggregation uses `numpy.nanmean` and `numpy.nanstd` to ignore these NaNs. Each JSON also records the NaN count for every metric; inspect these counts before using the official results instead of relying only on final means.

Both HiCImputeData and FLAMINGOData use the saved raw values directly, without `log`, `log1p`, clipping of negative values, or additional normalization. Masks are always defined from the original GT and observed matrices. See the README in each dataset subdirectory for triangular-feature rules and details of the FLAMINGOData `denoise_recon_inv.npz` input.

## Slurm Recalculation Workflow

Do not load matrices or calculate metrics on the login node. First generate and validate the task manifest:

```bash
mkdir -p logs
sbatch submit_metric_control.sbatch prepare
```

After confirming that the prepare job succeeded and `imputation_metric_tasks.csv` contains 134 lines including the header, submit the 133 array tasks. `%10` limits concurrency and may be adjusted for cluster availability:

```bash
sbatch --array=0-132%10 submit_metric_array.sbatch
```

Aggregate after every array task succeeds:

```bash
sbatch submit_metric_control.sbatch aggregate
```

An `afterok` dependency can gate aggregation automatically:

```bash
array_job=$(sbatch --parsable --array=0-132%10 submit_metric_array.sbatch)
sbatch --dependency=afterok:${array_job} submit_metric_control.sbatch aggregate
```

The calculation environment is fixed to `micromamba/envs/2_schic-scvi-3d`, and Slurm uses `cpuQ/cpuq`. If method output paths, filenames, or feature order change, modify only `imputation_metric_config.py` and rerun the complete workflow beginning with prepare.

## Partial Recalculation

The main program supports `--method`, `--dataset-family`, and `--dataset` filters. Official result directories still use fixed filenames. For a partial recalculation, retain the complete 133-row task manifest and submit only the task IDs that need updating.

For example, recalculate all seven FLAMINGOData methods with:

```bash
array_job=$(sbatch --parsable --array=0-48%7 submit_metric_array.sbatch)
sbatch --dependency=afterok:${array_job} submit_metric_control.sbatch aggregate FLAMINGOData
```

This rewrites the 49-row `FLAMINGOData_PCC_MAE_SCC_metrics.csv` without touching the 84-row HiCImputeData CSV. `aggregate FLAMINGOData` rejects FLAMINGOData JSON files with a transform other than `raw`, preventing old `log1p` results from being mixed with new results.

When updating only one method, and all other per-cell JSON files for that dataset family already exist, filter the required task IDs from the complete manifest. For example, to recalculate scVI-3D:

```bash
method="scVI-3D"
task_ids=$(awk -F, -v method="${method}" \
  'NR > 1 && $2 == "FLAMINGOData" && $4 == method {ids = ids (ids ? "," : "") $1} END {print ids}' \
  imputation_metric_tasks.csv)
array_job=$(sbatch --parsable --array="${task_ids}%10" submit_metric_array.sbatch)
sbatch --dependency=afterok:${array_job} submit_metric_control.sbatch aggregate FLAMINGOData
```

The example above targets FLAMINGOData. For HiCImputeData, change the final argument to `HiCImputeData`. Aggregation fails without producing incomplete output if any method/condition JSON is missing from the selected dataset family.

The current script does not provide an independent `--output-dir` for a table containing only one method and isolated from the official CSV. Add an output-directory argument before attempting such a run; do not reuse the official aggregate output location.

## Relationship to the Plotting Directory

Contact-map plots use lightweight metric copies in their respective directories:

```text
../2_imputedContactHeatmap/nature-style-plot/1_HiCImputedData/HiCImputeData_PCC_MAE_metrics.tsv
../2_imputedContactHeatmap/nature-style-plot/2_FLAMINGOData/FLAMINGOData_PCC_MAE_metrics.tsv
```

After recalculation, update heatmap PCC annotations by copying PCC/MAE fields from the final CSV in this directory to the corresponding TSV, matched by `method + data_name/dataset`. Plotting scripts neither recalculate PCC nor overwrite these metric copies automatically.
