**English** | [中文](README_zh.md)

# Imputation Runtime Figures

This directory summarizes the end-to-end runtimes of imputation methods on two simulated datasets and generates candidate main-text figures, individual dataset figures, supplementary analyses, and numerical summary tables. All code paths are resolved relative to this directory and do not depend on absolute paths from other servers.

## Directory Structure

```text
plot_imputation_runtime.py          # Entry point for data loading, aggregation, and all plots
imputation_runtime_style.py         # Shared fonts, colors, axes, and runtime plot elements
submit_imputation_runtime.sbatch    # Slurm entry point for cpuQ/cpuq
data/                               # Raw runtime records
figures/                            # PDF and 600 dpi PNG outputs
results/                            # Summary CSV and LaTeX tables
logs/                               # Slurm standard output and error logs
```

## Input Data

```text
data/TensorFLAMINGO_simulations_500x500_runtime.csv
data/HiCImpute_simulations_61x61_runtime.csv
```

These CSV files were derived from `all_methods_FLAMINGO_v3_impute_time.csv` and `all_methods_HiCImputeData_impute_time.csv`, respectively. This reorganization standardizes only the filenames and locations; it does not alter any runtimes, hardware descriptions, configurations, or log provenance.

The two data-condition names used in the figures are:

- `Tensor-FLAMINGO simulations 500x500`
- `HiCImpute simulations 61x61`

The CSV identifiers `Higashi_nbr0` and `Higashi_nbr5` are displayed only as `Higashi-nbr0` and `Higashi-nbr5` in figures and summary tables; the raw fields remain unchanged. The hardware label for `Tensor-FLAMINGO` is standardized to `CPU x20`. This normalization affects only figures and summary tables and does not rewrite the hardware field in the raw CSV files.

## Statistical Definitions

- The median `impute_time_seconds` for each method determines the bar or point position.
- Dark points represent individual runs; summary tables also report sample size, minimum, maximum, and mean.
- The x-axis uses a logarithmic scale in seconds, with reference markers for seconds, minutes, hours, and days along the top.
- A separate readout column on the right reports the median runtime and primary hardware without allowing labels outside the axes to increase the final figure width.
- For methods explicitly recorded as total runtime for a parallel batch, the mean-per-dataset figure divides total runtime by the number of datasets and marks the entry with `*` in the summary table. The main-text bar chart still shows the actual end-to-end batch wall time.

## Running the Workflow

In accordance with the HPC policy, run the complete plotting workflow on a Slurm CPU node:

```bash
cd paperplots/0_ImputeTime
sbatch submit_imputation_runtime.sbatch
```

The workflow exports both vector PDF and 600 dpi PNG files by default. After the plotting job finishes, inspect it with:

```bash
squeue -j <JOB_ID>
tail -n 50 logs/imputation_runtime_<JOB_ID>.out
tail -n 50 logs/imputation_runtime_<JOB_ID>.err
```

## Figure Outputs

The following PDF/PNG pairs are generated under `figures/`:

```text
imputation_runtime_combined
imputation_runtime_lollipop
imputation_runtime_tensorflamingo_simulations_500x500
imputation_runtime_hicimpute_simulations_61x61
imputation_runtime_hicimpute_simulations_61x61_mean_per_dataset
imputation_runtime_hicimpute_simulations_61x61_scaling
imputation_runtime_summary_table
```

`combined` is a vertically arranged two-panel runtime bar chart. `lollipop` uses the same data and logarithmic axis to show medians, ranges, and individual runs. Separate plots are provided for the two data conditions. The mean-per-dataset runtime and input-size scaling plots are limited to HiCImpute simulations 61x61 because this dataset contains comparable T1/T2/T3 and 1k/2k/4k/7k conditions.

## Numerical Outputs

The following files are generated under `results/`:

```text
tensorflamingo_simulations_500x500_runtime_summary.csv
hicimpute_simulations_61x61_runtime_summary.csv
imputation_runtime_all_summary.csv
imputation_runtime_summary.tex
```

The tables and figures use the same loading, hardware classification, batch identification, and runtime aggregation functions, preventing discrepancies between plotted values and supplementary tables.
