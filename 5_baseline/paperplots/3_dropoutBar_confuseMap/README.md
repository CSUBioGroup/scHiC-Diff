**English** | [中文](README_zh.md)

# Dropout Recovery and SZ/DO Supplementary Diagnostics

This directory produces the revised simulated-data figures. The main-text figure contains only dropout-position PCC and MAE for HiCImputeData and FLAMINGOData. Structural-zero (SZ) versus dropout (DO) results are retained only as a HiCImputeData supplementary diagnostic because real scHi-C data do not provide observable SZ/DO labels and FLAMINGOData has no `GT == 0` class.

Run all commands from `paperplots/3_dropoutBar_confuseMap/`.

## Evaluation Scope

The continuous and classification diagnostics share GT, observed matrices, imputed outputs, cells, feature order, and raw contact scale, but answer different questions:

```text
classification universe = observed == 0
held-out DO             = observed == 0 and GT > 0
true SZ                 = observed == 0 and GT == 0

held-out DO is a strict subset of the classification universe
```

- The main-text PCC/MAE bars evaluate continuous recovery only at true DO, namely held-out positive contacts.
- F1, confusion matrices, ROC, and PR curves are supplementary-only SZ/DO diagnostics on all observed-zero HiCImputeData contacts.

Do not extend PCC/MAE to all observed-zero contacts. Otherwise, the large number of true SZ positions would dominate MAE, while PCC would mix numerical recovery with class discrimination.

## Official Code

```text
nature_plot_style.py
plot_dropout_pcc_mae_bar.py
calculate_sz_do_crossfit_metrics.py
plot_sz_do_metric_lines.py
plot_sz_do_confusion_matrices.py
plot_main_dropout_figure.py
calculate_sz_do_roc_pr.py
plot_sz_do_roc_pr_curves.py
submit_hicimputed_dropout_bar.sbatch
submit_flamingo_dropout_bar.sbatch
submit_hicimputed_sz_do_crossfit.sbatch
submit_hicimputed_sz_do_roc_pr.sbatch
submit_main_dropout_figure.sbatch
```

`plot_dropout_pcc_mae_bar.py` reads the authoritative raw-scale PCC/MAE/SCC summary tables from `../1_pccAndMae_all/`; it does not recalculate metrics during plotting.

`calculate_sz_do_crossfit_metrics.py` recalculates HiCImputeData SZ/DO operating-point metrics from the current official imputed matrices. The protocol is fixed as follows:

- Candidate positions are restricted to `observed == 0`; observed positives do not enter classification.
- SZ is the positive class; `prediction < threshold` means predicted SZ.
- The threshold maximizes MCC. Ties are resolved first by maximizing balanced accuracy and then by selecting the smaller threshold.
- Each threshold is applied only to a held-out fold that did not participate in threshold selection. Official metrics are calculated after merging five-fold out-of-fold (OOF) predictions.
- Cells are stratified by T1/T2/T3, with 100 cells per type and 20 cells of each type held out in each fold.
- The remaining 240 cells are pooled for calibration, and a threshold is selected for every `method x depth x fold` combination.

scHiC-Diff itself emits continuous imputed contact values and has no structural-zero classifier. The rule `prediction < threshold => predicted SZ` is a post-processing diagnostic. Thresholds are method- and depth-specific, not a universal value such as 0.5.

`calculate_sz_do_roc_pr.py` scans every distinct imputed value in the pooled five-fold held-out score partitions for each method, cell type, and depth. It writes the exact threshold scan, ROC-AUC, and PR-AUC (average precision). The ranking score is `-imputed_contact`, so ROC/PR comparisons are unaffected by different monotonic output scales across methods.

## Running the Workflow

Continuous-value PCC/MAE bar charts:

```bash
sbatch submit_hicimputed_dropout_bar.sbatch
sbatch submit_flamingo_dropout_bar.sbatch
sbatch submit_main_dropout_figure.sbatch
```

HiCImputeData supplementary SZ/DO operating-point metrics, complete ROC/PR scans, and supporting figures:

```bash
sbatch submit_hicimputed_sz_do_crossfit.sbatch
sbatch submit_hicimputed_sz_do_roc_pr.sbatch
```

To recalculate only metrics or redraw selected outputs:

```bash
MODE=metrics sbatch submit_hicimputed_sz_do_crossfit.sbatch
MODE=curves sbatch submit_hicimputed_sz_do_crossfit.sbatch
MODE=plots sbatch submit_hicimputed_sz_do_crossfit.sbatch
MODE=main sbatch submit_hicimputed_sz_do_crossfit.sbatch
```

## Official Outputs

```text
1_HiCImputedData/
├── HiCImputeData_SZ_DO_cell_folds.tsv
├── HiCImputeData_SZ_DO_5fold_thresholds.tsv
├── HiCImputeData_SZ_DO_5fold_fold_metrics.tsv
├── HiCImputeData_SZ_DO_5fold_OOF_metrics.tsv
├── HiCImputeData_SZ_DO_5fold_OOF_threshold_scan.tsv.gz
├── HiCImputeData_SZ_DO_5fold_OOF_curve_plot_points.tsv
├── HiCImputeData_SZ_DO_5fold_OOF_ROC_PR_AUC.tsv
├── figures/dropout_bar/
├── figures/sz_do_roc_pr/
├── figures/sz_do_metric_lines/
├── figures/sz_do_confusion_matrix/
└── logs/

2_FLAMINGOData/
├── FLAMINGOData_SZ_DO_eligibility_audit.tsv
├── figures/dropout_bar/
└── logs/

figures/main_dropout_figure/
└── Simulated_dropout_PCC_MAE_main_figure.pdf/png
```

The four five-fold tables contain 300 cell-fold assignments, 140 calibration thresholds, 420 held-out-fold metrics, and 84 merged OOF metrics, respectively. The line plots read the 84 OOF records directly. Confusion matrices are supplementary figures grouped by cell type into T1, T2, and T3 plots. In each plot, four rows represent 1K/2K/4K/7K and seven columns represent the seven imputation methods.

Matrix rows are true SZ followed by true DO; columns are predicted SZ followed by predicted DO. Values are normalized within each true class. The top-left value is SZ recall, while the bottom-right value is DO recall, which is equivalent to specificity when SZ is the positive class. SZ F1 shown in a matrix must match the value in the line plot for the same condition.

### Supplementary Operating-Point Metrics and Confusion Matrices

The line plots and confusion matrices use exactly the same evaluation definition. Both read `1_HiCImputedData/HiCImputeData_SZ_DO_5fold_OOF_metrics.tsv` directly and use the same five-fold cell-wise OOF predictions and thresholds. For a row-normalized confusion matrix with SZ as the positive class:

```text
                     predicted SZ       predicted DO
true SZ              SZ recall          1 - SZ recall
true DO              1 - DO recall      DO recall (= specificity)
```

The top-left cell is therefore SZ recall from the line plot, and the bottom-right cell is DO recall (specificity). SZ precision is `TP / (TP + FP)` and cannot be read from a single cell of a row-normalized matrix. SZ F1 is the harmonic mean of SZ precision and SZ recall. A method may obtain very high SZ recall by classifying most observed-zero contacts as SZ, while generating many false-positive SZ calls and therefore low SZ precision, DO recall, SZ F1, and MCC. SZ recall alone does not represent overall classification performance.

Macro-averaged scHiC-Diff results over the 12 `cell type x sequencing depth` conditions are:

```text
SZ precision                  = 0.994
SZ recall                     = 0.899
SZ F1                         = 0.944
DO recall (= specificity)     = 0.993
MCC                = 0.906
Balanced accuracy  = 0.946
```

These results indicate that scHiC-Diff uses a conservative but better-balanced decision rule. It identifies approximately 90% of true SZ positions while almost never classifying true DO as SZ. Its SZ recall is not the highest, but its macro-averaged SZ F1, SZ precision, DO recall, MCC, and balanced accuracy show better overall SZ/DO discrimination.

For example, under T1-1K:

```text
scHiC-Diff       SZ precision=1.000  SZ recall=0.890  SZ F1=0.942  DO recall=1.000
HiCImpute        SZ precision=0.176  SZ recall=1.000  SZ F1=0.300  DO recall=0.188
Tensor-FLAMINGO  SZ precision=0.153  SZ recall=0.984  SZ F1=0.265  DO recall=0.055
```

The near-one SZ recall values of HiCImpute and Tensor-FLAMINGO in this condition mainly result from overpredicting SZ; their confusion matrices classify 81% and 94% of true DO positions as SZ, respectively. The scHiC-Diff matrix is `[[0.89, 0.11], [0.00, 1.00]]`, showing that its advantage comes from controlling both false-negative SZ and false-positive SZ rather than maximizing SZ recall alone.

scHiC-Diff is not best in every individual condition. For example, HiCImpute reaches `SZ F1=0.984` in T1-4K, exceeding the scHiC-Diff value of `SZ F1=0.943`. The manuscript should therefore state that scHiC-Diff achieves the best or most robust overall SZ precision-recall/SZ-DO tradeoff, not that it is best under every condition. The red border and bold font used for scHiC-Diff in figures identify the proposed method and are not performance evidence.

### Revised Manuscript Placement and Caption

`plot_main_dropout_figure.py` creates the main-text figure at `figures/main_dropout_figure/Simulated_dropout_PCC_MAE_main_figure.pdf`. Panel A is HiCImputeData and Panel B is FLAMINGOData; both panels show only dropout-position PCC and MAE. The figure is rendered natively in Matplotlib, uses a shared seven-method legend, and contains no SZ/DO classification panel.

> **Figure X | Dropout recovery on two simulated scHi-C benchmarks.** PCC (higher is better) and MAE (lower is better) are calculated only at true dropout contacts (`observed = 0` and `GT > 0`) on the raw contact scale. Panel A shows HiCImputeData across cell types and sequencing depths; Panel B shows FLAMINGOData across the W and P simulation sweeps. Bars and error bars show cell-wise means and standard deviations.

The SZ/DO figures, including fixed-MCC operating-point lines, confusion matrices, full ROC/PR curves, and the ROC-AUC/AP heatmap, belong in the Supplementary Materials. Recommended Methods text and reviewer-response language are in `STRUCTURAL_ZERO_REVISION.md`.

## Why SZ/DO Can Be Plotted Only for HiCImputeData

Zeros introduced after downsampling do not by themselves establish structural zeros. Both labels must be defined jointly from the observed matrix and GT at the same position:

```text
candidate = observed == 0
true SZ   = observed == 0 and GT == 0
true DO   = observed == 0 and GT > 0
```

### HiCImputeData Explicitly Constructs Structural Zeros in GT

HiCImputeData uses `scHiC_simulate()` from the HiCImpute package to generate data from three-dimensional coordinates. The simulator first calculates the expected contact intensity `lambda` from bead-pair distance and covariates, then:

1. Selects positions below the `gamma` quantile of `lambda` as low-intensity candidates; this project uses `gamma=0.2`.
2. Randomly selects half of these candidates as true-zero candidates and explicitly sets the corresponding GT contacts to 0.
3. Uses `eta=0.8` so that most structural zeros are shared across all cells, while the remaining positions are randomly kept at zero by cell.
4. Applies Poisson sampling to the zero-containing `truecount`, followed by sequencing-depth downsampling to obtain observed matrices. Downsampling preserves existing true zeros while changing some `GT > 0` contacts into new observed zeros.

The observed-zero candidates in HiCImputeData therefore contain both SZ positions that were already zero in GT and DO positions lost through Poisson sampling or downsampling. For example, `K562_T1_1k` contains:

```text
GT entries               = 183000
GT == 0 (true SZ)         = 16509
observed-zero candidates  = 111495
true DO (GT > 0)          = 94986
```

The local simulation call and sequencing-depth downsampling record is at `../../../1_Dataset/1-HiCImpute_Simulation_Data/Documents/Double check the simulated K562 data in HiCImpute.R`; true-zero construction is implemented in `R/scHiC_simulate.R` in the official HiCImpute source.

### FLAMINGOData Has Positive GT Throughout and Can Produce Only DO

Although FLAMINGOData is also downsampled, its per-cell GT is a dense positive contact matrix generated by transforming the positive distance of each off-diagonal bead pair with `log1p(d^(-1/alpha))`. A finite positive distance remains greater than zero after this transformation, so GT contains no positions that can serve as structural zeros. Subsequent Poisson thinning removes some positive contacts from observed; all such positions satisfy `observed == 0 and GT > 0` and are DO rather than SZ.

In the H5AD files, `X` and `layers['counts']` store sparse downsampled observed values, while `layers['gt']` stores undownsampled per-cell GT. All seven official H5AD files have shape `1500 x 124750`, and every `layers['gt']` contains all `187125000` nonzero entries. The exact W=0.5 audit is:

```text
candidate contacts = 186186121
true SZ (GT == 0)  = 0
true DO (GT > 0)   = 186186121
```

Complete counts for all seven conditions are stored in `2_FLAMINGOData/FLAMINGOData_SZ_DO_eligibility_audit.tsv`.

The additional masking in the `heldout_masked` H5AD is applied to positive contacts in the original observed matrix, while `layers['gt']` remains positive. These held-out positions are also DO and do not create SZ.

Under the current `GT == 0` definition, FLAMINGOData therefore contains only one true class. F1, precision, recall, MCC, and a 2 x 2 confusion matrix cannot be calculated or plotted as SZ/DO binary-classification metrics. FLAMINGOData retains only continuous-value recovery evaluation by PCC/MAE at true-DO positions. Do not force the HiCImputeData workflow onto it.

Related generation and H5AD packaging code is located at:

- `../../../1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/1_RawData/3_fixed_flamnigoGen/2_generate_v2_4090_style.py`
- `../../../1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/prepare_hierarchical_h5ad.py`

An independent binary structural-zero ground-truth mask must be supplied by the simulator before classification evaluation can be performed for FLAMINGOData. Alternatively, an effective-SZ label based on a predefined and justified `GT <= tau` could be introduced. The latter would be a new low-contact classification task and must not be interpreted jointly with the HiCImputeData `GT == 0` SZ/DO results.
