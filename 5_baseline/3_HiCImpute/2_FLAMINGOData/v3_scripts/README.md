# HiCImpute on FLAMINGO v3: preprocessing and imputation guide

This directory contains the corrected FLAMINGO v3 pipeline for running the HiCImpute `MCMCImpute()` algorithm.

The most important rule is:

> HiCImpute expects feature vectors in **R `upper.tri` column-major order**, but FLAMINGO h5ad/evaluation files use **numpy `np.triu_indices` row-major order**. Always reorder inputs before calling HiCImpute, and always restore outputs before evaluation.

## Paths

```text
v3_scripts/
  v3_prepare_hicimpute_flamingo.py   # raw FLAMINGO matrices -> HiCImpute input binaries
  v3_run_hicimpute_flamingo.R        # HiCImpute R imputation + output collection
  v3_restore_hicimpute_r_to_triu.py  # R-order Impute_All -> canonical numpy triu NPZ
  v3_submit_hicimpute_flamingo.sbatch
  v3_submit_restore_hicimpute_r_to_triu.sbatch
  README.md

v3_inputData/<dataset>/
  schic.bin
  expected.bin
  bulk.bin
  feature_order.npy
  obs_names.txt
  var_names.txt
  metadata.json
  .complete

v3_outputData/
  bin/
  rds/
  npz_lower_tri/
  npz_triu_corrected/
  metrics/
```

Raw FLAMINGO v3 data source:

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/1_RawData/3_fixed_flamnigoGen
```

Processed h5ad / evaluation GT source:

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets
```

## Current datasets

The v3 pipeline uses 7 datasets:

```text
v3_hybrid_W0p5_500cells_level0
v3_hybrid_W0p6_500cells_level0
v3_hybrid_W0p7_500cells_level0
v3_hybrid_W0p7_500cells_level0_r0p01
v3_hybrid_W0p7_500cells_level0_r0p05
v3_hybrid_W0p8_500cells_level0
v3_hybrid_W0p9_500cells_level0
```

Each raw dataset has:

```text
sim_<dataset>/
  gt_contact_data/type_{1..3}_cell_{1..500}_contact.txt
  downsampled_contact_data/type_{1..3}_cell_{1..500}.txt
  params.json
```

Each contact file is a dense 500 x 500 symmetric matrix with zero diagonal.

Cell order used by this pipeline is:

```text
type_1_cell_1, ..., type_1_cell_500,
type_2_cell_1, ..., type_2_cell_500,
type_3_cell_1, ..., type_3_cell_500
```

This matches h5ad obs indices 0..1499.

## Why the previous v3 preprocessing was wrong

The previous v3 pipeline read h5ad features directly. h5ad `var_names` are ordered as:

```text
chrFLAMINGO_0_1, chrFLAMINGO_0_2, chrFLAMINGO_0_3, ...
```

This is numpy row-major upper-triangle order:

```python
iu, ju = np.triu_indices(n, k=1)
```

For n=5, numpy row-major order is:

```text
(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
```

HiCImpute internally reconstructs matrices with R `upper.tri`:

```r
m[upper.tri(m, diag = FALSE)] <- single[, k]
```

R uses column-major order. For n=5, R `upper.tri` order is:

```text
(0,1), (0,2), (1,2), (0,3), (1,3), (2,3), (0,4), (1,4), (2,4), (3,4)
```

If numpy row-major features are passed directly to HiCImpute, feature positions are scrambled. This destroys the spatial neighborhoods used by HiCImpute functions such as `neivar()` and `correctfac()`, causing very poor imputation metrics.

## Correct feature-order conversion

Let `counts_numpy` and `gt_numpy` be cells x features in numpy row-major upper-triangle order.

The correct conversion to HiCImpute input order is:

```python
iu, ju = np.triu_indices(n_beads, k=1)
order = np.lexsort((iu, ju))
counts_r = counts_numpy[:, order]
gt_r = gt_numpy[:, order]
bulk_r = counts_r.sum(axis=0)
```

Here:

```text
order[k] = numpy row-major feature index that belongs at R upper.tri column-major position k
```

This `order` is saved as:

```text
v3_inputData/<dataset>/feature_order.npy
```

## Correct binary layout for R

HiCImpute receives `scHiC` as a matrix with shape:

```text
features x cells = 124750 x 1500
```

R reads binary data with:

```r
matrix(readBin(path, "double", n = n_features * n_cells),
       nrow = n_features, ncol = n_cells)
```

Therefore the file must be written in column-major layout: all features for cell 0, then all features for cell 1, etc.

In Python, do not rely on `np.asfortranarray(...).tofile()`. `tofile()` writes C-order bytes. Use explicit F-order ravel:

```python
feats_by_cells = cells_by_features.T.astype("<f8")
feats_by_cells.ravel(order="F").tofile(path)
```

This is implemented in `v3_prepare_hicimpute_flamingo.py`.

## Preprocessing command

Prepare all 7 v3 inputs:

```bash
/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python3 \
  /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_scripts/v3_prepare_hicimpute_flamingo.py \
  --overwrite
```

Prepare selected datasets:

```bash
/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python3 \
  /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_scripts/v3_prepare_hicimpute_flamingo.py \
  --datasets v3_hybrid_W0p7_500cells_level0 \
  --overwrite
```

Expected file sizes per dataset:

```text
schic.bin      1,497,000,000 bytes  # 124750 * 1500 * 8
expected.bin   1,497,000,000 bytes
bulk.bin             998,000 bytes  # 124750 * 8
feature_order.npy    998,128 bytes
```

A dataset is ready when `.complete` exists and all expected files are present.

## Preprocessing validation

Minimal validation:

```bash
python3 - <<'PY'
from pathlib import Path
root = Path('/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_inputData')
expected = {
    'schic.bin': 124750*1500*8,
    'expected.bin': 124750*1500*8,
    'bulk.bin': 124750*8,
}
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    ok = True
    msgs = []
    for name, size in expected.items():
        p = d/name
        if not p.exists() or p.stat().st_size != size:
            ok = False
            msgs.append(f'{name}: bad size')
    for name in ['feature_order.npy','metadata.json','obs_names.txt','var_names.txt','.complete']:
        if not (d/name).exists():
            ok = False
            msgs.append(f'{name}: missing')
    print(d.name, 'OK' if ok else 'BAD', '; '.join(msgs))
PY
```

R round-trip validation for one dataset:

```bash
/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/Rscript -e '
stem <- "v3_hybrid_W0p7_500cells_level0"
base <- "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_inputData"
rawbase <- "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/1_RawData/3_fixed_flamnigoGen"
schic <- readBin(file.path(base, stem, "schic.bin"), "double", n=124750, endian="little")
m <- matrix(0, 500, 500)
m[upper.tri(m, diag=FALSE)] <- schic
raw <- as.matrix(read.table(file.path(rawbase, paste0("sim_", stem), "downsampled_contact_data/type_1_cell_1.txt")))
cat("counts match:", isTRUE(all.equal(m[upper.tri(m)], raw[upper.tri(raw)])), "\n")
'
```

The result must be:

```text
counts match: TRUE
```

## Imputation command

Run one dataset directly with R:

```bash
/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/Rscript \
  /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_scripts/v3_run_hicimpute_flamingo.R \
  --dataset v3_hybrid_W0p7_500cells_level0 \
  --input-root /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_inputData \
  --output-root /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_outputData \
  --niter 5000 \
  --burnin 1000 \
  --mc-cores 40 \
  --seed 1234 \
  --python /public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python3
```

Do not run long HiCImpute jobs on a login node. Submit through SLURM.

## SLURM submission

Use the paid account:

```text
--account=pi_limin_r
```

The current submit script is:

```text
v3_scripts/v3_submit_hicimpute_flamingo.sbatch
```

For formal runs, prefer high CPU count and enough walltime. Example:

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_scripts
NITER=50000 BURNIN=5000 sbatch \
  --array=0-6 \
  --partition=cpuQ \
  --qos=cpuq \
  --account=pi_limin_r \
  --time=1-00:00:00 \
  --cpus-per-task=40 \
  --mem=80G \
  --job-name=v3_hicimpute_formal \
  v3_submit_hicimpute_flamingo.sbatch
```

If `cpuQ` cannot schedule 40 CPUs, reduce `--cpus-per-task` only if walltime is increased. A 16-core run with `niter=5000` timed out at 6 hours on one dataset, so 16 cores are not recommended for formal runs unless walltime is much longer.

## Output order and evaluation

HiCImpute produces `result$Impute_All` in R column-major feature order. Before saving NPZ for evaluation, the R script restores numpy row-major order:

```r
impute_all_numpy <- impute_all_r[order + 1L, , drop = FALSE]
pred_cells_by_features <- t(impute_all_numpy)
```

The legacy R collection step saved files under `npz_lower_tri/`, but the feature
vector is semantically NumPy row-major `triu(k=1)`, not `tril`. The canonical
files used by the current heatmap pipeline are restored directly from
`Impute_All.bin` with `feature_order.npy`:

```text
v3_outputData/npz_triu_corrected/<dataset>_hicimpute_Impute_All_triu.npz
```

Its shape is:

```text
1500 x 124750  # cells x features, numpy row-major upper-triangle order
```

This order matches h5ad `layers['gt']` and the official evaluation script.
Generate all seven canonical files on CPU nodes with:

```bash
sbatch v3_scripts/v3_submit_restore_hicimpute_r_to_triu.sbatch
```

Do not infer feature order from an old directory or filename containing
`lower_tri`; the canonical semantic order is `triu(k=1)`.

Evaluate raw PCC/MAE:

```bash
/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python3 \
  /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/1_pccAndMae_all/13_cal_FLAMINGO_Baseline_metrics.py \
  --v3-lower-tri-pred-dir /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_outputData/npz_lower_tri \
  --v3-lower-tri-gt-dir /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets \
  --v3-lower-tri-output-csv /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_outputData/metrics/HiCImpute_FLAMINGO_v3_raw.csv
```

Evaluate log1p PCC/MAE:

```bash
/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python3 \
  /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/1_pccAndMae_all/13_cal_FLAMINGO_Baseline_metrics.py \
  --v3-lower-tri-pred-dir /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_outputData/npz_lower_tri \
  --v3-lower-tri-gt-dir /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets \
  --v3-log1p \
  --v3-lower-tri-output-csv /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_outputData/metrics/HiCImpute_FLAMINGO_v3_log1p.csv
```

## Adapting to a new dataset

For a new dataset, ensure these conditions:

1. Each cell has an observed contact matrix and a GT contact matrix.
2. Matrices are square, symmetric, same bead count, zero diagonal.
3. Cells have a deterministic order shared by observed, GT, output, and evaluation.
4. Extract upper-triangle features in numpy row-major order first:

   ```python
   vec = mat[np.triu_indices(n_beads, k=1)]
   ```

5. Convert to HiCImpute input order before writing binaries:

   ```python
   iu, ju = np.triu_indices(n_beads, k=1)
   order = np.lexsort((iu, ju))
   counts_r = counts_numpy[:, order]
   gt_r = gt_numpy[:, order]
   bulk_r = counts_r.sum(axis=0)
   ```

6. Write `features x cells` binaries in true R column-major byte order:

   ```python
   feats_by_cells = counts_r.T.astype('<f8')
   feats_by_cells.ravel(order='F').tofile('schic.bin')
   ```

7. Run HiCImpute with:

   ```r
   MCMCImpute(scHiC = schic, bulk = bulk, expected = expected,
              startval = startval, n = n_beads,
              mc.cores = mc_cores,
              cutoff = 0.5,
              niter = niter,
              burnin = burnin)
   ```

8. Restore output to numpy row-major order before evaluation:

   ```r
   impute_all_numpy <- result$Impute_All[order + 1L, , drop = FALSE]
   pred_cells_by_features <- t(impute_all_numpy)
   ```

9. Save evaluation NPZ as cells x features in numpy row-major order.

## Common failure modes

- Passing h5ad feature order directly to HiCImpute. This is wrong for FLAMINGO v3.
- Forgetting to restore HiCImpute output to numpy row-major order before evaluation.
- Writing binary files with `np.asfortranarray(...).tofile()`. This still writes C-order bytes. Use `ravel(order='F').tofile()`.
- Running long HiCImpute jobs on login nodes.
- Using too few CPU cores or too short a walltime. A 16-core, 6-hour `niter=5000` run timed out for one 1500-cell dataset.
- Evaluating an R-order output NPZ against h5ad GT. This gives artificially poor PCC.

## Current status

As of this README update, all 7 FLAMINGO v3 input datasets have been prepared under:

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/3_HiCImpute/2_FLAMINGOData/v3_inputData
```

The prepared input files have been validated for size and one R round-trip check.
