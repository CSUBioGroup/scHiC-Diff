# scVI-3D: Contact Matrix Imputation for scHi-C Data

This directory contains a complete, portable pipeline for running the scVI-3D
(Zheng et al., 2022) imputation method on single-cell Hi-C contact data.
It was developed and validated on the FLAMINGO v3 paramsweep simulation
datasets, but the scripts are designed to work with **any single-chromosome
contact matrix dataset** after appropriate preprocessing.

---

## Table of Contents

1. [How scVI-3D Works](#how-scvi-3d-works)
2. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
3. [Input Format Specification](#input-format-specification)
4. [Running the Pipeline](#running-the-pipeline)
5. [Adapting to New Datasets](#adapting-to-new-datasets)
6. [Common Pitfalls](#common-pitfalls)
7. [SLURM Submission](#slurm-submission)
8. [Performance & Resources](#performance--resources)
9. [Validated Results](#validated-results)

---

## How scVI-3D Works

scVI-3D (Zheng et al. 2022) was originally designed for denoising and
imputing scHi-C contact matrices. The algorithm operates as follows:

### Algorithm Overview

1. **Read per-cell contacts** — each cell is represented as a list of
   contacts: `chrA, binA_bp, chrB, binB_bp, count`

2. **Build band matrices** — For each diagonal distance *d* (band), the
   2D contact matrix is flattened. Band *d* consists of positions
   *k* = 0, 1, ..., *n_bins - d - 1*, where position *k* represents the
   contact at genomic locus pair (*k*, *k + d*). All cells are stacked
   into a *(n_cells, n_bins - d)* matrix per band.

3. **Train scVI VAE per band** — For each band matrix independently:
   - Create an `AnnData` object
   - Train `scvi.model.SCVI` (400 epochs, 100 latent dimensions)
   - Compute `bandDepth = mean(total counts per cell)` as the
     library-size reference
   - Impute with `model.get_normalized_expression(library_size=bandDepth)`

4. **Reconstruct full matrix** — Imputed band values are mapped back to
   upper-triangle matrix positions (binA, binB) for each cell.

### Why Band Matrices?

scHi-C contact matrices are symmetric and sparse. Rather than modeling
the full N×N matrix (250K entries for N=500), scVI-3D decomposes it into
*band matrices* — one per genomic distance. This:
- Reduces each scVI training problem to at most N features
- Exploits the 1D structure along each band
- Allows band-specific library-size normalization

### Key Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `resolution` | 1,000,000 (1Mb) | Genomic resolution in bp |
| `n_bins` | 500 | Number of genomic bins (= chromosome_size_bp / resolution + 1) |
| `band_max` | `"whole"` | Maximum band distance; `"whole"` = all bands (1 to n_bins−1) |
| `n_latent` | 100 | scVI latent space dimension |
| `max_epochs` | 400 (scVI default) | Training epochs per band |

---

## Data Preprocessing Pipeline

The pipeline has three stages: **Prepare** → **Impute** → **Evaluate**.

### Stage 0: Understand Your Source Data

scVI-3D works with **contact counts** — non-negative integers
representing interaction frequencies (e.g., read-pair counts in real
Hi-C, or Poisson-thinned IF values in simulations). The key properties
of the source data you need to know:

| Property | Where to find it | Example (FLAMINGO) |
|----------|-----------------|---------------------|
| `n_bins` | Matrix dimension (N×N → N) | 500 |
| `n_cells` | Number of per-cell matrices | 1500 |
| `n_features` | n_bins × (n_bins − 1) / 2 | 124,750 |
| `cell_type` | Cell annotation | T1, T2, T3 |
| `resolution` | Genomic bin size (bp) | 1,000,000 (1 Mb) |

The contact matrix is assumed to be **symmetric** with **zero diagonal**
(intra-chromosomal only, no self-contacts).

### Stage 1: Prepare Input (`prepare_scvi3d_input.py`)

**Purpose**: Convert raw per-cell contact matrices into the 5-column
tab-separated format that scVI-3D expects.

**What it does**, step by step:

```
Step 1.1: Read each cell's N×N contact matrix
Step 1.2: Extract upper-triangle (i < j) non-zero contacts
Step 1.3: Convert float IF → integer counts via ceil()  ← CRITICAL
Step 1.4: Convert bin indices to genomic coordinates:
          binA_bp = i * resolution
          binB_bp = j * resolution
Step 1.5: Write one file per cell:
          chrA  binA_bp  chrB  binB_bp  count
Step 1.6: Write genome.txt:
          chrom  (n_bins - 1) * resolution    ← formula matters!
Step 1.7: Write cell_summary.txt:
          name  batch  cell_type
Step 1.8: Sort cells by (type_id, cell_num) → matches ground-truth order
```

**Why ceil()?** scVI uses a Zero-Inflated Negative Binomial (ZINB)
likelihood whose support is **non-negative integers**. Feeding
floating-point values causes the warning *"The value argument must
be within the support of the distribution"* and produces unreliable
imputations. `np.ceil()` converts e.g. IF=2.566 → count=3 with
negligible information loss (PCC(original, rounded) ≈ 0.992).

**Why `(n_bins - 1) * resolution` for genome.txt?** scVI-3D computes
`n_bins = size_bp // resolution + 1` internally. To get `n_bins = N`:

```
size_bp // resolution + 1 = N
size_bp // resolution = N - 1
size_bp = (N - 1) * resolution
```

| n_bins | resolution | correct size_bp | wrong size_bp → actual n_bins |
|--------|-----------|-----------------|-------------------------------|
| 500 | 1Mb | 499,000,000 | 500,000,000 → 501 ❌ |
| 61 | 1Mb | 60,000,000 | 61,000,000 → 62 ❌ |

A 1-bin mismatch causes band 500 (contact at (0, 500)) to be
silently dropped during output collection.

**Supporting multiple source formats** — the current script reads from
FLAMINGO raw dense `.txt` files (500×500 matrix, tab-separated). To
support other formats, modify the `prepare_dataset()` function:

| Source format | What to change |
|---------------|---------------|
| h5ad with CSR `layers['counts']` | Replace `np.loadtxt()` with CSR row reading |
| `.npz` dense matrix | Use `np.load()` instead of `np.loadtxt()` |
| `.csv` with columns (i, j, count) | Parse directly; no matrix reconstruction needed |

The **output format** is always the same (see [Input Format Specification](#input-format-specification)).

### Stage 2: Impute (`run_scvi3d_flamingo.py`)

**Purpose**: Self-contained scVI-3D implementation. Reads the prepared
per-cell txt files, builds band matrices, trains scVI per band,
imputes, and collects the final feature matrix.

**What it does**, step by step:

```
Step 2.1: Read all per-cell contact txt files
Step 2.2: For each cell, rescale: binA = bp // resolution, diag = |binB - binA|
Step 2.3: Build band matrices:
          For each band distance d (1 to n_bins-1 or up to band_max):
            band_matrix[d] = (n_cells, n_bins - d) where
            column k = contact (k, k+d) for that cell
Step 2.4: For each band matrix:
            - Create sc.AnnData
            - scvi.model.SCVI.setup_anndata()
            - model = SCVI(adata, n_latent=100)
            - model.train(use_gpu=True)          ← 400 epochs default
            - bandDepth = matrix.sum(axis=1).mean()
            - imputed = model.get_normalized_expression(library_size=bandDepth)
Step 2.5: Convert imputed band values back to contact coordinates
          and write per-cell output txt files
Step 2.6: Collect: read all per-cell output txts, reconstruct N×N matrix,
          extract upper-triangle features via triu_indices(N, k=1),
          stack into (n_cells, n_features) CSR sparse matrix,
          save as <stem>_scVI3D_imputed.npz
```

**Band training order**: bands are trained **sequentially** (band 1,
then band 2, ...). Each band is a separate scVI model training session
on GPU. This is the bottleneck — training is GPU-bound, not CPU-bound.

**GPU memory usage**: each band's scVI model is tiny (<1 GB VRAM)
because it's a single-hidden-layer VAE (128 units). The data per band
is at most (1500 × 500) floats ≈ 6 MB. A V100 (32 GB) could
theoretically train 30+ bands in parallel, but the current
implementation serializes them for safety.

### Stage 3: Evaluate (`evaluate_scvi3d_flamingo.py`)

**Purpose**: Compute cell-wise PCC (Pearson) and MAE against the
ground truth, on log1p scale, with all/observed/heldout masks.

```
Step 3.1: Load prediction npz (n_cells, n_features)
Step 3.2: Load ground truth from h5ad layers['gt']
Step 3.3: Load observed mask from h5ad layers['counts'] > 0
Step 3.4: For each cell:
            gt_mask = gt > 0
            obs_mask = observed > 0 (positions with known contacts)
            held_mask = gt_mask & ~obs_mask (positions to impute)
            PCC_all  = corr(log1p(pred[gt_mask]),  log1p(gt[gt_mask]))
            PCC_obs  = corr(log1p(pred[obs_mask]), log1p(gt[obs_mask]))
            PCC_held = corr(log1p(pred[held_mask]),log1p(gt[held_mask]))
Step 3.5: Aggregate mean ± std across all cells
```

**Log1p transform**: Applied because contact counts span several orders
of magnitude and the distribution is right-skewed. `log1p(x) = log(1+x)`
compresses the range and makes PCC more meaningful.

---

## Input Format Specification

scVI-3D expects three files per dataset in a single directory:

### 1. Per-cell contact files: `cell_1.txt` ... `cell_N.txt`

Each file has 5 tab-separated columns, **no header row**:

```
chrA  binA_bp  chrB  binB_bp  count
```

| Column | Type | Description |
|--------|------|-------------|
| chrA | string | Chromosome name (e.g., `chr1`, `chrFLAMINGO`) |
| binA_bp | integer | Genomic coordinate of bin A in base pairs (e.g., `0`, `1000000`) |
| chrB | string | Same chromosome as chrA (intra-chromosomal only) |
| binB_bp | integer | Genomic coordinate of bin B in base pairs |
| count | integer | Contact count (**must be integer** for scVI) |

**Rules**:
- Only **upper triangle** contacts: `binA_bp < binB_bp` (diagonal excluded)
- All contacts on the **same chromosome** (`chrA == chrB`)
- `count` must be **non-negative integer** (use `ceil()` for float IF values)
- Files are numbered `cell_1.txt` through `cell_N.txt`
- Order must match the ground truth row order (for evaluation alignment)

### 2. Genome size file: `genome.txt`

One line, tab-separated:

```
<chromosome>  <size_bp>
```

**Formula**: `size_bp = (n_bins - 1) * resolution`

Example for N=500 bins at 1Mb resolution: `chrFLAMINGO  499000000`

### 3. Cell summary file: `cell_summary.txt`

Tab-separated with header row:

```
name    batch    cell_type
cell_1.txt    batch1    T1
cell_2.txt    batch1    T1
...
cell_N.txt    batch1    T3
```

- `name`: filename of the per-cell contact file
- `batch`: batch label (use `batch1` if no batch effect)
- `cell_type`: cell type label

---

## Running the Pipeline

### Quick Start (SLURM)

```bash
# Stage 1: Prepare input (CPU, ~1 min)
sbatch v3_scripts/submit_prepare_all_cpu.sbatch

# Stage 2: Impute (GPU, one per dataset, array 0-6)
sbatch v3_scripts/submit_scvi3d_flamingo.sbatch

# Stage 3: Evaluate (CPU, ~3 min for 7 datasets)
sbatch v3_scripts/submit_eval_scvi3d_cpu.sbatch
```

### Standalone (no SLURM)

```bash
# Stage 1: Prepare one dataset
python v3_scripts/prepare_scvi3d_input.py \
    --raw-root <dir with FLAMINGO raw sim_*> \
    --input-root v3_inputData \
    --datasets v3_hybrid_W0p7_500cells_level0_r0p01 \
    --workers 20 --overwrite

# Stage 2: Impute one dataset
python v3_scripts/run_scvi3d_flamingo.py \
    --input-dir v3_inputData/v3_hybrid_W0p7_500cells_level0_r0p01 \
    --output-dir v3_outputData/impute_work/<stem> \
    --collect-dir v3_outputData/npz_lower_tri \
    --stem v3_hybrid_W0p7_500cells_level0_r0p01 \
    --resolution 1000000 --n-bins 500 \
    --band-max whole --n-latent 100 \
    --gpu --overwrite

# Stage 3: Evaluate one dataset
python v3_scripts/evaluate_scvi3d_flamingo.py \
    --pred-dir v3_outputData/npz_lower_tri \
    --gt-dir <dir with *_scdiff2.h5ad> \
    --output-csv v3_outputData/metrics/scVI3D_metrics.csv \
    --workers 20 --log1p
```

---

## Adapting to New Datasets

The scripts are designed to be portable. Here is what you need to
change for a new single-chromosome contact matrix dataset.

### Step 1: Determine dataset properties

```python
n_bins     = <matrix dimension>          # e.g., 200 for a 200×200 matrix
n_cells    = <number of cells>
resolution = <bin size in bp>            # e.g., 1_000_000 for 1Mb
chrom      = <chromosome name>           # e.g., "chr1"
cell_types = <list of cell type labels>  # e.g., ["T1"]*300 + ["T2"]*300
```

### Step 2: Prepare input data

If your source data is already in one of these formats:

| Source Format | How to convert |
|--------------|----------------|
| Dense N×N matrix per cell (like FLAMINGO raw) | Use `prepare_scvi3d_input.py` as-is; set `N_BINS` and `CHROM_NAME` |
| CSR sparse matrix (h5ad `layers['counts']`) | Modify `prepare_dataset()` to read CSR rows via `indptr/indices/data` |
| (i, j, count) triplets per cell | Write directly to 5-column format |
| `.npz` dense array (n_cells, n_features) | Reconstruct N×N, extract upper-triangle |

The prepare script must produce:

```
<input_dir>/
  cell_1.txt ... cell_N.txt    ← 5-column: chr binA_bp chr binB_bp count
  genome.txt                   ← chrom \t (n_bins - 1) * resolution
  cell_summary.txt             ← name \t batch \t cell_type
```

### Step 3: Set parameters for imputation

| Parameter | How to set |
|-----------|-----------|
| `--n-bins` | Matrix dimension N |
| `--resolution` | Bin size in bp |
| `--band-max` | Always `"whole"` for full imputation |
| `--n-latent` | 100 is standard for scHi-C; reduce for smaller datasets |
| `--max-epochs` | Default 400; reduce for quick testing |

### Step 4: Adapt evaluation

The evaluation script reads h5ad `layers['gt']` for ground truth. For
new datasets, either:
- Store GT in the same h5ad format
- Modify `evaluate_scvi3d_flamingo.py` to read GT from your format
  (the `_compute_cell()` function loads GT from h5ad)

### Example: preparing a 200×200 dataset

```python
# In prepare_scvi3d_input.py, change the constants:
N_BINS = 200                       # was 500
CHROM_NAME = "chr1"                # was "chrFLAMINGO"
RESOLUTION = 1_000_000             # 1Mb

# genome.txt will be: chr1 \t 199000000  (= 199 × 1Mb)
# n_features = 200 * 199 / 2 = 19900
```

Then run the same pipeline commands with `--n-bins 200` on the impute
script.

---

## Common Pitfalls

### 1. Float counts (most common)

**Symptom**: `UserWarning: The value argument must be within the support
of the distribution` in scVI training logs. Unreliable (near-zero) PCC.

**Cause**: scVI's ZINB likelihood requires integer counts. FLAMINGO
simulation produces float IF values.

**Fix**: Apply `np.ceil()` or `np.round()` to convert floats to integers
during the prepare step. `ceil()` preserves all non-zero contacts;
`round()` may lose very small values (<0.5).

### 2. Wrong genome.txt size

**Symptom**: `n_bins` computed by scVI-3D is off by 1 (e.g., 501 instead
of 500). Extra band 500 produces contacts at (0, 500) that are silently
dropped during output collection.

**Cause**: `genome.txt` uses `n_bins * resolution` instead of
`(n_bins - 1) * resolution`.

**Fix**: Use the formula `size_bp = (n_bins - 1) * resolution`.

### 3. band_max too small

**Symptom**: Only a fraction of features are non-zero in the output npz
(e.g., 4945 / 124750 = 4% with band_max=10). PCC is very low.

**Cause**: `band_max` limits how many diagonal bands are imputed.
Remaining features are left as zero.

**Fix**: Always use `--band-max whole` for production runs. Use small
values (e.g., 10 or 50) only for quick testing.

### 4. Cell ordering mismatch

**Symptom**: PCC is lower than expected even with correct input format.

**Cause**: The cell order in `cell_1.txt ... cell_N.txt` does not match
the row order of the ground truth matrix. Evaluation compares row i of
prediction with row i of GT — if they are different cells, PCC is wrong.

**Fix**: Ensure cells are sorted by `(type_id, cell_num)` during
preparation so that all type-1 cells come first, then type-2, etc.
Verify: `cell_summary.txt` should show T1×500, then T2×500, then T3×500.

### 5. Missing diagonal exclusion

**Symptom**: binA == binB contacts appear in output (self-contacts).

**Cause**: Diagonal contacts (i, i) were not excluded during preparation.

**Fix**: Always use `k=1` in `np.triu_indices(N, k=1)` to exclude
the main diagonal.

---

## SLURM Submission

### Available Configurations

| Script | Partition | Account | Resources | Time |
|--------|-----------|---------|-----------|------|
| `submit_prepare_all_cpu.sbatch` | cpuQ | pi_limin_r | 48 cores, 120 GB | 1 h |
| `submit_scvi3d_flamingo.sbatch` | gpu4Q | pi_limin_r | 1 GPU, 10 cores, 80 GB | 1 day |
| `submit_scvi3d_7in2gpu.sbatch` | gpu2Q | pi_limin_r | 2 GPUs, 20 cores, 200 GB | 1 day |
| `submit_eval_scvi3d_cpu.sbatch` | cpuQ | pi_limin_r | 20 cores | 2 h |

### Quick commands

```bash
# Check GPU availability
sinfo -p gpu2Q,gpu4Q -o "%P %n %G %C %t" | grep -E "idle|mix"

# Check own jobs
squeue -u $USER

# Cancel all own jobs on a partition
scancel -u $USER -p gpu2Q
```

### Environment

```bash
# Activate the correct conda environment
micromamba activate 2_schic-scvi-3d

# Or use the full path to Python
/public/home/hpc254701055/micromamba/envs/2_schic-scvi-3d/bin/python
```

Required packages:
```
Python 3.8
scvi-tools == 0.14.6
scanpy >= 1.7
anndata >= 0.7
numpy, scipy, pandas, scikit-learn, h5py, joblib
```

---

## Performance & Resources

### Timing (per dataset, 1500 cells × 500 bins)

| Stage | CPU | GPU | Wall Time |
|-------|-----|-----|-----------|
| Prepare | 48 cores | — | ~8 s |
| Impute (band_max=50) | 10 cores | 1 × V100 | ~40 min |
| Impute (band_max=whole) | 10 cores | 1 × V100 | ~5 h |
| Evaluate | 20 cores | — | ~20 s |

### GPU Memory

Each band's scVI training uses **<1 GB VRAM**:
- Model parameters (1-layer VAE): ~0.5 MB
- Training data (1500 × 500 floats): ~6 MB
- CUDA context + PyTorch overhead: ~500–800 MB
- **Total per process: <1 GB**

A V100 has 32 GB. The bottleneck is GPU compute time (400 epochs per
band × 499 bands), not memory. Parallelizing bands on a single GPU is
theoretically possible but requires careful CUDA context management.

### Disk Usage

| Artifact | Size | Keep? |
|----------|------|-------|
| v3_inputData (7 datasets) | ~714 MB | Yes (can regenerate) |
| v3_outputData/npz_lower_tri (7 npz) | ~11 GB | **Yes (final results)** |
| v3_outputData/impute_work | ~14 GB (intermediate) | **No (delete after impute)** |

Delete intermediate files after imputation:
```bash
rm -rf v3_outputData/impute_work
```

---

## Validated Results

The pipeline was validated on 7 FLAMINGO v3 paramsweep datasets
(1500 cells each, 500 beads, chrFLAMINGO) with `band_max="whole"`
and `ceil()` integer conversion.

### Dataset Descriptions

| Dataset | W (structure similarity) | Retention | Noise |
|---------|--------------------------|-----------|-------|
| `v3_hybrid_W0p5_500cells_level0` | 0.5 | ~0.5% | level_0 (0.0) |
| `v3_hybrid_W0p6_500cells_level0` | 0.6 | ~0.5% | level_0 (0.0) |
| `v3_hybrid_W0p7_500cells_level0` | 0.7 | ~0.5% | level_0 (0.0) |
| `v3_hybrid_W0p7_500cells_level0_r0p01` | 0.7 | ~1% | level_0 (0.0) |
| `v3_hybrid_W0p7_500cells_level0_r0p05` | 0.7 | ~5% | level_0 (0.0) |
| `v3_hybrid_W0p8_500cells_level0` | 0.8 | ~0.5% | level_0 (0.0) |
| `v3_hybrid_W0p9_500cells_level0` | 0.9 | ~0.5% | level_0 (0.0) |

### Results (log1p scale, cell-wise Pearson PCC)

| Dataset | PCC All | PCC Observed | PCC Heldout | MAE All |
|---------|---------|-------------|------------|---------|
| W0p5 | 0.112 ± 0.013 | 0.373 ± 0.060 | 0.118 ± 0.013 | 1.429 |
| W0p6 | 0.116 ± 0.013 | 0.358 ± 0.055 | 0.126 ± 0.012 | 1.428 |
| W0p7 | 0.125 ± 0.013 | 0.352 ± 0.049 | 0.139 ± 0.011 | 1.424 |
| **W0p7_r0p01** | **0.179 ± 0.017** | 0.318 ± 0.029 | **0.205 ± 0.011** | 1.406 |
| **W0p7_r0p05** | **0.427 ± 0.016** | 0.329 ± 0.025 | **0.452 ± 0.014** | 1.252 |
| W0p8 | 0.131 ± 0.013 | 0.350 ± 0.045 | 0.149 ± 0.010 | 1.426 |
| W0p9 | 0.135 ± 0.013 | 0.353 ± 0.044 | 0.155 ± 0.010 | 1.421 |

### Key Observations

1. **Higher retention → better imputation**: W0p7_r0p05 (5% observed)
   achieves PCC_all=0.427 and PCC_heldout=0.452, significantly
   outperforming the ~0.5% retention datasets (PCC_all ~0.11-0.14).

2. **Observed PCC is consistent** (~0.32-0.37 across all datasets),
   indicating scVI-3D can recover the observed contacts well
   regardless of retention rate.

3. **Heldout PCC tracks total PCC**: The model's ability to predict
   unobserved contacts is the primary driver of overall performance.

4. **W (structure similarity) has minor effect** at low retention:
   PCC_all increases from 0.112 (W=0.5) to 0.135 (W=0.9), suggesting
   slightly better structure helps but doesn't dominate.

### Output Format

The final `*_scVI3D_imputed.npz` is a CSR sparse matrix of shape
`(n_cells, n_features)` where:
- Row *i* = cell *i* (matches ground truth row order)
- Column *k* = upper-triangle feature *k*, ordered by
  `np.triu_indices(n_bins, k=1)` row-major:
  (0,1), (0,2), ..., (0, N-1), (1,2), ..., (N-2, N-1)
- Values are imputed (denoised) contact counts
- The matrix is dense (all n_features positions are imputed)

This exactly matches the FLAMINGO h5ad `layers['gt']` ordering for
direct element-wise comparison.

---

## Directory Structure

```
2_FLAMINGOData/
├── README.md                          ← This file
├── v3_inputData/                      ← Prepared scVI-3D input (714 MB total)
│   └── <stem>/                        ← One per dataset
│       ├── cell_1.txt ... cell_N.txt  ← 5-col per-cell contacts
│       ├── genome.txt                 ← chrom \t (n_bins-1)*resolution
│       └── cell_summary.txt           ← name \t batch \t cell_type
├── v3_outputData/                     ← Imputation results
│   ├── npz_lower_tri/                 ← Final imputed matrices (~1.6 GB each)
│   │   └── <stem>_scVI3D_imputed.npz
│   └── metrics/                       ← PCC/MAE CSV
│       └── scVI3D_FLAMINGO_v3_*.csv
└── v3_scripts/                        ← All portable scripts
    ├── prepare_scvi3d_input.py        ← Stage 1: raw data → scVI-3D input
    ├── run_scvi3d_flamingo.py         ← Stage 2: band matrix + scVI + collect
    ├── evaluate_scvi3d_flamingo.py    ← Stage 3: PCC/MAE evaluation
    ├── submit_prepare_all_cpu.sbatch  ← SLURM: prepare (CPU)
    ├── submit_scvi3d_flamingo.sbatch  ← SLURM: impute (GPU, array 0-6)
    ├── submit_scvi3d_7in2gpu.sbatch   ← SLURM: impute (2 GPUs, 7 datasets)
    └── submit_eval_scvi3d_cpu.sbatch  ← SLURM: evaluate (CPU)
```

## References

- Zheng, Y., Shen, S. & Keleş, S. (2022). Normalization and
  de-noising of single-cell Hi-C data with BandNorm and scVI-3D.
  *Genome Biology*, 23, 222.
  https://doi.org/10.1186/s13059-022-02809-z
- scVI-tools: https://docs.scvi-tools.org/ (version 0.14.6 used here)