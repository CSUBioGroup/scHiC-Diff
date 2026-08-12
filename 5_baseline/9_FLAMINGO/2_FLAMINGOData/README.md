# flamingo-tsvdImpute-FLAMINGOData

```yaml
name: flamingo-tsvdImpute-FLAMINGOData
description: Reproduce the successful FLAMINGO t-SVD/LRTC imputation workflow for V3 h5ad contact-scale FLAMINGOData, including input preparation, Slurm submission, metric merging, and log1p/contact heldout evaluation.
```

This directory keeps the successful V3 contact-scale FLAMINGO workflow. It is written as an agent-readable runbook, not as an installed Codex skill.

## Goal

Run FLAMINGO tensor LRTC on V3 simulation h5ad contact data, using sparse observed contact matrices as input and `gt` contact as ground truth for evaluation.

The successful experiment is:

- Input scale: raw contact counts from h5ad layer `counts`
- Ground truth scale: raw contact counts from h5ad layer `gt`
- Model input to FLAMINGO: raw contact matrix text files
- Primary evaluation: `log1p(pred_contact)` vs `log1p(gt_contact)` on heldout positions
- Secondary evaluation: raw contact PCC/MAE and all-position metrics

## Important Files

Keep these files:

```text
2_FLAMINGOData/
├── scripts/v3_h5ad_contact_lrtc.py
├── run_v3_h5ad_contact_lrtc_cpu20_array.sbatch
├── combine_v3_h5ad_contact_lrtc_metrics_cpu.sbatch
└── tests/test_v3_h5ad_contact_lrtc.py
```

The workflow also calls the shared FLAMINGO runner outside this directory:

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/run_flamingo_pyfftw_completion.py
```

## Source Data

The source h5ad files are under:

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/3_fixed_flamingoGen_datasets/5_paramsweep_datasets
```

The workflow discovers files matching:

```text
v3_hybrid_*_500cells_level0*_scdiff2.h5ad
```

Files containing `heldout_masked` in the filename are intentionally excluded.

Each selected h5ad has:

- `layers["counts"]`: sparse observed contact counts. Missing/unobserved entries are zero.
- `layers["gt"]`: ground truth contact counts.
- `n_cells = 1500`
- `n_beads = 500`
- `n_features = 124750`, the lower-triangle size for 500 beads.

The successful manifest is:

```text
v3ContactInput/manifest.tsv
```

It contains seven datasets:

```text
w0p5_r005_contact
w0p6_r005_contact
w0p7_r005_contact
w0p7_r0p01_contact
w0p7_r0p05_contact
w0p8_r005_contact
w0p9_r005_contact
```

## Input Preparation

Preparation is implemented in:

```text
scripts/v3_h5ad_contact_lrtc.py
```

Main command:

```bash
/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python scripts/v3_h5ad_contact_lrtc.py prep --manifest v3ContactInput/manifest.tsv --input-parent v3ContactInput --task-id <0-6>
```

For each dataset it:

1. Loads h5ad layer `counts` as observed contact.
2. Loads h5ad layer `gt` as truth contact.
3. Cleans non-finite and negative values to zero.
4. Converts each cell's lower-triangle vector into a symmetric 500 x 500 matrix.
5. Writes FLAMINGO input text matrices to:

```text
v3ContactInput/<dataset>/contact_matrices/RawCount_Cell_####.txt
```

It also writes:

```text
v3ContactInput/<dataset>/observed_contact_features.npz
v3ContactInput/<dataset>/truth_contact_features.npz
v3ContactInput/<dataset>/observed_contact_tensor.npy
v3ContactInput/<dataset>/truth_contact_tensor.npy
v3ContactInput/<dataset>/input_file_index.csv
v3ContactInput/<dataset>/metadata.json
```

The marker file `v3ContactInput/<dataset>/.complete` makes repeated prep idempotent unless `--force` is used.

## FLAMINGO Imputation

The Slurm array script is:

```text
run_v3_h5ad_contact_lrtc_cpu20_array.sbatch
```

It uses:

```text
partition: cpuQ
cpus-per-task: 20
memory: 50G
array: 0-6
max_iter: 500
tol: 1e-4
mu: 1e-4
max_mu: 1e10
rho: 1.1
selection: final
keep_observed: true
```

The FLAMINGO command inside the sbatch is:

```bash
"${PYTHON_BIN}" "${LRTC_SCRIPT}" \
  --input-root "${INPUT_PARENT}" \
  --input-subdir contact_matrices \
  --output-root "${OUTPUT_PARENT}" \
  --datasets "${DATASET}" \
  --max-iter "${FLAMINGO_MAX_ITER:-500}" \
  --tol "${FLAMINGO_TOL:-1e-4}" \
  --mu "${FLAMINGO_MU:-1e-4}" \
  --max-mu "${FLAMINGO_MAX_MU:-1e10}" \
  --rho "${FLAMINGO_RHO:-1.1}" \
  --n-threads "${SLURM_CPUS_PER_TASK:-20}" \
  --selection final \
  --keep-observed
```

Output is written to:

```text
v3ContactOutput/<dataset>/
```

Key output files per dataset:

```text
completed_tensor.npy
high_resolution.npy
high_res_contact_maps_FLAMINGO/
process_time.tsv
evaluation_runtime.json
v3_h5ad_contact_lrtc_cell_level_metrics.csv
v3_h5ad_contact_lrtc_summary_metrics.csv
v3_h5ad_contact_lrtc_summary_by_cell_type.csv
```

## Evaluation Definition

Evaluation is implemented by `scripts/v3_h5ad_contact_lrtc.py eval`.

For each cell:

```text
observed_mask = counts > 0
truth_mask    = gt > 0
heldout_mask  = (~observed_mask) & truth_mask
all_mask      = truth_mask
```

Metrics are computed for both raw contact and log1p contact:

```text
contact_all
contact_observed
contact_heldout
log1p_contact_all
log1p_contact_observed
log1p_contact_heldout
```

Primary quality columns:

```text
pcc_log1p_contact_heldout_mean
spearman_log1p_contact_heldout_mean
mae_log1p_contact_heldout_mean
pcc_contact_heldout_mean
mae_contact_heldout_mean
```

## One-Command Reproduction

From this directory:

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData
```

Submit the full array and then submit merge with dependency:

```bash
jid=$(sbatch --parsable run_v3_h5ad_contact_lrtc_cpu20_array.sbatch); sbatch --dependency=afterok:${jid} combine_v3_h5ad_contact_lrtc_metrics_cpu.sbatch
```

Monitor:

```bash
squeue -u hpc254701055 -o "%.18i %.9P %.35j %.8u %.2t %.10M %.6D %R"
```

After all array tasks and the merge job finish, check:

```text
v3ContactOutput/all_v3_h5ad_contact_lrtc_summary_metrics.csv
v3ContactOutput/all_v3_h5ad_contact_lrtc_summary_by_cell_type.csv
v3ContactOutput/all_v3_h5ad_contact_lrtc_process_times.csv
```

The existing concise metrics table is:

```text
v3ContactOutput/v3_h5ad_contact_lrtc_key_metrics.csv
```

If rerunning from scratch in the same directory, move or remove old `v3ContactInput` and `v3ContactOutput` first. Otherwise, prep may reuse `.complete` input directories and outputs may be overwritten.

## Generate Key Metrics Table

Use this command to regenerate `v3_h5ad_contact_lrtc_key_metrics.csv` from per-dataset summary files:

```bash
python - <<'PY'
from pathlib import Path
import csv

root = Path('/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData/v3ContactOutput')
order = [
    'w0p5_r005_contact',
    'w0p6_r005_contact',
    'w0p7_r005_contact',
    'w0p8_r005_contact',
    'w0p9_r005_contact',
    'w0p7_r0p01_contact',
    'w0p7_r0p05_contact',
]
cols = [
    ('dataset_id', 'dataset_id'),
    ('pcc_log1p_contact_heldout_mean', 'log1p_heldout_PCC'),
    ('spearman_log1p_contact_heldout_mean', 'log1p_heldout_SCC'),
    ('mae_log1p_contact_heldout_mean', 'log1p_heldout_MAE'),
    ('pcc_contact_heldout_mean', 'raw_contact_heldout_PCC'),
    ('mae_contact_heldout_mean', 'raw_contact_heldout_MAE'),
    ('pcc_log1p_contact_all_mean', 'log1p_all_PCC'),
    ('mae_log1p_contact_all_mean', 'log1p_all_MAE'),
    ('pcc_contact_all_mean', 'raw_contact_all_PCC'),
    ('mae_contact_all_mean', 'raw_contact_all_MAE'),
]
rows = []
for ds in order:
    path = root / ds / 'v3_h5ad_contact_lrtc_summary_metrics.csv'
    with path.open(newline='') as handle:
        row = next(csv.DictReader(handle))
    rows.append({
        dst: row[src] if dst == 'dataset_id' else f'{float(row[src]):.6f}'
        for src, dst in cols
    })

out = root / 'v3_h5ad_contact_lrtc_key_metrics.csv'
with out.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=[dst for _, dst in cols])
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY
```

## Successful Result Snapshot

Current key metrics:

| dataset | log1p heldout PCC | log1p heldout SCC | log1p heldout MAE | raw contact heldout PCC | raw contact heldout MAE | log1p all PCC | log1p all MAE | raw contact all PCC | raw contact all MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `w0p5_r005_contact` | 0.663334 | 0.650504 | 0.306534 | 0.687177 | 1.327835 | 0.665539 | 0.304996 | 0.689131 | 1.321173 |
| `w0p6_r005_contact` | 0.718260 | 0.710079 | 0.292807 | 0.723260 | 1.270803 | 0.719856 | 0.291339 | 0.724599 | 1.264430 |
| `w0p7_r005_contact` | 0.780670 | 0.778494 | 0.279521 | 0.778210 | 1.199463 | 0.781677 | 0.278120 | 0.778881 | 1.193450 |
| `w0p8_r005_contact` | 0.822266 | 0.824658 | 0.272024 | 0.822273 | 1.152771 | 0.822919 | 0.270660 | 0.822508 | 1.146993 |
| `w0p9_r005_contact` | 0.841958 | 0.845941 | 0.266947 | 0.845861 | 1.114153 | 0.842485 | 0.265609 | 0.845974 | 1.108569 |
| `w0p7_r0p01_contact` | 0.835537 | 0.833098 | 0.191878 | 0.827715 | 0.863116 | 0.837493 | 0.189955 | 0.830722 | 0.854464 |
| `w0p7_r0p05_contact` | 0.846369 | 0.839665 | 0.229366 | 0.830649 | 1.046642 | 0.851789 | 0.217875 | 0.841904 | 0.994207 |

## Notes For Codex

When asked to reproduce this workflow:

1. Work from `2_FLAMINGOData`.
2. Do not use older `distanceOutput`, `contactLinearOutput`, PD, or raw-distance scripts.
3. Use `scripts/v3_h5ad_contact_lrtc.py` for manifest, prep, eval, and merge.
4. Use `run_v3_h5ad_contact_lrtc_cpu20_array.sbatch` for the seven-dataset array.
5. Use `combine_v3_h5ad_contact_lrtc_metrics_cpu.sbatch` after all array tasks finish.
6. Treat `log1p_contact_heldout` as the primary imputation-quality metric.
7. Remember this is raw-contact input to FLAMINGO; log1p is used for evaluation, not for the LRTC input matrix.

## FLAMINGO Method Notes (Cross-Dataset Reference)

This section documents the FLAMINGO t-SVD ADMM method itself, so future Codex processes can correctly apply it to new datasets. See also `1_HiCImputeData/README.md` and `5_lee_SuperTAD_pileline/scripts/README_FLAMINGO.md`.

### Two Input Space Modes

FLAMINGO can run in two input spaces. **This dataset (`2_FLAMINGOData`) uses contact space directly.** The choice depends on data density:

| Mode | Input to FLAMINGO | Conversion | Used by | When to use |
|------|-------------------|------------|---------|-------------|
| **Contact space** | Raw contact counts (IF) | None — run t-SVD directly on IF | `2_FLAMINGOData`, `3_ramaniData` | Dense-ish data (≥15% observed), simulation with ground truth |
| **PD space** | PD = IF^(-0.25) | IF→PD before, PD→IF after | `1_HiCImputeData`, `5_lee_SuperTAD_pileline` | Sparse single-cell data, or when following original FLAMINGO paper |

### Data Flow

**Contact space** (this dataset):
```
h5ad counts layer (cell × lower-triangle IF)
    → prep: reconstruct symmetric contact matrix → RawCount_Cell_XXX.txt (IF values)
    → FLAMINGO t-SVD ADMM (run_flamingo_pyfftw_completion.py)
    → completed contact tensor (directly usable, no PD↔IF conversion)
```

**PD space** (1_HiCImputeData, 5_lee_SuperTAD_pileline):
```
contact NPZ (cell × N IF)
    → prep: IF → PD (PD = IF^(-0.25)) → RawCount_Cell_XXX.txt (PD values)
    → FLAMINGO t-SVD ADMM (run_flamingo_pyfftw_completion.py)
    → post: completed PD → IF (IF = PD^(-4)), clip outliers, restore observed
    → contact NPZ (count space)
```

### Key Hyper-Parameters

| Parameter | Contact space (this dataset) | PD space (1_HiCImputeData best) | PD space (5_lee) |
|-----------|------------------------------|----------------------------------|-------------------|
| `mu` | 1e-4 | 1e-4 | 1e-4 |
| `max_mu` | 1e10 | 1e10 | 1e10 |
| `rho` | 1.1 | 1.1 | 1.1 |
| `max_iter` | 500 | 500 | 500 |
| `selection` | **final** | **best** | **best** |
| `keep_observed` | true | true | true |

**Why `selection` differs**: In contact space, the ADMM converges stably (final iteration is good). In PD space, the ADMM **diverges** in later iterations (residual → 1e54 as mu grows), so `selection=best` is needed to pick the lowest-residual iteration (typically iter 60-100) which has moderate PD values for missing entries.

### Critical Pitfalls (PD Space Only)

1. **`selection=final` with PD space → no-op imputation**: The final iteration has PD=0 for 97% of missing entries → contact=0. This just keeps observed data and fills missing with zeros. PCC looks OK only because observed data already has good PCC. Use `selection=best` for real imputation.

2. **`mu=1.0` (or any large mu) with PD space → contact blow-up**: threshold=1/mu=1.0 keeps too many singular values, producing tiny nonzero PD (0.02) for missing entries → contact = 0.02^(-4) ≈ 6e5. **Always use `mu=1e-4`** (threshold=10000) which shrinks almost all singular values to zero; with `selection=best`, the early iterations produce moderate PD (0.7-1.2) → contact [0.5, 4].

3. **PD→contact conversion blow-up**: `contact = PD^(-4)` explodes when PD → 0. In post-processing, entries with PD below a threshold should be set to **contact=0** (no contact), NOT floored to a minimum PD (which forces ALL missing entries to have large contact). See `5_lee_SuperTAD_pileline/scripts/lee_flamingo_pipeline.py` `post_cell_type()` for the correct zeroing logic.

4. **Never run FLAMINGO on the login node**: The t-SVD ADMM is CPU-intensive. Always submit to `cpuQ` via SLURM sbatch scripts.

### Shared FLAMINGO Runner

All datasets call the same unchanged runner:

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/run_flamingo_pyfftw_completion.py
```

It expects `RawCount_Cell_XXX.txt` files (1-indexed) in `{input_root}/{dataset}/{input_subdir}/` and writes `completed_tensor.npy` to `{output_root}/{dataset}/`.
