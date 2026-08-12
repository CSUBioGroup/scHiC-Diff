# HiCImpute Simulation — FLAMINGO t-SVD ADMM PD-Space LRTC

```yaml
name: flamingo-tsvdImpute-HiCImputeData
description: FLAMINGO t-SVD/LRTC imputation on HiCImpute simulation data in PD (power-law distance) space, with IF<->PD conversion and distance/contact evaluation.
```

This directory contains the **original successful FLAMINGO PD-space workflow** that all other datasets in `9_FLAMINGO/` are derived from. It is the reference implementation for running FLAMINGO t-SVD ADMM tensor completion on HiC-C data via the power-law distance (PD) transformation.

## Goal

Run FLAMINGO tensor LRTC on HiCImpute simulation data. The input is IF (contact frequency) from bead-pair simulations; FLAMINGO operates in PD (power-law distance) space where the tensor has lower rank.

## Data Flow (PD-Space Pipeline)

```
HiCImpute sim.npz (cell × lower-triangle IF features)
    → prep: IF → PD (PD = IF^(-0.25)) → symmetric PD matrices → RawCount_Cell_XXX.txt
    → FLAMINGO t-SVD ADMM (run_flamingo_pyfftw_completion.py)
    → post: completed PD tensor → IF (IF = PD^(-4)) → contact matrices
```

### IF ↔ PD Conversion

```python
ALPHA = 0.25  # FLAMINGO power-law exponent

def contact_to_pd(contact):
    pd = np.zeros_like(contact)
    mask = contact > 0
    pd[mask] = contact[mask] ** (-ALPHA)   # PD = IF^(-0.25)
    return pd

def pd_to_contact(pd):
    contact = np.zeros_like(pd)
    mask = pd > 0
    contact[mask] = pd[mask] ** (-1.0 / ALPHA)  # IF = PD^(-4)
    return contact
```

**Critical**: PD=0 means "unobserved/no contact" and maps to contact=0. But small nonzero PD (e.g. 0.02) maps to contact = 0.02^(-4) ≈ 6e5 — a blow-up. The t-SVD completion must NOT produce tiny nonzero PD for missing entries (see Pitfalls below).

## Key Files

```text
1_HiCImputeData/
├── scripts/
│   ├── hicimpute_distance_lrtc.py       # prep (IF→PD) + eval (distance metrics)
│   └── convert_pd_to_contact.py         # post: completed PD → contact NPZ
├── run_hicimpute_distance_lrtc_cpu_array.sbatch        # SLURM array (selection=final)
├── run_hicimpute_distance_lrtc_best_cpu_array.sbatch   # SLURM array (selection=best)
├── input_distance/                      # prep output: PD RawCount txt + observed/truth tensors
│   ├── manifest.tsv
│   └── {dataset}/distance_matrices/RawCount_Cell_XXX.txt
├── output_distance/                     # selection=final results
│   └── {dataset}/completed_tensor.npy
└── output_distance_best/                # selection=best results
    └── {dataset}/completed_tensor.npy
```

The shared FLAMINGO runner (called via subprocess, NOT modified):

```text
/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/run_flamingo_pyfftw_completion.py
```

## FLAMINGO Hyper-Parameters

Two successful configurations exist:

### output_distance (selection=final)

```text
mu: 1e-4          # threshold = 1/mu = 10000 (shrinks almost all singular values to zero)
max_mu: 1e10
rho: 1.1
max_iter: 500
tol: 1e-4
selection: final   # last iteration
keep_observed: true
```

**Behavior**: 97% of missing entries get PD=0 (contact=0). This is effectively a no-op — it keeps observed data and fills missing with zeros. PCC is good only because the observed data already has good PCC.

### output_distance_best (selection=best) — RECOMMENDED

```text
mu: 1e-4
max_mu: 1e10
rho: 1.1
max_iter: 500
tol: 1e-4
selection: best    # lowest-residual iteration (typically iter 60-100)
keep_observed: true
```

**Behavior**: 100% of missing entries get moderate PD (0.74-1.16) → contact [0.55, 3.39]. This is **real imputation** — missing entries are filled with reasonable contact values.

## SLURM Submission

```bash
cd /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/1_HiCImputeData

# Best run (recommended)
sbatch run_hicimpute_distance_lrtc_best_cpu_array.sbatch

# Final run (no-op imputation, keeps observed only)
sbatch run_hicimpute_distance_lrtc_cpu_array.sbatch
```

Partition: `cpuQ`, account: `pi_limin_r`, QOS: `cpuq`, 8 CPUs, 16G, 12h, array 0-14 (15 datasets).

## Successful Result Snapshot (output_distance_best)

Missing entries get moderate PD → reasonable imputed contact [0.55, 3.39], observed entries [1, 10] kept as-is. See `output_distance_best/all_distance_summary_metrics.csv` for PCC/MAE/RMSE.

## Notes For Codex

When reproducing this workflow:

1. Work from `1_HiCImputeData`.
2. Use `scripts/hicimpute_distance_lrtc.py` for prep + eval, `scripts/convert_pd_to_contact.py` for post.
3. Use `run_hicimpute_distance_lrtc_best_cpu_array.sbatch` for the recommended run (selection=best).
4. The FLAMINGO runner is at `4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/run_flamingo_pyfftw_completion.py` — do NOT modify it.
5. `mu=1e-4` + `selection=best` is the key combo for real imputation. See `5_lee_SuperTAD_pileline/scripts/README_FLAMINGO.md` for a detailed explanation of why.
6. All compute runs on cpuQ nodes — never run FLAMINGO on the login node.