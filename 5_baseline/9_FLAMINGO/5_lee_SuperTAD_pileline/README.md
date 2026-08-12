# Lee PDGFRA FLAMINGO contact-space pipeline

This directory contains the validated FLAMINGO workflow for the Lee PDGFRA
scHi-C benchmark. The canonical method name is `FLAMINGO_fixed_contact`.

## Data flow

1. Read each 49 x 49 observed contact matrix from
   `input_lee/per_cell_npz/`.
2. Complete the tensor directly in contact-count space with the corrected
   FFT/SVD implementation.
3. Restore observed contacts and write symmetric, nonnegative, zero-diagonal
   NPZ matrices.
4. Generate 100 pseudo-bulk trials per cell type.
5. Require negative contact-versus-distance trend and positive PCC.
6. Run Linux SuperTAD and draw panel A.

The code has one supported representation: contact counts. There is no
alternative matrix-space conversion or post-completion threshold policy.

## Canonical inputs and outputs

- Input cells: `input_lee/per_cell_npz/`
- Ground truth: `target/`
- Per-cell imputation: `imputed/FLAMINGO_fixed_contact/`
- Trial matrices and PCC: `trials/FLAMINGO_fixed_contact/`
- Method SuperTAD calls: `supertad/FLAMINGO_fixed_contact/`
- Target SuperTAD calls: `supertad/target/`
- Final panel A: `nature-style-plots/results_fixed/`
- Validation evidence: `scripts/test_outputs/`
- Linux SuperTAD: `bin/SuperTAD_linux_x86_64`
- Pinned SuperTAD source: `6_SuperTAD/SuperTAD/`

Expected per-cell NPZ counts are Astro 449, Endo 202, ODC 1244, and OPC 203.
Expected trial and method-SuperTAD counts are 400 each.

## Reproduction

Submit all matrix work through Slurm from this directory:

```bash
# Optional Endo pilot
sbatch scripts/run_lee_flamingo_contact_pilot_cpu.sbatch

# Four-cell production run
sbatch scripts/run_lee_flamingo_contact_full_cpu_array.sbatch

# Gates and downstream outputs
sbatch scripts/evaluate_lee_flamingo_contact_full.sbatch
sbatch scripts/run_trials_fixed_contact.sbatch
sbatch scripts/audit_trials_fixed_contact.sbatch
sbatch scripts/run_supertad_fixed_contact.sbatch
sbatch scripts/audit_supertad_fixed_contact.sbatch
sbatch scripts/plot_fixed_contact_panel_a.sbatch

# Contact-only regression suite
sbatch scripts/run_all_pipeline_tests.sbatch
```

Use Slurm `afterok` dependencies for a production rerun. The completion jobs
call the corrected runner at:

`../../../4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/run_flamingo_pyfftw_completion.py`

See `scripts/README_FLAMINGO.md` for the exact hyperparameters and acceptance
gates. See `CLEANUP_MANIFEST.md` for the retained layout and job evidence.
