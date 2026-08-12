# FLAMINGO fixed-contact implementation notes

## Data contract

- Input: one symmetric, nonnegative 49 x 49 contact-count matrix per cell
- Source: `input_lee/per_cell_npz/{CellType}_cell_*.npz`
- Cell types: Astro, Endo, ODC, OPC
- Completion tensor: contact-count space throughout
- Final output: `imputed/FLAMINGO_fixed_contact/{CellType}_cell_NNNN.npz`
- Output format: SciPy CSR NPZ, symmetric, nonnegative, zero diagonal
- Positive observed contacts: restored before output is written

`lee_flamingo_pipeline.py` intentionally provides no alternate matrix-space
mode and no post-completion clipping policy.

## Validated completion settings

| Parameter | Value |
| --- | --- |
| Input subdirectory | `contact_matrices` |
| Maximum iterations | `500` |
| Tolerance | `1e-4` |
| Initial mu | `1e-4` |
| Maximum mu | `1e10` |
| rho | `1.1` |
| SVD backend | `serial` |
| Selected iterate | `best` |
| Patience | `0` |
| Minimum relative improvement | `0.0` |
| Restore observed contacts | always enabled |
| CPUs per production array task | `8` |

The completion backend must retain both FFT/SVD regression fixes:

- Reconstruct non-Hermitian complex FFT slices with the full complex SVD.
- Process every FFT frequency for odd and even cell counts when the threshold
  is zero.

## Production scripts

- `lee_flamingo_pipeline.py`: contact preparation and postprocessing
- `run_lee_flamingo_contact_pilot_cpu.sbatch`: Endo pilot
- `run_lee_flamingo_contact_full_cpu_array.sbatch`: four-cell production run
- `evaluate_flamingo_contact_maps.py`: structural, distance-trend, and PCC gate
- `evaluate_lee_flamingo_contact_full.sbatch`: four-cell gate
- `run_trials_fixed_contact.sbatch`: 100 trials x four cell types
- `audit_flamingo_trials.py`: audit all 400 trial matrices
- `run_supertad_fixed_contact.sbatch`: target and trial TAD calls
- `audit_supertad_outputs.py`: validate drawable SuperTAD output
- `build_supertad_linux.sbatch`: reproducible Linux binary build
- `plot_fixed_contact_panel_a.sbatch`: final panel A
- `run_all_pipeline_tests.sbatch`: contact-only unittest suite

## Required gates

1. Per-cell NPZ counts are Astro 449, Endo 202, ODC 1244, and OPC 203.
2. Every completed matrix is finite, symmetric, nonnegative, and preserves
   observed contacts.
3. Each cell type has `rho(value, genomic distance) < 0`.
4. Each cell type has positive 49 x 49 and 8 x 8 PCC.
5. All 400 trial matrices pass the same trend and PCC audit.
6. All 404 SuperTAD outputs are valid and drawable.
7. The contact-only regression suite passes on a Slurm CPU node.

Temporary tensors are written under `work/FLAMINGO_fixed_contact/`. They are
reproducible and may be removed after all acceptance gates pass.
