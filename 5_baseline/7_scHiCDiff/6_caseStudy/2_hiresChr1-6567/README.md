# Hires chr1 scHiC-Diff imputation

This directory compares the case-study `eval_denoising_v2` configuration with the successful Ramani `recon_masked` configuration on the fixed 7,466-cell, 5,050-feature chr1:65-67 Mb input.

## Fixed resources

- Input: `/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/6_caseData/2_callLoop_apa/inputData/6_hires_chr1/hires_20kb_chr1.h5ad`
- Code: `/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/3_DiffusionModel/scHiC-Diff-master`
- Slurm: `pi_limin_r`, GPU QoS `gpuq`, one GPU per array task

## Effective settings

| Tag | Training batch | Full epochs | Learning rate | Loss |
| --- | ---: | ---: | ---: | --- |
| `eval_v2` (original, invalid) | 8000 | 1500 | 1e-8 | `recon_full` |
| `eval_v2_compat` | 8000 | 1500 | 8e-5 | `recon_full` + v2 masking |
| `ramani` | 1024 | 3000 | 2e-4 | `recon_masked` |

The code's test loader uses batch size 9999, so all 7,466 cells are serialized in one test batch for both configurations.

## Commands

```bash
sbatch scripts/test_eval_v2_compat.sbatch
bash scripts/submit_phase.sh smoke gpu4RQ
bash scripts/submit_phase.sh full gpu4RQ
bash scripts/submit_eval_v2_compat.sh smoke gpu4RQ
bash scripts/submit_eval_v2_compat.sh full gpu4RQ
```

The optional second argument selects one of `gpu2Q`, `gpu4Q`, `gpu8Q`, or `gpu4RQ`. The submitter checks live Slurm TRES allocation and refuses to submit when the selected partition has no free GPU. The full command also refuses to submit until both smoke `done.flag` files exist.

Every result directory is validated for shape, finite values, and nonnegative values before its completion marker is written.

## Quality status

- Do not use `results/full/eval_v2/denoise_recon_inv.npz`. That run used the
  wrong effective learning rate and masking semantics; its independent quality
  diagnosis is under `results/full/eval_v2/diagnostics/`.
- Corrected eval_v2 artifacts are written under
  `results/corrected/eval_v2/{smoke,full}/`. A usable result requires
  `validation.json`, `quality.json`, `quality_passed.flag`, and final
  `done.flag`.
- `results/full/ramani/denoise_recon_inv.npz` passed structural validation but
  failed the independent depth gate: median prediction depth 667 versus target
  depth 85. The unmodified audit is in `results/full/ramani/quality.json`.

## Final outputs

- Corrected eval_v2: `results/corrected/eval_v2/full/denoise_recon_inv.npz`
- Ramani (depth calibration required): `results/full/ramani/denoise_recon_inv.npz`

Slurm job IDs and scheduler logs are under `logs/{smoke,full}/`. scHiC-Diff CSV logs and checkpoints are under `model_logs/{smoke,full}/`.
