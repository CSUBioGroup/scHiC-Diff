**English** | [中文](README_zh.md)

# scHiC-Diff
scHiC-Diff is a conditional diffusion model for scHi-C imputation and enhancement that reconstructs chromatin contact maps through iterative denoising. Using a Diffusion Transformer to model long-range dependencies, scHiC-Diff consistently improves interaction recovery and downstream analyses across nine simulated datasets and three real scHi-C datasets. It also enhances chromatin organization reconstruction and chromatin loop detection on developmental scHi-C data.
<img width="1688" height="600" alt="image" src="https://github.com/user-attachments/assets/e0fefd75-7b7f-4546-910d-551254e15cac" />


## Environment summary
	•	Python: 3.9
	•	CUDA runtime: 11.6
	•	PyTorch: 1.12.1 + cu116
	•	PyTorch Lightning: 1.9.0
	•	Bib ecosystem: Scanpy / scvi-tools / rapids-singlecell / etc.


scdiff Environment Setup (micromamba + CUDA 11.6)

This repository provides the recommended Python environment configuration for running scdiff under Python 3.9 with CUDA 11.6 support.

#### 1. Prerequisites
	•	Linux recommended (HPC/Ubuntu).
	•	NVIDIA GPU + driver installed.
	•	micromamba installed and available in your shell.
	•	CUDA runtime is provided via cudatoolkit=11.6 (Conda), so no need to install system CUDA toolkit manually.

#### 2. Create a micromamba environment

```
micromamba create -n scdiff python=3.9 pip=24.0
micromamba activate scdiff
```

#### 3. Install CUDA runtime (Conda)

Install CUDA runtime libraries (CUDA 11.6):
```
micromamba install -c conda-forge cudatoolkit=11.6
```
Note: cudatoolkit provides CUDA runtime libraries inside the environment, but your machine still needs a working NVIDIA driver.


#### 4. Install PyTorch (CUDA 11.6 build)
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu116
```


#### 5. Install core training libraries
```
pip install pytorch-lightning==1.9.0 pyro-ppl==1.8.6
```



#### 6. Install full Python dependencies by requirements.txt

```
pip install -r requirements.txt
```


#### 7. Optional: bitsandbytes + triton

If your workflow requires bitsandbytes and triton:
```
pip install bitsandbytes==0.39.1 triton==2.3.0
```
Warning: bitsandbytes can be sensitive to CUDA runtime and driver versions. If it fails to import, try removing it or installing a compatible build for your GPU/driver setup.


#### 8. Verify installation

Run:
```
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Train from Scratch and Impute Automatically

The example below follows the simulated-data Slurm workflow used in this project. A single invocation trains scHiC-Diff from scratch and automatically runs `trainer.test()` for imputation after training finishes. Do not run this workload on an HPC login node.

### Input

Prepare one H5AD file in which:

- rows are cells;
- columns are flattened chromatin-contact features;
- `adata.X` contains nonnegative observed contact values;
- unique and stable `adata.var_names` define the feature order;
- a sparse CSR matrix is recommended for `adata.X`.

The same H5AD file is assigned to the training, validation, and test dataset entries. scHiC-Diff creates the configured masks and splits internally.

### Single-Dataset Slurm Example

Save the following as `run_scdiff_example.sbatch`. Replace the account, GPU partition, environment, project directory, input H5AD, and dataset label for your system.

```bash
#!/bin/bash
#SBATCH --job-name=scdiff_example
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --qos=<GPU_QOS>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=scdiff_example_%j.log
#SBATCH --error=scdiff_example_%j.log

set -euo pipefail

PROJECT_DIR="/path/to/scHiC-Diff"
PYTHON_EXEC="/path/to/scdiff/bin/python"
INPUT_H5AD="/path/to/example_dataset.h5ad"
DATASET_NAME="example_dataset"
OUTPUT_DIR="${PROJECT_DIR}/results/${DATASET_NAME}"
SEED=10
MAX_EPOCHS=1000
TEST_BATCH_SIZE=128

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"

"${PYTHON_EXEC}" main.py \
  -b configs/recon_masked.yaml \
  --train true \
  --seed "${SEED}" \
  --save_path "${OUTPUT_DIR}" \
  --logdir "${PROJECT_DIR}/logs" \
  --postfix "train_from_scratch_${DATASET_NAME}.seed${SEED}" \
  --wandb_offline true \
  "lightning.trainer.devices=[0]" \
  "lightning.trainer.max_epochs=${MAX_EPOCHS}" \
  "data.params.batch_size=128" \
  "data.params.test_batch_size=${TEST_BATCH_SIZE}" \
  "data.params.num_workers=${SLURM_CPUS_PER_TASK}" \
  "data.params.train.params.dataset=${DATASET_NAME}" \
  "data.params.train.params.fname=${INPUT_H5AD}" \
  "data.params.validation.params.dataset=${DATASET_NAME}" \
  "data.params.validation.params.fname=${INPUT_H5AD}" \
  "data.params.test.params.dataset=${DATASET_NAME}" \
  "data.params.test.params.fname=${INPUT_H5AD}"
```

Submit the job with:

```bash
sbatch run_scdiff_example.sbatch
```

`data.params.test_batch_size` controls inference batching. Reduce it, for example from `128` to `64` or `32`, if GPU memory is insufficient. Training batch size and test batch size are configured independently.

### Outputs

After training, the same command automatically runs test-time imputation and writes the following sparse NPZ matrices to `OUTPUT_DIR`:

```text
raw_x.npz
denoise_recon.npz
denoise_target.npz
denoise_recon_inv.npz
```

`denoise_recon.npz` contains predictions on the normalized scale, while `denoise_recon_inv.npz` contains predictions transformed back from the log-normalized scale. Training logs, effective configurations, and checkpoints are written under the directory specified by `--logdir`.

For multi-dataset Slurm arrays, extend the example with a dataset-name array and select one H5AD file using `SLURM_ARRAY_TASK_ID`, as done by the project simulation workflows.

