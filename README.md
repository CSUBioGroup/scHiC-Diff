# scHiC-Diff
scHiC-Diff is a conditional diffusion model for scHi-C imputation that reconstructs chromatin contact maps through iterative denoising. Using a Diffusion Transformer to model long-range dependencies, scHiC-Diff consistently improves interaction recovery and downstream analyses across nine simulated datasets and three real scHi-C datasets. It also enhances chromatin organization reconstruction and chromatin loop detection on developmental scHi-C data.
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


