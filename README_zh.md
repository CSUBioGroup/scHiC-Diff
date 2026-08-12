[English](README.md) | **中文**

# scHiC-Diff

scHiC-Diff 是一个用于单细胞 Hi-C 插补与增强的条件扩散模型，通过迭代去噪重建染色质接触图。该模型使用 Diffusion Transformer 建模长程依赖，在两个互补的模拟数据集和三个真实 scHi-C 数据集上改善了相互作用恢复及下游分析，并增强了发育 scHi-C 数据中的染色质结构重建和染色质 loop 检测。

<img width="1688" height="600" alt="image" src="https://github.com/user-attachments/assets/e0fefd75-7b7f-4546-910d-551254e15cac" />

## 环境概览

- Python：3.9
- CUDA runtime：11.6
- PyTorch：1.12.1 + cu116
- PyTorch Lightning：1.9.0
- 主要生态：Scanpy、scvi-tools、rapids-singlecell 等

## scdiff 环境安装（micromamba + CUDA 11.6）

本仓库推荐在 Python 3.9 和 CUDA 11.6 环境中运行 scdiff。

### 1. 前提条件

- 推荐 Linux（HPC 或 Ubuntu）。
- 已安装 NVIDIA GPU 驱动。
- 已安装 micromamba，并可在 shell 中调用。
- CUDA runtime 由 Conda 环境中的 `cudatoolkit=11.6` 提供，无需手工安装系统 CUDA toolkit。

### 2. 创建 micromamba 环境

```bash
micromamba create -n scdiff python=3.9 pip=24.0
micromamba activate scdiff
```

### 3. 安装 CUDA runtime

```bash
micromamba install -c conda-forge cudatoolkit=11.6
```

`cudatoolkit` 在环境内提供 CUDA runtime 库，但计算节点仍需要可用的 NVIDIA 驱动。

### 4. 安装 CUDA 11.6 版本的 PyTorch

```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu116
```

### 5. 安装核心训练库

```bash
pip install pytorch-lightning==1.9.0 pyro-ppl==1.8.6
```

### 6. 安装完整 Python 依赖

```bash
pip install -r requirements.txt
```

### 7. 可选：bitsandbytes 与 triton

```bash
pip install bitsandbytes==0.39.1 triton==2.3.0
```

`bitsandbytes` 对 CUDA runtime 和驱动版本较敏感。如果导入失败，请移除它或安装与 GPU 和驱动兼容的版本。

### 8. 验证安装

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 从头训练并自动插补

下面的示例参考本项目的模拟数据 Slurm 流程。一次命令会从头训练 scHiC-Diff，并在训练结束后自动调用 `trainer.test()` 执行插补。请勿在 HPC 登录节点直接运行该计算任务。

### 输入数据

准备一个满足以下要求的 H5AD 文件：

- 每行为一个细胞；
- 每列为一个展平后的染色质 contact 特征；
- `adata.X` 保存非负的观测 contact 值；
- 唯一且顺序稳定的 `adata.var_names` 定义特征顺序；
- 推荐将 `adata.X` 保存为稀疏 CSR 矩阵。

示例将同一个 H5AD 文件配置为训练、验证和测试数据。scHiC-Diff 会根据配置在内部生成 mask 和数据划分。

### 单数据集 Slurm 示例

将以下内容保存为 `run_scdiff_example.sbatch`。请根据运行环境修改账户、GPU 分区、Python 环境、项目目录、输入 H5AD 和数据集名称。

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

提交作业：

```bash
sbatch run_scdiff_example.sbatch
```

`data.params.test_batch_size` 控制推理阶段的 batch 大小。如果 GPU 显存不足，可将其从 `128` 降为 `64` 或 `32`。训练 batch size 和测试 batch size 可独立配置。

### 输出文件

训练结束后，同一命令会自动执行测试插补，并在 `OUTPUT_DIR` 中写出以下稀疏 NPZ 矩阵：

```text
raw_x.npz
denoise_recon.npz
denoise_target.npz
denoise_recon_inv.npz
```

`denoise_recon.npz` 是归一化尺度上的预测结果，`denoise_recon_inv.npz` 是从 log 归一化尺度反变换后的预测结果。训练日志、实际配置和 checkpoint 保存在 `--logdir` 指定的目录下。

如需批量处理多个数据集，可在该示例基础上增加数据集名称数组，并使用 `SLURM_ARRAY_TASK_ID` 选择对应 H5AD 文件，方式与本项目的模拟数据流程一致。
