# scHiC-Diff 在 HiCImputeData 上的插补结果

## 一、什么是 scHiC-Diff 插补?

scHiC-Diff 是一个基于条件扩散模型的单细胞 Hi-C 去噪/插补方法。其核心思路:

1. **输入**: 稀疏的观测 contact 矩阵 (有大量缺失值)
2. **训练阶段**: 对非零元素随机 mask 80% (`mask_none_zero=0.8`),对零元素额外 mask 10% (`zero_to_none_zero=0.1`),用条件扩散模型学习从 mask 后的输入重建完整矩阵
3. **推理阶段**: 对观测数据做扩散去噪 (1000 步),输出插补后的完整 contact 矩阵
4. **输出**: `denoise_recon_inv.npz` -- 反归一化后的插补结果,与 GT 直接可比

### HiCImputeData 上的插补任务

- **数据**: 12 个 K562 单细胞 Hi-C 模拟数据集 (chr19, 61 bins, 100 cells/dataset)
- **输入**: h5ad 文件,包含观测的稀疏 contact 矩阵 (100 cells × 1830 上三角 features)
- **输出**: (100, 1830) sparse CSR,每个 cell 的插补 contact 向量
- **评估**: 与 GT (`0_gtData/1_Gt_HiCImputeData/{dataset}_true.npz`) 计算 cell-wise PCC 和 MAE

```
K562_T1_1k  K562_T1_2k  K562_T1_4k  K562_T1_7k
K562_T2_1k  K562_T2_2k  K562_T2_4k  K562_T2_7k
K562_T3_1k  K562_T3_2k  K562_T3_4k  K562_T3_7k
```

T1/T2/T3 = 3 种细胞类型; 1k/2k/4k/7k = 4 种测序深度 (每细胞 1000/2000/4000/7000 reads)。

---

## 二、模型版本说明

scHiC-Diff 有多个代码版本,本目录保存了两个批次的结果,分别来自不同版本:

### Batch 10: `v5_scdiff_1mbsucess` 版本

| 属性 | 值 |
|------|-----|
| 项目目录 | `/public/home/hpc254701055/2_projects/10_schicdiff/v5_scdiff_1mbsucess/3_DiffusionModel/scHiC-Diff-master` |
| Python 环境 | `/public/home/hpc254701055/micromamba/envs/scdiff2/bin/python` |
| Config 文件 | `configs/recon_masked.yaml` |
| 代码特点 | v5 版本,包含 EarlyStopping 回调 (patience=25) |

### Batch 6: `1_scHiC` 原始版本

| 属性 | 值 |
|------|-----|
| 项目目录 | `/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/3_DiffusionModel/scHiC-Diff-master` |
| Python 环境 | `/public/home/hpc254701055/micromamba/envs/scdiff2/bin/python` |
| Config 文件 | `configs/recon_masked.yaml` |
| 代码特点 | 原始 scHiC-Diff 代码,**无 EarlyStopping 回调** |

### 两个版本的关键差异

| 特性 | Batch 6 (`1_scHiC` 原始) | Batch 10 (`v5_scdiff_1mbsucess`) |
|------|--------------------------|----------------------------------|
| EarlyStopping | **无** (main.py 中未定义) | **有**: patience=25, min_delta=1e-4 |
| ES monitor | N/A | `val/loss_MSE_ema` |
| ModelCheckpoint monitor | `val/loss_ema` | `val/loss_ema` |
| num_sanity_val_steps | 未设置 (默认=2) | 0 |
| test DataLoader batch_size | 9999 (硬编码) | 9999 (硬编码) |
| save_result 时机 | 每个 test batch 分别保存 | 每个 test batch 分别保存 |

> **注意**: EarlyStopping 的 monitor 是 `val/loss_MSE_ema` (MSE 损失),而 ModelCheckpoint 的 monitor 是 `val/loss_ema` (MSE+VLB 合并损失)。两者在 config 中通过 `monitor` 字段指定的是 ModelCheckpoint 的监控指标,EarlyStopping 的监控指标在 `main.py` 中硬编码。

### 其他版本 (未使用,仅供参考)

| 版本 | 路径 | 特点 |
|------|------|------|
| `v5_scdiff_fast` | `v5_scdiff_fast/3_DiffusionModel/scHiC-Diff-master` | monitor 改为 `val/loss_MSE_ema`,patience=25 |
| `v5_scdiff_fast_batched_test` | `v5_scdiff_fast_batched_test/3_DiffusionModel/scHiC-Diff-master` | patience=3000 (不早停), test_batch_size 可配置, save_result 在所有 batch 拼接后执行 |

---

## 三、训练配置详情

### 通用配置 (Batch 6 和 Batch 10 一致)

| 参数 | 值 | 说明 |
|------|-----|------|
| **数据** | | |
| batch_size | 128 | 训练 batch size |
| num_workers | 20 | DataLoader 工作进程数 |
| splits | train=0.8, valid=0.2 | 训练/验证集划分 |
| post_cond_flag | true | 使用条件输入 |
| normalize | true | 数据归一化 |
| return_raw | true | 返回原始矩阵 |
| **模型** | | |
| base_learning_rate | 2e-4 | AdamW 学习率 |
| timesteps | 1000 | 扩散步数 |
| denoise_t_sample | 1000 | 去噪采样步数 |
| parameterization | x0 | 预测 x0 (非 epsilon) |
| loss_type | l2 | L2 损失 |
| loss_strategy | recon_masked | 仅重建 masked 元素 |
| mask_strategy | none_zero | mask 非零元素策略 |
| mask_none_zero | 0.8 | mask 80% 非零元素 |
| zero_to_none_zero | 0.1 | mask 10% 零元素 (相对非零 mask 量) |
| clip_denoised | true | 裁剪去噪输出 |
| **架构** | | |
| depth | 6 | Transformer 层数 |
| embed_dim | 512 | 嵌入维度 |
| dim_head | 64 | 每头维度 |
| num_heads | 8 | 注意力头数 |
| decoder_embed_dim | 512 | 解码器嵌入维度 |
| decoder_dim_head | 64 | 解码器每头维度 |
| decoder_num_heads | 8 | 解码器注意力头数 |
| activation | gelu | 激活函数 |
| norm_layer | layernorm | 归一化层 |
| cond_type | crossattn | 条件输入类型 (交叉注意力) |
| cond_tokens | 1 | 条件 token 数 |
| encoder_type | mlp | 编码器类型 |
| decoder_embed_type | embedder | 解码器嵌入类型 |
| decoder_mask | inv_enc | 解码器 mask 策略 |
| **训练** | | |
| max_epochs | 1000 | 最大训练轮数 |
| seed | 10 | 随机种子 |
| accelerator | gpu | GPU 训练 |
| devices | [0] | 使用 GPU 0 |
| enable_progress_bar | false | 禁用进度条 |
| log_every_n_steps | 20 | 日志记录间隔 |

### Batch 10 独有: EarlyStopping

```python
# v5_scdiff_1mbsucess main.py 第564-565行
'early_stopping_callback': {
    'target': 'pytorch_lightning.callbacks.EarlyStopping',
    'params': {
        'monitor': 'val/loss_MSE_ema',
        'min_delta': 1e-4,
        'patience': 25,
        'verbose': True,
        'mode': 'min',
        'strict': True
    }
}
```

- monitor: `val/loss_MSE_ema` (验证集 MSE 损失的 EMA)
- min_delta: 1e-4 (改善量至少 0.0001 才算改善)
- patience: 25 (25 个 epoch 无改善则停止)
- 实际触发: epoch 86-164 之间

---

## 四、目录结构

```
7_scHiCDiff/1_HiCImputeData/
├── README.md                                          # 本文件
├── input/                                             # 12 个 h5ad 输入文件
├── scripts/                                           # SLURM 提交脚本 (当前为空)
├── output/
│   ├── training_results_v5_1schic_like_bs128_lr2e4/  # Batch 10 插补结果 (12 dataset)
│   ├── training_results_1scHiC_from_scratch/         # Batch 6 插补结果 (12 dataset)
│   ├── configs/
│   │   ├── batch10_v5_1schic_like_bs128_lr2e4/       # Batch 10 配置
│   │   │   ├── recon_masked.yaml                     # 原始 config 文件
│   │   │   ├── 2026-06-14T15-04-59-project.yaml      # 训练时配置快照 (项目)
│   │   │   └── 2026-06-14T15-04-59-lightning.yaml    # 训练时配置快照 (Lightning)
│   │   └── batch6_1scHiC_from_scratch/               # Batch 6 配置
│   │       ├── recon_masked.yaml                     # 原始 config 文件
│   │       ├── 2026-06-14T11-22-58-project.yaml      # 训练时配置快照 (项目)
│   │       └── 2026-06-14T11-22-58-lightning.yaml    # 训练时配置快照 (Lightning)
│   └── npz_lower_tri/                                # 标准化 NPZ (用于指标计算)
│       └── {dataset}_scHiCDiff_imputed.npz           # Batch 10 的 denoise_recon_inv.npz 拷贝
└── logs/
    ├── batch10_v5_1schic_like_bs128_lr2e4/           # Batch 10 训练日志 (12 log dir)
    └── batch6_1scHiC_from_scratch/                   # Batch 6 训练日志 (12 log dir)
```

---

## 五、Batch 10: v5_1schic_like_bs128_lr2e4

### 训练耗时

| 数据集 | 起始时间 | 完成时间 | Epoch | 耗时 |
|--------|---------|---------|-------|------|
| K562_T1_1k | 15:04:59 | 15:07:32 | 164 | 2.5 min |
| K562_T1_2k | 15:04:59 | 15:06:38 | 96 | 1.6 min |
| K562_T1_4k | 15:04:59 | 15:06:31 | 86 | 1.5 min |
| K562_T1_7k | 15:04:59 | 15:06:51 | 110 | 1.9 min |
| K562_T2_1k | 15:04:59 | 15:07:06 | 130 | 2.1 min |
| K562_T2_2k | 15:04:59 | 15:07:20 | 147 | 2.4 min |
| K562_T2_4k | 15:08:15 | 15:11:14 | 98 | 3.0 min |
| K562_T2_7k | 15:06:38 | 15:09:04 | 154 | 2.4 min |
| K562_T3_1k | 15:06:46 | 15:08:31 | 103 | 1.8 min |
| K562_T3_2k | 15:06:59 | 15:08:59 | 121 | 2.0 min |
| K562_T3_4k | 15:07:14 | 15:09:00 | 102 | 1.8 min |
| K562_T3_7k | 15:07:28 | 15:09:17 | 108 | 1.8 min |

- 平均: 2.1 min/dataset, 118 epochs
- 所有 dataset 在 ~7 分钟内全部完成 (并行)
- 早停在 epoch 86-164 触发 (patience=25)

### 插补指标 (PCC/MAE, cell-wise)

| 数据集 | PCC mean | PCC std | MAE mean | MAE std |
|--------|---------|---------|---------|---------|
| K562_T1_1k | 0.8663 | 0.0012 | 0.4683 | 0.0016 |
| K562_T1_2k | 0.9728 | 0.0004 | 0.2118 | 0.0013 |
| K562_T1_4k | 0.9896 | 0.0002 | 0.3830 | 0.0030 |
| K562_T1_7k | 0.9913 | 0.0002 | 0.5412 | 0.0031 |
| K562_T2_1k | 0.8938 | 0.0011 | 0.4716 | 0.0015 |
| K562_T2_2k | 0.9770 | 0.0003 | 0.2467 | 0.0013 |
| K562_T2_4k | 0.9915 | 0.0002 | 0.3106 | 0.0022 |
| K562_T2_7k | 0.9943 | 0.0001 | 0.6825 | 0.0033 |
| K562_T3_1k | 0.8855 | 0.0012 | 0.4771 | 0.0016 |
| K562_T3_2k | 0.9771 | 0.0002 | 0.2506 | 0.0012 |
| K562_T3_4k | 0.9920 | 0.0002 | 0.2799 | 0.0021 |
| K562_T3_7k | 0.9946 | 0.0001 | 0.6443 | 0.0043 |
| **平均** | **0.9605** | -- | **0.4139** | -- |

---

## 六、Batch 6: 1scHiC_from_scratch

### 训练耗时

| 数据集 | 起始时间 | 完成时间 | Epoch | 耗时 |
|--------|---------|---------|-------|------|
| K562_T1_1k | 11:22:58 | 11:25:17 | 145 | 2.3 min |
| K562_T1_2k | 10:56:21 | 10:59:48 | 125 | 3.5 min |
| K562_T1_4k | 10:56:21 | 10:59:39 | 114 | 3.3 min |
| K562_T1_7k | 10:56:22 | 10:59:55 | 133 | 3.5 min |
| K562_T2_1k | 11:22:58 | 11:24:58 | 122 | 2.0 min |
| K562_T2_2k | 10:56:22 | 10:59:24 | 94 | 3.0 min |
| K562_T2_4k | 10:56:22 | 10:59:50 | 126 | 3.5 min |
| K562_T2_7k | 10:56:22 | 10:59:44 | 119 | 3.4 min |
| K562_T3_1k | 11:22:58 | 11:25:02 | 127 | 2.1 min |
| K562_T3_2k | 11:06:17 | 11:08:29 | 137 | 2.2 min |
| K562_T3_4k | 10:56:23 | 10:59:32 | 102 | 3.1 min |
| K562_T3_7k | 11:08:36 | 11:11:32 | 192 | 2.9 min |

- 平均: 2.9 min/dataset, 128 epochs
- 分 3 波并行: 10:56 (9 dataset), 11:06 (1 dataset), 11:22 (3 dataset)
- 总 wall time: ~28 min
- 注意: 此版本无 EarlyStopping,实际停止 epoch (94-192) 低于 max_epochs=1000,原因可能是训练任务被手动终止或其他因素

### 插补指标 (PCC/MAE, cell-wise)

| 数据集 | PCC mean | PCC std | MAE mean | MAE std |
|--------|---------|---------|---------|---------|
| K562_T1_1k | 0.8668 | 0.0013 | 0.4687 | 0.0015 |
| K562_T1_2k | 0.9733 | 0.0004 | 0.2115 | 0.0014 |
| K562_T1_4k | 0.9900 | 0.0002 | 0.3808 | 0.0027 |
| K562_T1_7k | 0.9915 | 0.0002 | 0.5415 | 0.0032 |
| K562_T2_1k | 0.8936 | 0.0010 | 0.4710 | 0.0015 |
| K562_T2_2k | 0.9760 | 0.0003 | 0.2489 | 0.0013 |
| K562_T2_4k | 0.9918 | 0.0002 | 0.3107 | 0.0024 |
| K562_T2_7k | 0.9941 | 0.0001 | 0.6837 | 0.0037 |
| K562_T3_1k | 0.8875 | 0.0012 | 0.4761 | 0.0014 |
| K562_T3_2k | 0.9773 | 0.0003 | 0.2502 | 0.0013 |
| K562_T3_4k | 0.9920 | 0.0002 | 0.2786 | 0.0022 |
| K562_T3_7k | 0.9946 | 0.0001 | 0.6412 | 0.0036 |
| **平均** | **0.9607** | -- | **0.4136** | -- |

---

## 七、Batch 10 vs Batch 6 对比

| 指标 | Batch 10 (v5_scdiff_1mbsucess) | Batch 6 (1_scHiC 原始) | 差异 |
|------|------|------|------|
| 代码版本 | v5_scdiff_1mbsucess | 1_scHiC 原始 | 不同 |
| EarlyStopping | 有 (patience=25) | 无 | 不同 |
| PCC 平均 | 0.9605 | 0.9607 | ~0 (一致) |
| MAE 平均 | 0.4139 | 0.4136 | ~0 (一致) |
| 平均 epoch | 118 | 128 | Batch 10 略少 |
| 平均耗时 | 2.1 min | 2.9 min | Batch 10 更快 |
| 1k PCC | 0.8819 | 0.8826 | 一致 |

**结论**: 两个不同代码版本、不同早停策略的批次,插补指标几乎完全一致 (PCC 差异 < 0.001),说明 scHiC-Diff 在 HiCImputeData 上的插补结果是稳定可复现的。

---

## 八、输出文件说明

每个数据集目录下包含 4 个 NPZ 文件:

| 文件 | 含义 | shape | 说明 |
|------|------|-------|------|
| `raw_x.npz` | 原始输入 | (100, 1830) | observed, 有缺失, 无归一化 |
| `denoise_recon.npz` | 归一化插补结果 | (100, 1830) | 模型输出, 归一化空间 |
| `denoise_recon_inv.npz` | **反归一化插补结果** | (100, 1830) | 用于评估, 与 GT 同尺度 |
| `denoise_target.npz` | 测试加载器目标 | (100, 1830) | 观测输入矩阵的归一化副本，不是 GT |

所有 NPZ 为 scipy.sparse.csr_matrix, float32, numpy row-major 上三角顺序 (`np.triu_indices(61, k=1)`)。

`npz_lower_tri/` 目录:
- `{dataset}_scHiCDiff_imputed.npz` -- Batch 10 的 `denoise_recon_inv.npz` 标准化命名拷贝,用于指标计算

> GT is stored separately under `0_gtData/` and is used only by the external
> simulation-evaluation scripts. It is not an input, target, or label for
> scHiC-Diff training, validation, early stopping, or inference.

---

## 九、配置文件

### Batch 10 配置 (`output/configs/batch10_v5_1schic_like_bs128_lr2e4/`)

| 文件 | 说明 |
|------|------|
| `recon_masked.yaml` | 原始 config 文件 (来自 `v5_scdiff_1mbsucess/.../configs/`) |
| `2026-06-14T15-04-59-project.yaml` | 训练时配置快照 (数据/模型参数,含实际 save_path) |
| `2026-06-14T15-04-59-lightning.yaml` | 训练时 Lightning 配置快照 (trainer 参数) |

### Batch 6 配置 (`output/configs/batch6_1scHiC_from_scratch/`)

| 文件 | 说明 |
|------|------|
| `recon_masked.yaml` | 原始 config 文件 (来自 `1_scHiC/.../configs/`) |
| `2026-06-14T11-22-58-project.yaml` | 训练时配置快照 |
| `2026-06-14T11-22-58-lightning.yaml` | 训练时 Lightning 配置快照 |

---

## 十、训练日志

每个日志目录 (12 个,对应 12 个 dataset) 包含:
- `csv/version_0/metrics.csv` -- 逐 epoch 的 loss/metric 记录
- `csv/version_0/hparams.yaml` -- 超参数
- `configs/` -- 配置快照
- `checkpoints/` -- 模型 checkpoint (已清理,为空)

---

## 十一、GT 数据与指标计算

**GT**: `0_gtData/1_Gt_HiCImputeData/{dataset}_true.npz`
- shape=(100, 1830) sparse CSR, numpy row-major 上三角顺序

**指标计算**:
```bash
/public/home/hpc254701055/micromamba/envs/hic-impute/bin/python \
  /public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/paperplots/1_pccAndMae_all/hpc_cal_HiCImputeData_ALL_PCC_MAE.py \
  --method scHiCDiff
```

---

## 十二、原始数据源

| 内容 | 路径 |
|------|------|
| Batch 10 原始结果 | `1_scHiC/3_DiffusionModel/scHiC-Diff-master/results/1_New_Simu_Data/10_hpc_v5run_1schic_like_bs128_lr2e4_train_results/` |
| Batch 6 原始结果 | `1_scHiC/3_DiffusionModel/scHiC-Diff-master/results/1_New_Simu_Data/6_hpc_run_train_from_scratch_results/` |
| Batch 10 原始日志 | `v5_scdiff_1mbsucess/3_DiffusionModel/scHiC-Diff-master/logs/` |
| Batch 6 原始日志 | `1_scHiC/3_DiffusionModel/scHiC-Diff-master/logs/` |
| Batch 10 原始代码 | `v5_scdiff_1mbsucess/3_DiffusionModel/scHiC-Diff-master/` |
| Batch 6 原始代码 | `1_scHiC/3_DiffusionModel/scHiC-Diff-master/` |
| 5 批次指标 CSV | `1_scHiC/3_DiffusionModel/scHiC-Diff-master/results/1_New_Simu_Data/existing_5_batches_PCC_MAE.csv` |
