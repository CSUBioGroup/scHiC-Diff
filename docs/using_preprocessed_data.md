# 使用离线预处理数据进行训练

## 概述

离线预处理功能允许你提前处理数据，从而：
- **加快训练初始化速度** (减少 20-50% 的初始化时间)
- **确保一致性**: 训练/验证/测试使用相同的 mask
- **节省内存**: 避免重复的数据处理
- **提高可重现性**: 固定的随机种子确保结果可重现

## 快速开始

### 1. 预处理数据

#### 方法 A: 批量预处理所有数据（推荐）

```bash
cd /path/to/scHiC-Diff-master
bash tools/1_offline_preprocess.sh
```

这会自动处理 `data/h5ad_final_1000000/` 目录下的所有 `.h5ad` 文件。

#### 方法 B: 预处理单个文件

```bash
python tools/offline_preprocess.py \
    --input data/h5ad_final_1000000/hires_1Mb_chr1.h5ad \
    --output data/h5ad_final_1000000/preprocessed/hires_1Mb_chr1_preprocessed.h5ad \
    --seed 10 \
    --valid-split 0.2 \
    --test-split 0.1 \
    --verbose
```

### 2. 验证预处理结果

```bash
# 验证所有预处理文件
bash tools/2_validate_preprocessing.sh

# 测试 mask 一致性
bash tools/3_test_mask_consistency.sh
```

### 3. 使用预处理数据训练

#### 在 HPC 集群上

```bash
# 提交作业
sbatch run/mb_gpu_hpc_run_hires_preprocessed.sh 10 chr1 chr2 chr3

# 或者使用默认参数 (seed=10, chr1)
sbatch run/mb_gpu_hpc_run_hires_preprocessed.sh
```

#### 在本地环境

```bash
# 训练单个染色体
bash run/local_run_hires_preprocessed.sh 10 chr1

# 训练多个染色体
bash run/local_run_hires_preprocessed.sh 10 chr1 chr2 chr3
```

## 文件结构

### 预处理前
```
data/h5ad_final_1000000/
├── hires_1Mb_chr1.h5ad
├── hires_1Mb_chr2.h5ad
└── ...
```

### 预处理后
```
data/h5ad_final_1000000/
├── hires_1Mb_chr1.h5ad                    # 原始文件
├── hires_1Mb_chr2.h5ad
├── ...
└── preprocessed/                           # 预处理文件目录
    ├── hires_1Mb_chr1_preprocessed.h5ad
    ├── hires_1Mb_chr2_preprocessed.h5ad
    └── ...
```

### 验证报告
```
data/preprocessing_reports/
├── hires_1Mb_chr1_preprocessed_validation_report.txt
├── hires_1Mb_chr2_preprocessed_validation_report.txt
└── ...
```

## 预处理数据格式

预处理后的 `.h5ad` 文件包含：

### Layers
- `counts`: 原始计数数据（稀疏矩阵）
- `train_mask`: 训练集 mask（布尔矩阵）
- `valid_mask`: 验证集 mask（布尔矩阵）
- `test_mask`: 测试集 mask（布尔矩阵）

### 主矩阵 (adata.X)
- 归一化并经过 log1p 变换的数据

### 元数据 (adata.uns)
```python
{
    'preprocessing_info': {
        'preprocessed': True,
        'preprocessing_version': '1.0',
        'timestamp': '2025-11-24 11:21:59',
        'parameters': {
            'seed': 10,
            'normalize': True,
            'log_transform': True,
            'mask_strategy': 'none_zero',
            'mask_type': 'mar',
            'valid_split': 0.2,
            'test_split': 0.1
        },
        'processing_stats': {
            'original_shape': [7469, 19306],
            'cells_after_filtering': 7469,
            'original_nonzero': 18326653,
            'target_depth': 17394.0,
            'train_mask_count': 140531184,
            'valid_mask_count': 3665330,
            'test_mask_count': 144196514
        }
    }
}
```

## 性能对比

### 使用原始数据
```
初始化时间: ~5-10 分钟
- 加载数据: 2 分钟
- 归一化: 2 分钟
- Log 变换: 1 分钟
- 生成 mask: 2 分钟
```

### 使用预处理数据
```
初始化时间: ~2-5 分钟
- 加载数据: 2 分钟
- 跳过归一化和变换
- 直接使用预生成的 mask
```

**性能提升**: 20-50% 的初始化时间减少

## 高级用法

### 自定义预处理参数

```bash
python tools/offline_preprocess.py \
    --input data/h5ad_final_1000000/hires_1Mb_chr1.h5ad \
    --output data/h5ad_final_1000000/preprocessed/hires_1Mb_chr1_preprocessed.h5ad \
    --seed 42 \
    --valid-split 0.15 \
    --test-split 0.15 \
    --mask-strategy none_zero \
    --mask-type mar \
    --target-depth 20000 \
    --verbose
```

### 参数说明

- `--seed`: 随机种子，确保可重现性（默认: 10）
- `--valid-split`: 验证集比例（默认: 0.2）
- `--test-split`: 测试集比例（默认: 0.1）
- `--mask-strategy`: Mask 策略
  - `none_zero`: 只 mask 非零元素（推荐）
  - `all`: Mask 所有元素
- `--mask-type`: Mask 类型
  - `mar`: Missing At Random（推荐）
  - `mcar`: Missing Completely At Random
- `--target-depth`: 归一化目标深度（默认: 自动计算）
- `--verbose`: 显示详细输出

### 批量预处理特定染色体

```bash
# 只预处理 chr1, chr2, chr3
for chr in chr1 chr2 chr3; do
    python tools/offline_preprocess.py \
        --input data/h5ad_final_1000000/hires_1Mb_${chr}.h5ad \
        --output data/h5ad_final_1000000/preprocessed/hires_1Mb_${chr}_preprocessed.h5ad \
        --seed 10
done
```

## 故障排除

### 问题 1: 找不到预处理文件

**错误信息**:
```
ERROR: Preprocessed file not found: /path/to/file_preprocessed.h5ad
```

**解决方案**:
```bash
# 检查文件是否存在
ls -lh data/h5ad_final_1000000/preprocessed/

# 如果不存在，运行预处理
bash tools/1_offline_preprocess.sh
```

### 问题 2: 验证失败

**错误信息**:
```
✗ hires_1Mb_chr1_preprocessed.h5ad 验证失败
```

**解决方案**:
```bash
# 查看详细验证报告
cat data/preprocessing_reports/hires_1Mb_chr1_preprocessed_validation_report.txt | python -m json.tool

# 重新预处理该文件
python tools/offline_preprocess.py \
    --input data/h5ad_final_1000000/hires_1Mb_chr1.h5ad \
    --output data/h5ad_final_1000000/preprocessed/hires_1Mb_chr1_preprocessed.h5ad \
    --seed 10 \
    --verbose
```

### 问题 3: 内存不足

**解决方案**:
- 使用更大内存的节点
- 一次只预处理一个文件
- 考虑使用更小的数据集进行测试

### 问题 4: Mask 不一致

**解决方案**:
```bash
# 运行 mask 一致性测试
bash tools/3_test_mask_consistency.sh

# 如果测试失败，检查随机种子是否一致
# 确保预处理和训练使用相同的 seed
```

## 最佳实践

1. **始终验证预处理结果**
   ```bash
   bash tools/2_validate_preprocessing.sh
   ```

2. **使用一致的随机种子**
   - 预处理和训练使用相同的 seed
   - 确保结果可重现

3. **保留原始数据**
   - 预处理文件存放在单独的目录
   - 不要覆盖原始数据

4. **定期检查验证报告**
   - 查看数据统计信息
   - 检测异常值

5. **版本控制**
   - 记录预处理参数
   - 保存验证报告
   - 便于追踪和复现

## 与原始流程的对比

### 原始流程
```
原始数据 → 训练脚本 → 每次训练时处理数据 → 训练
```

### 预处理流程
```
原始数据 → 离线预处理 → 预处理数据 → 训练脚本 → 直接训练
         ↓
      验证报告
```

## 参考

- [离线预处理详细文档](offline_preprocessing.md)
- [测试指南](../RUN_TESTS.md)
- [配置文件示例](../examples/offline_preprocessing_example.py)
