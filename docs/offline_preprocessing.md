# 离线数据预处理指南

## 概述

离线数据预处理功能允许您将数据预处理步骤（归一化、log1p变换、掩码生成）从训练过程中分离出来，提前完成并保存结果。这样可以显著提升训练效率，避免重复的预处理计算。

## 主要优势

1. **性能提升**: 训练初始化时间减少20-50%
2. **内存优化**: 避免重复的数据拷贝和处理
3. **数值一致性**: 所有数据集使用相同的预处理结果
4. **代码兼容性**: 现有训练代码无需修改

## 快速开始

### 1. 执行离线预处理

```bash
python tools/offline_preprocess.py \
    --input data/raw/sample.h5ad \
    --output data/preprocessed/sample_preprocessed.h5ad \
    --valid-split 0.1 \
    --test-split 0.1 \
    --mask-strategy none_zero \
    --mask-type mar \
    --seed 10 \
    --save-report processing_report.json \
    --verbose
```

### 2. 验证预处理结果

```bash
python tools/validate_preprocessing.py \
    --original data/raw/sample.h5ad \
    --preprocessed data/preprocessed/sample_preprocessed.h5ad \
    --output validation_report.json \
    --verbose
```

### 3. 在训练中使用预处理数据

只需修改配置文件中的数据路径：

```yaml
# 原来的配置
data:
  params:
    train:
      params:
        fname: data/raw/sample.h5ad

# 修改后的配置
data:
  params:
    train:
      params:
        fname: data/preprocessed/sample_preprocessed.h5ad
```

## 详细说明

### 预处理脚本参数

#### 必需参数
- `--input`: 输入原始H5AD文件路径
- `--output`: 输出预处理H5AD文件路径

#### 数据分割参数
- `--valid-split`: 验证集比例 (默认: 0.1)
- `--test-split`: 测试集比例 (默认: 0.1)
- 训练集比例 = 1.0 - valid_split - test_split

#### 掩码参数
- `--mask-strategy`: 掩码策略 (默认: "none_zero")
  - `none_zero`: 基于非零元素的掩码策略
  - `random`: 随机掩码策略
- `--mask-type`: 掩码类型 (默认: "mar")
  - `mar`: Missing At Random
  - `mnar`: Missing Not At Random
- `--seed`: 随机种子 (默认: 10)

#### 预处理选项
- `--no-normalize`: 跳过归一化步骤
- `--no-log-transform`: 跳过log1p变换步骤

#### 输出选项
- `--save-report`: 保存处理报告的路径
- `--verbose`: 显示详细输出

### 验证工具参数

- `--original`: 原始数据文件路径
- `--preprocessed`: 预处理数据文件路径
- `--output`: 验证报告输出路径
- `--verbose`: 显示详细输出

## 数据格式

### 输入数据要求

- 格式: H5AD (AnnData)
- 必需字段: `adata.X` (主数据矩阵)
- 可选字段: `adata.obs`, `adata.var` (元数据)

### 输出数据结构

预处理后的H5AD文件包含以下结构：

```python
adata.X                    # 预处理后的主数据矩阵 (归一化 + log1p)
adata.layers['counts']     # 原始计数数据
adata.layers['train_mask'] # 训练掩码
adata.layers['valid_mask'] # 验证掩码
adata.layers['test_mask']  # 测试掩码
adata.obs                  # 细胞元数据
adata.var                  # 基因元数据
adata.uns['preprocessing'] # 预处理参数和统计信息
```

## 工作流程

### 1. 预处理流程

```
原始数据 → 元数据准备 → 库大小归一化 → Log1p变换 → 掩码生成 → 保存结果
```

### 2. 验证流程

```
原始数据 + 预处理数据 → 基本属性检查 → 数值一致性检查 → 掩码属性检查 → 异常检测 → 生成报告
```

### 3. 训练流程

```
预处理数据 → 数据集类检测预处理状态 → 跳过重复预处理 → 直接使用预处理结果
```

## 性能对比

### 使用原始数据
- 每次训练重复执行预处理
- 每个数据集(train/val/test)都重复处理
- 总预处理时间 = 单次预处理时间 × 3

### 使用预处理数据
- 预处理只执行一次，离线完成
- 训练时直接加载预处理结果
- 显著减少训练初始化时间

## 故障排除

### 常见问题

1. **内存不足**
   - 使用 `--memory-efficient` 选项
   - 考虑分批处理大型数据集

2. **数据格式不兼容**
   - 确保输入数据为有效的H5AD格式
   - 检查数据是否包含必需的字段

3. **验证失败**
   - 检查原始数据和预处理数据的路径
   - 确保两个文件都存在且可读

4. **训练时无法识别预处理数据**
   - 确保预处理数据包含 `adata.uns['preprocessing']` 字段
   - 检查数据集类是否正确更新

### 调试技巧

1. 使用 `--verbose` 选项获取详细输出
2. 检查处理报告和验证报告
3. 使用验证工具比较原始数据和预处理数据

## 最佳实践

1. **数据备份**: 在预处理前备份原始数据
2. **参数记录**: 保存预处理参数以确保可重现性
3. **质量检查**: 始终运行验证工具检查预处理结果
4. **版本控制**: 对预处理脚本和配置进行版本控制
5. **文档记录**: 记录预处理步骤和参数选择的原因

## 示例

完整的使用示例请参考 `examples/offline_preprocessing_example.py`。

## 支持

如果遇到问题或需要帮助，请：

1. 检查本文档的故障排除部分
2. 查看示例代码和配置文件
3. 运行验证工具检查数据质量