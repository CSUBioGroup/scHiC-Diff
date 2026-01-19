# 离线预处理工具使用指南

## 工具概览

本目录包含用于离线数据预处理的工具脚本：

| 脚本 | 用途 | 使用场景 |
|------|------|----------|
| `1_offline_preprocess.sh` | 批量预处理所有数据 | 首次预处理或更新所有数据 |
| `1_offline_preprocess_selective.sh` | 选择性预处理指定染色体 | 只需处理特定染色体 |
| `2_validate_preprocessing.sh` | 验证预处理结果 | 确保预处理正确性 |
| `3_test_mask_consistency.sh` | 测试 mask 一致性 | 验证 mask 生成的可重现性 |
| `offline_preprocess.py` | 核心预处理脚本 | 单文件预处理（被上述脚本调用） |
| `validate_preprocessing.py` | 核心验证脚本 | 单文件验证（被验证脚本调用） |

## 快速开始

### 1. 批量预处理所有数据

```bash
# 处理 data/h5ad_final_1000000/ 目录下的所有 .h5ad 文件
bash tools/1_offline_preprocess.sh
```

**特点**:
- ✅ 自动发现所有 `.h5ad` 文件
- ✅ 跳过已预处理的文件
- ✅ 跳过已存在的输出文件（避免重复处理）
- ✅ 显示处理进度和统计信息

**输出**:
```
处理文件 #1: hires_1Mb_chr1.h5ad
处理文件 #2: hires_1Mb_chr2.h5ad
...
总文件数: 24
成功处理: 24
处理失败: 0
```

### 2. 选择性预处理指定染色体

```bash
# 只处理 chr1, chr2, chr3
bash tools/1_offline_preprocess_selective.sh chr1 chr2 chr3

# 只处理 chr1
bash tools/1_offline_preprocess_selective.sh chr1

# 不指定参数时，处理所有文件（等同于 1_offline_preprocess.sh）
bash tools/1_offline_preprocess_selective.sh
```

**特点**:
- ✅ 灵活指定需要处理的染色体
- ✅ 自动构建文件名（`hires_1Mb_${chrom}.h5ad`）
- ✅ 检查文件是否存在
- ✅ 适合增量处理

**使用场景**:
- 新增了几个染色体的数据
- 某些染色体的预处理失败，需要重新处理
- 测试预处理流程

### 3. 验证预处理结果

```bash
# 验证所有预处理文件
bash tools/2_validate_preprocessing.sh
```

**验证内容**:
- ✅ 数据形状是否匹配
- ✅ counts 数据是否保留
- ✅ 必需的 layers 是否存在
- ✅ Mask 属性是否正确
- ✅ 数据分布是否合理
- ✅ 异常值检测

**输出**:
```
验证文件 #1: hires_1Mb_chr1_preprocessed.h5ad
✓ hires_1Mb_chr1_preprocessed.h5ad 验证通过

总文件数: 1
验证通过: 1
验证失败: 0
```

### 4. 测试 Mask 一致性

```bash
# 测试使用相同 seed 多次生成 mask 是否一致
bash tools/3_test_mask_consistency.sh
```

**测试内容**:
- ✅ 使用相同 seed 生成 3 次 mask
- ✅ 比较 3 次生成的 mask 是否完全一致
- ✅ 验证可重现性

**输出**:
```
✅ train_mask 在3次生成中完全一致
✅ valid_mask 在3次生成中完全一致
✅ 结论: 使用相同seed多次生成的mask完全一致
```

## 详细使用说明

### 配置参数

所有脚本使用相同的预处理参数（在脚本开头定义）：

```bash
# 预处理参数
VALID_SPLIT=0.2          # 验证集比例
TEST_SPLIT=0.1           # 测试集比例
MASK_STRATEGY="none_zero" # Mask 策略
MASK_TYPE="mar"          # Mask 类型
SEED=10                  # 随机种子
RESOLUTION="1Mb"         # 分辨率
```

**修改参数**:
1. 编辑脚本文件
2. 修改对应的变量值
3. 保存并重新运行

### 目录结构

```
data/
├── h5ad_final_1000000/              # 原始数据目录
│   ├── hires_1Mb_chr1.h5ad
│   ├── hires_1Mb_chr2.h5ad
│   └── ...
│   └── preprocessed/                # 预处理输出目录
│       ├── hires_1Mb_chr1_preprocessed.h5ad
│       ├── hires_1Mb_chr2_preprocessed.h5ad
│       └── ...
└── preprocessing_reports/           # 验证报告目录
    ├── hires_1Mb_chr1_report.json
    ├── hires_1Mb_chr1_preprocessed_validation_report.txt
    └── ...
```

### 单文件预处理

如果需要更精细的控制，可以直接使用 Python 脚本：

```bash
python tools/offline_preprocess.py \
    --input data/h5ad_final_1000000/hires_1Mb_chr1.h5ad \
    --output data/h5ad_final_1000000/preprocessed/hires_1Mb_chr1_preprocessed.h5ad \
    --seed 10 \
    --valid-split 0.2 \
    --test-split 0.1 \
    --mask-strategy none_zero \
    --mask-type mar \
    --save-report data/preprocessing_reports/chr1_report.json \
    --verbose
```

**参数说明**:
- `--input`: 输入文件路径（必需）
- `--output`: 输出文件路径（必需）
- `--seed`: 随机种子（默认: 10）
- `--valid-split`: 验证集比例（默认: 0.2）
- `--test-split`: 测试集比例（默认: 0.1）
- `--mask-strategy`: Mask 策略（默认: none_zero）
- `--mask-type`: Mask 类型（默认: mar）
- `--save-report`: 保存处理报告（可选）
- `--verbose`: 显示详细输出（可选）

### 单文件验证

```bash
python tools/validate_preprocessing.py \
    --original data/h5ad_final_1000000/hires_1Mb_chr1.h5ad \
    --preprocessed data/h5ad_final_1000000/preprocessed/hires_1Mb_chr1_preprocessed.h5ad \
    --output data/preprocessing_reports/chr1_validation.txt \
    --verbose
```

## 常见问题

### Q1: 如何重新处理已存在的文件？

**A**: 删除输出文件后重新运行：

```bash
# 删除特定文件
rm data/h5ad_final_1000000/preprocessed/hires_1Mb_chr1_preprocessed.h5ad

# 删除所有预处理文件
rm data/h5ad_final_1000000/preprocessed/*.h5ad

# 重新运行预处理
bash tools/1_offline_preprocess.sh
```

### Q2: 如何修改预处理参数？

**A**: 编辑脚本文件中的参数部分：

```bash
# 编辑脚本
vim tools/1_offline_preprocess.sh

# 修改参数
VALID_SPLIT=0.15  # 改为 15%
SEED=42           # 改为 42

# 保存并运行
bash tools/1_offline_preprocess.sh
```

### Q3: 预处理失败怎么办？

**A**: 检查错误信息并采取相应措施：

1. **内存不足**: 使用更大内存的机器或减少数据量
2. **文件不存在**: 检查输入路径是否正确
3. **权限问题**: 确保有读写权限
4. **Python 环境**: 确保所有依赖包已安装

```bash
# 查看详细错误信息
bash tools/1_offline_preprocess.sh 2>&1 | tee preprocess.log

# 检查 Python 环境
python -c "import anndata, numpy, scipy; print('OK')"
```

### Q4: 如何并行处理多个文件？

**A**: 使用 GNU Parallel 或手动启动多个进程：

```bash
# 方法 1: 使用 GNU Parallel
parallel bash tools/1_offline_preprocess_selective.sh ::: chr1 chr2 chr3 chr4

# 方法 2: 手动启动多个进程
bash tools/1_offline_preprocess_selective.sh chr1 &
bash tools/1_offline_preprocess_selective.sh chr2 &
bash tools/1_offline_preprocess_selective.sh chr3 &
wait
```

### Q5: 验证报告在哪里？

**A**: 验证报告保存在 `data/preprocessing_reports/` 目录：

```bash
# 查看所有报告
ls -lh data/preprocessing_reports/

# 查看特定报告（JSON 格式）
cat data/preprocessing_reports/hires_1Mb_chr1_preprocessed_validation_report.txt | python -m json.tool

# 提取关键信息
cat data/preprocessing_reports/hires_1Mb_chr1_preprocessed_validation_report.txt | \
    python -m json.tool | grep -A 5 "validation_summary"
```

## 工作流程

完整的预处理和验证流程：

```bash
# 1. 预处理数据
bash tools/1_offline_preprocess.sh

# 2. 验证预处理结果
bash tools/2_validate_preprocessing.sh

# 3. 测试 mask 一致性
bash tools/3_test_mask_consistency.sh

# 4. 使用预处理数据训练
bash run/local_run_hires_preprocessed.sh 10 chr1
```

## 性能提示

1. **批量处理**: 使用 `1_offline_preprocess.sh` 一次性处理所有文件
2. **增量处理**: 使用 `1_offline_preprocess_selective.sh` 只处理新增文件
3. **并行处理**: 对于大量文件，考虑并行处理
4. **磁盘空间**: 确保有足够的磁盘空间（预处理文件约为原始文件的 1.5-2 倍）
5. **内存使用**: 每个文件处理需要约 2-4GB 内存

## 参考文档

- [离线预处理详细文档](../docs/offline_preprocessing.md)
- [使用预处理数据训练](../docs/using_preprocessed_data.md)
- [测试指南](../RUN_TESTS.md)
