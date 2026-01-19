#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
离线预处理使用示例

演示如何使用离线预处理脚本和验证工具。
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_offline_preprocessing_example():
    """运行离线预处理示例"""
    
    print("=" * 80)
    print("离线预处理使用示例")
    print("=" * 80)
    
    # 示例数据路径（请根据实际情况修改）
    input_data = "data/raw/sample.h5ad"  # 原始数据路径
    output_data = "data/preprocessed/sample_preprocessed.h5ad"  # 预处理输出路径
    validation_report = "validation_report.json"  # 验证报告路径
    processing_report = "processing_report.json"  # 处理报告路径
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_data), exist_ok=True)
    
    print(f"输入数据: {input_data}")
    print(f"输出数据: {output_data}")
    print()
    
    # 检查输入文件是否存在
    if not os.path.exists(input_data):
        print(f"错误: 输入文件不存在: {input_data}")
        print("请确保数据文件存在，或修改脚本中的路径")
        return False
    
    # 步骤1: 执行离线预处理
    print("步骤1: 执行离线预处理")
    print("-" * 40)
    
    preprocess_cmd = [
        "python", "tools/offline_preprocess.py",
        "--input", input_data,
        "--output", output_data,
        "--valid-split", "0.1",
        "--test-split", "0.1",
        "--mask-strategy", "none_zero",
        "--mask-type", "mar",
        "--seed", "10",
        "--save-report", processing_report,
        "--verbose"
    ]
    
    print("执行命令:")
    print(" ".join(preprocess_cmd))
    print()
    
    try:
        result = subprocess.run(preprocess_cmd, check=True, capture_output=True, text=True)
        print("预处理完成!")
        if result.stdout:
            print("输出:")
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"预处理失败: {e}")
        if e.stderr:
            print("错误信息:")
            print(e.stderr)
        return False
    
    # 步骤2: 验证预处理结果
    print("\n步骤2: 验证预处理结果")
    print("-" * 40)
    
    validate_cmd = [
        "python", "tools/validate_preprocessing.py",
        "--original", input_data,
        "--preprocessed", output_data,
        "--output", validation_report,
        "--verbose"
    ]
    
    print("执行命令:")
    print(" ".join(validate_cmd))
    print()
    
    try:
        result = subprocess.run(validate_cmd, check=True, capture_output=True, text=True)
        print("验证完成!")
        if result.stdout:
            print("输出:")
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"验证失败: {e}")
        if e.stderr:
            print("错误信息:")
            print(e.stderr)
        return False
    
    # 步骤3: 展示如何在训练中使用预处理数据
    print("\n步骤3: 在训练中使用预处理数据")
    print("-" * 40)
    
    print("要在训练中使用预处理数据，只需修改配置文件中的数据路径:")
    print()
    print("原来的配置:")
    print("  data:")
    print("    params:")
    print("      train:")
    print("        params:")
    print(f"          fname: {input_data}")
    print()
    print("修改后的配置:")
    print("  data:")
    print("    params:")
    print("      train:")
    print("        params:")
    print(f"          fname: {output_data}")
    print()
    print("现有的训练代码无需任何修改即可使用预处理数据!")
    
    print("\n" + "=" * 80)
    print("示例完成!")
    print("=" * 80)
    print(f"预处理数据已保存到: {output_data}")
    print(f"处理报告已保存到: {processing_report}")
    print(f"验证报告已保存到: {validation_report}")
    
    return True


def show_performance_comparison():
    """展示性能对比示例"""
    
    print("\n" + "=" * 80)
    print("性能对比示例")
    print("=" * 80)
    
    print("使用原始数据训练:")
    print("- 每次训练都需要重复执行归一化和log1p变换")
    print("- 每个数据集(train/val/test)都会重复预处理")
    print("- 总预处理时间 = 单次预处理时间 × 3")
    print()
    
    print("使用预处理数据训练:")
    print("- 预处理只执行一次，离线完成")
    print("- 训练时直接加载预处理结果")
    print("- 显著减少训练初始化时间")
    print()
    
    print("预期性能改进:")
    print("- 训练初始化时间减少: 20-50%")
    print("- 内存使用减少: 避免重复的数据拷贝和处理")
    print("- 数值一致性: 所有数据集使用相同的预处理结果")


def main():
    """主函数"""
    
    # 切换到项目根目录
    os.chdir(project_root)
    
    print("当前工作目录:", os.getcwd())
    print()
    
    # 运行示例
    success = run_offline_preprocessing_example()
    
    if success:
        show_performance_comparison()
    else:
        print("示例执行失败，请检查错误信息并重试")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())