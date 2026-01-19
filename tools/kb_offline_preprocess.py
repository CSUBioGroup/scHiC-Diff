#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
离线数据预处理脚本

将原始H5AD数据进行预处理（归一化、log1p变换、掩码生成），
生成可直接用于训练的预处理数据，避免训练时重复预处理。

示例用法：
    python tools/offline_preprocess.py \
        --input data/raw/sample.h5ad \
        --output data/preprocessed/sample_preprocessed.h5ad \
        --valid-split 0.1 \
        --test-split 0.1 \
        --mask-strategy none_zero \
        --mask-type mar \
        --seed 10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scdiff.utils.data import mask_data_offline


class OfflinePreprocessor:
    """离线数据预处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processing_stats = {}
        self.start_time = None
        
    def load_data(self, input_path: str) -> ad.AnnData:
        """加载原始数据"""
        print(f"[OfflinePreprocessor] Loading data from {input_path}")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        adata = ad.read_h5ad(input_path)
        print(f"[OfflinePreprocessor] Loaded data shape: {adata.shape}")
        
        # 记录原始统计信息
        self.processing_stats['original_shape'] = list(adata.shape)  # 转换为list避免tuple序列化问题
        self.processing_stats['original_nonzero'] = int(np.count_nonzero(adata.X.data if hasattr(adata.X, 'data') else adata.X))
        
        return adata
    
    def prepare_metadata(self, adata: ad.AnnData) -> ad.AnnData:
        """准备元数据"""
        print("[OfflinePreprocessor] Preparing metadata...")
        
        # 确保变量名唯一
        adata.var_names_make_unique()
        
        # 设置默认的细胞类型和批次信息
        if 'cell_type' not in adata.obs.columns:
            adata.obs['cell_type'] = 'unknown'
        if 'batch' not in adata.obs.columns:
            adata.obs['batch'] = 0
            
        # 保存原始计数数据
        if 'counts' not in adata.layers:
            adata.layers['counts'] = adata.X.copy()
            
        return adata
    
    def normalize_data(self, adata: ad.AnnData) -> ad.AnnData:
        """执行库大小归一化"""
        if not self.config.get('normalize', True):
            print("[OfflinePreprocessor] Skipping normalization as requested")
            return adata
            
        print("[OfflinePreprocessor] Performing library size normalization...")
        
        # 检查零深度细胞
        if hasattr(adata.X, 'toarray'):
            orig_x = adata.X.toarray()
        else:
            orig_x = adata.X
            
        depth_every_cell = np.array(np.sum(orig_x, axis=1)).flatten()
        zero_depth_cells = np.where(depth_every_cell == 0)[0]
        
        if len(zero_depth_cells) > 0:
            print(f"[OfflinePreprocessor] Found {len(zero_depth_cells)} zero-depth cells, removing them...")
            sc.pp.filter_cells(adata, min_genes=1)
            print(f"[OfflinePreprocessor] Remaining cells: {adata.shape[0]}")
            
            # 重新计算深度
            if hasattr(adata.X, 'toarray'):
                orig_x = adata.X.toarray()
            else:
                orig_x = adata.X
            depth_every_cell = np.array(np.sum(orig_x, axis=1)).flatten()
        
        # 计算目标深度
        # target_depth = float(np.median(depth_every_cell))
        # print(f"[OfflinePreprocessor] Target library depth: {target_depth}")
        # --- 修改开始: 允许强制指定 target_depth ---
        fixed_depth = self.config.get('target_depth', None)
        
        if fixed_depth is not None:
            target_depth = float(fixed_depth)
            print(f"[OfflinePreprocessor] Using FIXED target library depth: {target_depth}")
        else:
            target_depth = float(np.median(depth_every_cell))
            print(f"[OfflinePreprocessor] Using MEDIAN target library depth: {target_depth}")
        # --- 修改结束 ---
        
        # 执行归一化
        adata.X = adata.X.astype(np.float64)
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=target_depth)
        
        # 记录归一化统计信息
        self.processing_stats['target_depth'] = float(target_depth)
        self.processing_stats['cells_after_filtering'] = int(adata.shape[0])
        
        return adata
    
    def apply_log_transform(self, adata: ad.AnnData) -> ad.AnnData:
        """应用log1p变换"""
        if not self.config.get('log_transform', True):
            print("[OfflinePreprocessor] Skipping log transformation as requested")
            return adata
            
        print("[OfflinePreprocessor] Applying log1p transformation...")
        
        # 内存高效的log1p变换
        if hasattr(adata.X, 'data'):  # 稀疏矩阵
            processed_X = adata.X.copy()
            processed_X.data = np.log1p(processed_X.data)
            adata.X = processed_X
        else:  # 稠密矩阵
            adata.X = np.log1p(adata.X)
            
        return adata
    
    def generate_masks(self, adata: ad.AnnData) -> ad.AnnData:
        """生成训练/验证/测试掩码"""
        print("[OfflinePreprocessor] Generating train/validation/test masks...")
        
        # 设置分割比例
        splits = {
            'valid': self.config.get('valid_split', 0.1),
            'test': self.config.get('test_split', 0.1)
        }
        splits['train'] = 1.0 - splits['valid'] - splits['test']
        
        print(f"[OfflinePreprocessor] Split ratios - Train: {splits['train']:.2f}, "
              f"Valid: {splits['valid']:.2f}, Test: {splits['test']:.2f}")
        
        # 初始化split标签
        adata.obs['split'] = 'train'
        
        # 生成掩码
        mask_strategy = self.config.get('mask_strategy', 'none_zero')
        mask_type = self.config.get('mask_type', 'mar')
        seed = self.config.get('seed', 10)
        
        print(f"[OfflinePreprocessor] Mask strategy: {mask_strategy}, type: {mask_type}")
        
        train_mask, valid_mask = mask_data_offline(
            adata, 
            mask_strategy=mask_strategy,
            mask_type=mask_type,
            valid_mask_rate=splits['valid'],
            seed=seed
        )
        
        # 测试掩码（全部为True，表示测试时不掩盖任何数据）
        test_mask = np.ones(adata.shape, dtype=bool)
        
        # 保存掩码到layers
        adata.layers['train_mask'] = train_mask
        adata.layers['valid_mask'] = valid_mask
        adata.layers['test_mask'] = test_mask
        
        # 记录掩码统计信息
        self.processing_stats['train_mask_count'] = int(np.sum(train_mask))
        self.processing_stats['valid_mask_count'] = int(np.sum(valid_mask))
        self.processing_stats['test_mask_count'] = int(np.sum(test_mask))
        
        print(f"[OfflinePreprocessor] Train mask elements: {np.sum(train_mask)}")
        print(f"[OfflinePreprocessor] Valid mask elements: {np.sum(valid_mask)}")
        print(f"[OfflinePreprocessor] Test mask elements: {np.sum(test_mask)}")
        
        return adata
    
    def add_preprocessing_metadata(self, adata: ad.AnnData) -> ad.AnnData:
        """添加预处理元数据"""
        # 创建可序列化的配置副本
        serializable_config = {}
        for key, value in self.config.items():
            if key == 'adata':
                continue  # 跳过不可序列化的对象
            elif isinstance(value, (int, float, str, bool, list)):
                serializable_config[key] = value
            else:
                serializable_config[key] = str(value)  # 转换为字符串
        
        # 创建可序列化的统计信息副本
        serializable_stats = {}
        for key, value in self.processing_stats.items():
            if isinstance(value, (int, float, str, bool, list)):
                serializable_stats[key] = value
            elif isinstance(value, tuple):
                serializable_stats[key] = list(value)  # tuple转为list
            elif isinstance(value, np.ndarray):
                serializable_stats[key] = value.tolist()  # numpy数组转为list
            else:
                serializable_stats[key] = str(value)  # 其他类型转为字符串
        
        preprocessing_info = {
            'preprocessed': True,
            'preprocessing_version': '1.0',
            'parameters': serializable_config,
            'processing_stats': serializable_stats,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
            
        adata.uns['preprocessing'] = preprocessing_info
        return adata
    
    def save_processed_data(self, adata: ad.AnnData, output_path: str) -> None:
        """保存预处理后的数据"""
        print(f"[OfflinePreprocessor] Saving processed data to {output_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存数据
        adata.write_h5ad(output_path)
        
        # 记录最终统计信息
        self.processing_stats['final_shape'] = list(adata.shape)  # 转换为list避免tuple序列化问题
        self.processing_stats['output_file'] = str(output_path)
        
        print(f"[OfflinePreprocessor] Successfully saved processed data")
        print(f"[OfflinePreprocessor] Final shape: {adata.shape}")
    
    def generate_report(self) -> Dict[str, Any]:
        """生成处理报告"""
        processing_time = time.time() - self.start_time if self.start_time else 0
        
        # 创建可序列化的配置副本
        serializable_config = {}
        for key, value in self.config.items():
            if key == 'adata':
                continue  # 跳过不可序列化的对象
            elif isinstance(value, (int, float, str, bool, list)):
                serializable_config[key] = value
            else:
                serializable_config[key] = str(value)
        
        # 创建可序列化的统计信息副本
        serializable_stats = {}
        for key, value in self.processing_stats.items():
            if isinstance(value, (int, float, str, bool, list)):
                serializable_stats[key] = value
            elif isinstance(value, tuple):
                serializable_stats[key] = list(value)
            elif isinstance(value, np.ndarray):
                serializable_stats[key] = value.tolist()
            else:
                serializable_stats[key] = str(value)
        
        report = {
            'processing_time_seconds': float(processing_time),
            'config_used': serializable_config,
            'statistics': serializable_stats,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
            
        return report
    
    def process(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """执行完整的预处理流程"""
        self.start_time = time.time()
        
        try:
            # 1. 加载数据
            adata = self.load_data(input_path)
            
            # 2. 准备元数据
            adata = self.prepare_metadata(adata)
            
            # 3. 归一化
            adata = self.normalize_data(adata)
            
            # 4. Log变换
            adata = self.apply_log_transform(adata)
            
            # 5. 生成掩码
            adata = self.generate_masks(adata)
            
            # 6. 添加预处理元数据
            adata = self.add_preprocessing_metadata(adata)
            
            # 7. 保存数据
            self.save_processed_data(adata, output_path)
            
            # 8. 生成报告
            report = self.generate_report()
            
            print(f"[OfflinePreprocessor] Processing completed in {report['processing_time_seconds']:.2f} seconds")
            
            return report
            
        except Exception as e:
            print(f"[OfflinePreprocessor] Error during processing: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description="离线数据预处理脚本")
    
    # 必需参数
    parser.add_argument("--input", type=str, required=True, help="输入H5AD文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出H5AD文件路径")
    
    # 分割参数
    parser.add_argument("--valid-split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-split", type=float, default=0.1, help="测试集比例")
    
    # 掩码参数
    parser.add_argument("--mask-strategy", type=str, default="none_zero", help="掩码策略")
    parser.add_argument("--mask-type", type=str, default="mar", help="掩码类型")
    parser.add_argument("--seed", type=int, default=10, help="随机种子")
    
    # 预处理选项
    parser.add_argument("--no-normalize", action="store_true", help="跳过归一化")
    parser.add_argument("--no-log-transform", action="store_true", help="跳过log变换")
    
    # 输出选项
    parser.add_argument("--save-report", type=str, default=None, help="保存处理报告的路径")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    # --- 新增参数 ---
    parser.add_argument("--target-depth", type=float, default=None, 
                        help="强制指定归一化的目标深度 (例如 10000)。如果不指定，则使用中位数。")
    args = parser.parse_args()
    
    # 构建配置
    config = {
        'input_path': args.input,
        'output_path': args.output,
        'valid_split': args.valid_split,
        'test_split': args.test_split,
        'mask_strategy': args.mask_strategy,
        'mask_type': args.mask_type,
        'seed': args.seed,
        'normalize': not args.no_normalize,
        'log_transform': not args.no_log_transform,
        'verbose': args.verbose,
        'target_depth': args.target_depth,  # <--- 将新参数加入 config
    }
    
    # 验证参数
    if args.valid_split + args.test_split >= 1.0:
        raise ValueError("验证集和测试集比例之和不能大于等于1.0")
    
    print("=" * 60)
    print("离线数据预处理开始")
    print("=" * 60)
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"验证集比例: {args.valid_split}")
    print(f"测试集比例: {args.test_split}")
    print(f"掩码策略: {args.mask_strategy}")
    print(f"掩码类型: {args.mask_type}")
    print(f"随机种子: {args.seed}")
    print("=" * 60)
    
    # 执行预处理
    preprocessor = OfflinePreprocessor(config)
    report = preprocessor.process(args.input, args.output)
    
    # 保存报告
    if args.save_report:
        with open(args.save_report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"[OfflinePreprocessor] Report saved to {args.save_report}")
    
    print("=" * 60)
    print("离线数据预处理完成")
    print("=" * 60)


if __name__ == "__main__":
    main()