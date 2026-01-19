#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据预处理验证工具

比较原始数据和预处理数据，验证预处理的正确性和一致性。

示例用法：
    python tools/validate_preprocessing.py \
        --original data/raw/sample.h5ad \
        --preprocessed data/preprocessed/sample_preprocessed.h5ad \
        --output validation_report.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import issparse


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif hasattr(obj, 'item'):  # 处理numpy标量
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DataValidator:
    """数据预处理验证器"""
    
    def __init__(self, original_data: ad.AnnData, preprocessed_data: ad.AnnData):
        self.original_data = original_data
        self.preprocessed_data = preprocessed_data
        self.validation_results = {}
        
    def check_basic_properties(self) -> Dict[str, Any]:
        """检查基本属性"""
        print("[DataValidator] Checking basic properties...")
        
        results = {
            'original_shape': self.original_data.shape,
            'preprocessed_shape': self.preprocessed_data.shape,
            'shape_match': self.original_data.shape == self.preprocessed_data.shape,
            'var_names_match': list(self.original_data.var_names) == list(self.preprocessed_data.var_names),
            'obs_names_match': list(self.original_data.obs_names) == list(self.preprocessed_data.obs_names),
        }
        
        return results
    
    def check_preprocessing_metadata(self) -> Dict[str, Any]:
        """检查预处理元数据"""
        print("[DataValidator] Checking preprocessing metadata...")
        
        results = {
            'has_preprocessing_info': 'preprocessing' in self.preprocessed_data.uns,
            'preprocessing_info': None
        }
        
        if results['has_preprocessing_info']:
            preprocessing_info = self.preprocessed_data.uns['preprocessing']
            results['preprocessing_info'] = preprocessing_info
            results['marked_as_preprocessed'] = preprocessing_info.get('preprocessed', False)
        
        return results
    
    def check_required_layers(self) -> Dict[str, Any]:
        """检查必需的layers"""
        print("[DataValidator] Checking required layers...")
        
        required_layers = ['counts', 'train_mask', 'valid_mask', 'test_mask']
        results = {
            'required_layers': required_layers,
            'present_layers': list(self.preprocessed_data.layers.keys()),
            'missing_layers': [],
            'all_layers_present': True
        }
        
        for layer in required_layers:
            if layer not in self.preprocessed_data.layers:
                results['missing_layers'].append(layer)
                results['all_layers_present'] = False
        
        return results
    
    def check_numerical_consistency(self) -> Dict[str, Any]:
        """检查数值一致性"""
        print("[DataValidator] Checking numerical consistency...")
        
        results = {}
        
        # 检查原始计数数据是否保留
        if 'counts' in self.preprocessed_data.layers:
            original_X = self.original_data.X
            preserved_counts = self.preprocessed_data.layers['counts']
            
            if issparse(original_X):
                original_X = original_X.toarray()
            if issparse(preserved_counts):
                preserved_counts = preserved_counts.toarray()
            
            # 计算差异
            diff = np.abs(original_X - preserved_counts)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            results['counts_preservation'] = {
                'max_difference': float(max_diff),
                'mean_difference': float(mean_diff),
                'counts_preserved': max_diff < 1e-10
            }
        
        # 检查预处理后的数据统计特征
        preprocessed_X = self.preprocessed_data.X
        if issparse(preprocessed_X):
            preprocessed_X = preprocessed_X.toarray()
        
        results['preprocessed_stats'] = {
            'mean': float(np.mean(preprocessed_X)),
            'std': float(np.std(preprocessed_X)),
            'min': float(np.min(preprocessed_X)),
            'max': float(np.max(preprocessed_X)),
            'nonzero_count': int(np.count_nonzero(preprocessed_X)),
            'sparsity': float(1.0 - np.count_nonzero(preprocessed_X) / preprocessed_X.size)
        }
        
        return results
    
    def check_mask_properties(self) -> Dict[str, Any]:
        """检查掩码属性"""
        print("[DataValidator] Checking mask properties...")
        
        results = {}
        
        if 'train_mask' in self.preprocessed_data.layers:
            train_mask = self.preprocessed_data.layers['train_mask']
            valid_mask = self.preprocessed_data.layers['valid_mask']
            test_mask = self.preprocessed_data.layers['test_mask']
            
            if issparse(train_mask):
                train_mask = train_mask.toarray()
            if issparse(valid_mask):
                valid_mask = valid_mask.toarray()
            if issparse(test_mask):
                test_mask = test_mask.toarray()
            
            total_elements = train_mask.size
            
            results['mask_stats'] = {
                'total_elements': int(total_elements),
                'train_mask_count': int(np.sum(train_mask)),
                'valid_mask_count': int(np.sum(valid_mask)),
                'test_mask_count': int(np.sum(test_mask)),
                'train_mask_ratio': float(np.sum(train_mask) / total_elements),
                'valid_mask_ratio': float(np.sum(valid_mask) / total_elements),
                'test_mask_ratio': float(np.sum(test_mask) / total_elements),
            }
            
            # 检查掩码重叠
            train_valid_overlap = np.sum(train_mask & valid_mask)
            results['mask_overlap'] = {
                'train_valid_overlap': int(train_valid_overlap),
                'no_overlap': train_valid_overlap == 0
            }
        
        return results
    
    def check_data_distribution(self) -> Dict[str, Any]:
        """检查数据分布"""
        print("[DataValidator] Checking data distribution...")
        
        results = {}
        
        # 获取预处理后的数据
        preprocessed_X = self.preprocessed_data.X
        if issparse(preprocessed_X):
            preprocessed_X = preprocessed_X.toarray()
        
        # 计算每个细胞的总计数
        cell_totals = np.sum(preprocessed_X, axis=1)
        
        results['cell_totals'] = {
            'mean': float(np.mean(cell_totals)),
            'median': float(np.median(cell_totals)),
            'std': float(np.std(cell_totals)),
            'min': float(np.min(cell_totals)),
            'max': float(np.max(cell_totals)),
            'zero_cells': int(np.sum(cell_totals == 0))
        }
        
        # 计算每个基因的总计数
        gene_totals = np.sum(preprocessed_X, axis=0)
        
        results['gene_totals'] = {
            'mean': float(np.mean(gene_totals)),
            'median': float(np.median(gene_totals)),
            'std': float(np.std(gene_totals)),
            'min': float(np.min(gene_totals)),
            'max': float(np.max(gene_totals)),
            'zero_genes': int(np.sum(gene_totals == 0))
        }
        
        return results
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常"""
        print("[DataValidator] Detecting anomalies...")
        
        anomalies = []
        
        # 检查是否有异常的细胞总计数
        preprocessed_X = self.preprocessed_data.X
        if issparse(preprocessed_X):
            preprocessed_X = preprocessed_X.toarray()
        
        cell_totals = np.sum(preprocessed_X, axis=1)
        
        # 使用Z-score检测异常细胞
        z_scores = np.abs(stats.zscore(cell_totals))
        outlier_threshold = 3.0
        outlier_cells = np.where(z_scores > outlier_threshold)[0]
        
        if len(outlier_cells) > 0:
            anomalies.append({
                'type': 'outlier_cells',
                'description': f'Found {len(outlier_cells)} cells with extreme total counts',
                'count': len(outlier_cells),
                'threshold': outlier_threshold,
                'cell_indices': outlier_cells.tolist()[:10]  # 只保存前10个
            })
        
        # 检查是否有全零的细胞或基因
        zero_cells = np.where(cell_totals == 0)[0]
        if len(zero_cells) > 0:
            anomalies.append({
                'type': 'zero_cells',
                'description': f'Found {len(zero_cells)} cells with zero total counts',
                'count': len(zero_cells),
                'cell_indices': zero_cells.tolist()[:10]
            })
        
        gene_totals = np.sum(preprocessed_X, axis=0)
        zero_genes = np.where(gene_totals == 0)[0]
        if len(zero_genes) > 0:
            anomalies.append({
                'type': 'zero_genes',
                'description': f'Found {len(zero_genes)} genes with zero total counts',
                'count': len(zero_genes),
                'gene_indices': zero_genes.tolist()[:10]
            })
        
        return anomalies
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """生成完整的验证报告"""
        print("[DataValidator] Generating validation report...")
        
        report = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'basic_properties': self.check_basic_properties(),
            'preprocessing_metadata': self.check_preprocessing_metadata(),
            'required_layers': self.check_required_layers(),
            'numerical_consistency': self.check_numerical_consistency(),
            'mask_properties': self.check_mask_properties(),
            'data_distribution': self.check_data_distribution(),
            'anomalies': self.detect_anomalies(),
        }
        
        # 计算总体验证状态
        validation_passed = (
            report['basic_properties']['shape_match'] and
            report['basic_properties']['var_names_match'] and
            report['basic_properties']['obs_names_match'] and
            report['required_layers']['all_layers_present'] and
            report['preprocessing_metadata']['has_preprocessing_info']
        )
        
        report['validation_summary'] = {
            'overall_status': 'PASSED' if validation_passed else 'FAILED',
            'validation_passed': validation_passed,
            'critical_issues': len([a for a in report['anomalies'] if a['type'] in ['zero_cells', 'zero_genes']]),
            'total_anomalies': len(report['anomalies'])
        }
        
        return report


def main():
    parser = argparse.ArgumentParser(description="数据预处理验证工具")
    
    parser.add_argument("--original", type=str, required=True, help="原始数据文件路径")
    parser.add_argument("--preprocessed", type=str, required=True, help="预处理数据文件路径")
    parser.add_argument("--output", type=str, default="validation_report.json", help="验证报告输出路径")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("数据预处理验证开始")
    print("=" * 60)
    print(f"原始数据: {args.original}")
    print(f"预处理数据: {args.preprocessed}")
    print(f"报告输出: {args.output}")
    print("=" * 60)
    
    # 加载数据
    print("加载原始数据...")
    original_data = ad.read_h5ad(args.original)
    print(f"原始数据形状: {original_data.shape}")
    
    print("加载预处理数据...")
    preprocessed_data = ad.read_h5ad(args.preprocessed)
    print(f"预处理数据形状: {preprocessed_data.shape}")
    
    # 执行验证
    validator = DataValidator(original_data, preprocessed_data)
    report = validator.generate_validation_report()
    
    # 保存报告
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    
    # 打印摘要
    print("=" * 60)
    print("验证结果摘要")
    print("=" * 60)
    summary = report['validation_summary']
    print(f"总体状态: {summary['overall_status']}")
    print(f"验证通过: {summary['validation_passed']}")
    print(f"严重问题: {summary['critical_issues']}")
    print(f"总异常数: {summary['total_anomalies']}")
    
    if args.verbose and summary['total_anomalies'] > 0:
        print("\n异常详情:")
        for anomaly in report['anomalies']:
            print(f"- {anomaly['type']}: {anomaly['description']}")
    
    print(f"\n详细报告已保存到: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()