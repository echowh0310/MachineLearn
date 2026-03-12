import os
import sys

# Add project root to Python path when running directly
if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, project_root)
    from core.data_processing.label_selector import LabelSelector, ProblemType
else:
    from .label_selector import LabelSelector, ProblemType

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional

class DatasetSplitter:
    """
    数据集划分器，用于数据集的分层拆分和比例调整
    """
    
    def __init__(self):
        self.label_selector = LabelSelector()
        self.random_seed = 42
    
    def split_dataset(self, data: pd.DataFrame, label_column: str, 
                     test_size: float = 0.2, val_size: float = 0.2, 
                     use_stratified: bool = True) -> Dict[str, pd.DataFrame]:
        """
        划分数据集
        
        Args:
            data: 原始数据框
            label_column: 标签列名
            test_size: 测试集比例 (0.1-0.3)
            val_size: 验证集相对于训练集的比例 (0.1-0.5)
            use_stratified: 是否使用分层采样
            
        Returns:
            划分后的数据集字典 {'train': train_data, 'val': val_data, 'test': test_data}
            划分后的数据集字典 {'train': train_data, 'val': val_data, 'test': test_data}
        """
        # 验证输入参数
        if test_size < 0.1 or test_size > 0.3:
            raise ValueError("测试集比例必须在0.1-0.3之间")
        
        if val_size < 0.1 or val_size > 0.5:
            raise ValueError("验证集比例必须在0.1-0.5之间")
        
        # 基础清洗：仅去重复行
        cleaned_data = data.drop_duplicates()
        
        # 检查清洗后的数据量
        if len(cleaned_data) == 0:
            raise ValueError("数据清洗后无样本")
        
        # 检测问题类型
        problem_type, _ = self.label_selector.detect_problem_type(cleaned_data, label_column)
        
        # 确定是否使用分层采样
        if use_stratified:
            # 检查是否适合分层采样
            label_series = cleaned_data[label_column].dropna()
            label_counts = label_series.value_counts()
            # 如果任何类别样本数小于5，则不适合分层采样
            if label_counts.min() < 5:
                use_stratified = False
        
        # 第一步：划分测试集
        if use_stratified:
            train_val_data, test_data = train_test_split(
                cleaned_data,
                test_size=test_size,
                random_state=self.random_seed,
                stratify=cleaned_data[label_column]
            )
        else:
            train_val_data, test_data = train_test_split(
                cleaned_data,
                test_size=test_size,
                random_state=self.random_seed
            )
        
        # 第二步：划分训练集和验证集
        if len(train_val_data) < 2:
            # 数据量太少，不划分验证集
            train_data = train_val_data
            val_data = pd.DataFrame()
        else:
            if use_stratified and len(train_val_data) > 10:
                train_data, val_data = train_test_split(
                    train_val_data,
                    test_size=val_size/(1-test_size),
                    random_state=self.random_seed,
                    stratify=train_val_data[label_column]
                )
            else:
                train_data, val_data = train_test_split(
                    train_val_data,
                    test_size=val_size/(1-test_size),
                    random_state=self.random_seed
                )
        
        # 校验并调整各数据集样本数
        datasets = self._validate_and_adjust_datasets(train_data, val_data, test_data, label_column)
        
        # 输出划分结果
        self._print_split_info(datasets, label_column)

        
        return datasets
    
    def _validate_and_adjust_datasets(self, train_data: pd.DataFrame, 
                                     val_data: pd.DataFrame, 
                                     test_data: pd.DataFrame, 
                                     label_column: str) -> Dict[str, pd.DataFrame]:
        """
        校验并调整各数据集样本数
        
        Args:
            train_data: 训练集
            val_data: 验证集
            test_data: 测试集
            label_column: 标签列名
            
        Returns:
            调整后的数据集字典
        """
        # 检查训练集样本数
        if len(train_data) == 0:
            raise ValueError("训练集样本数为0")
        
        # 检查测试集样本数
        if len(test_data) == 0:
            # 调整比例，减少测试集比例
            adjusted_test_size = 0.1
            print(f"测试集样本数为0，自动调整测试集比例为{adjusted_test_size}")
            
            combined = pd.concat([train_data, val_data])
            train_val_data, test_data = train_test_split(
                combined,
                test_size=adjusted_test_size,
                random_state=self.random_seed,
                stratify=combined[label_column] if len(combined) > 10 else None
            )
            
            # 重新划分训练集和验证集
            train_data, val_data = train_test_split(
                train_val_data,
                test_size=0.2,
                random_state=self.random_seed,
                stratify=train_val_data[label_column] if len(train_val_data) > 10 else None
            )
        
        # 检查验证集样本数
        if len(val_data) == 0 and len(train_data) > 5:
            # 从训练集划分一小部分作为验证集
            adjusted_val_size = 0.1
            print(f"验证集样本数为0，自动调整验证集比例为{adjusted_val_size}")
            
            train_data, val_data = train_test_split(
                train_data,
                test_size=adjusted_val_size,
                random_state=self.random_seed,
                stratify=train_data[label_column] if len(train_data) > 10 else None
            )
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def _print_split_info(self, datasets: Dict[str, pd.DataFrame], label_column: str):
        """
        打印数据集划分信息
        
        Args:
            datasets: 划分后的数据集字典
            label_column: 标签列名
        """
        print("\n数据集划分结果：")
        total_samples = sum(len(data) for data in datasets.values())
        
        for name, data in datasets.items():
            sample_count = len(data)
            ratio = sample_count / total_samples if total_samples > 0 else 0
            
            if sample_count > 0 and label_column in data.columns:
                label_dist = data[label_column].value_counts().to_dict()
                print(f"{name}: {sample_count} 样本 ({ratio:.1%})，标签分布: {label_dist} ")
            else:
                print(f"{name}: {sample_count} 样本 ({ratio:.1%})")
    
    def get_split_summary(self, datasets: Dict[str, pd.DataFrame], label_column: str) -> Dict[str, Any]:
        """
        获取数据集划分摘要
        
        Args:
            datasets: 划分后的数据集字典
            label_column: 标签列名
            
        Returns:
            划分摘要信息
        """
        summary = {
            'total_samples': sum(len(data) for data in datasets.values()),
            'split_details': {}
        }
        
        for name, data in datasets.items():
            sample_count = len(data)
            ratio = sample_count / summary['total_samples'] if summary['total_samples'] > 0 else 0
            
            details = {
                'sample_count': sample_count,
                'ratio': ratio
            }

            if sample_count > 0 and label_column in data.columns:
                details['label_distribution'] = data[label_column].value_counts().to_dict()
            
            summary['split_details'][name] = details

        return summary

if __name__ == "__main__":
    import sys
    import os
    
    # 将项目根目录添加到Python路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, project_root)
    
    # 使用绝对导入
    from core.data_processing.data_loader import DataLoader
    
    loader = DataLoader()
    splitter = DatasetSplitter()

    try:
        data = loader.load_data("d:\wh\AAAproject\Algorithm\Data\个贷违约\public.csv")
        print("数据加载成功")
        print(f"\n所有列名：{data.columns.tolist()}")
        # 测试数据集划分
        datasets = splitter.split_dataset(data, "class", test_size=0.3, val_size=0.4)
        
        # 测试获取划分摘要
        summary = splitter.get_split_summary(datasets, "class")
        print("\n划分摘要：")
        print(summary)

    except Exception as e:
        print(f"错误：{e}")
