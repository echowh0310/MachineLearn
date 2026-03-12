import pandas as pd
from typing import Dict, Any, List, Tuple
import os
import sys

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import yaml

import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    precision_score, recall_score, f1_score
)



class DataLoader:

    def __init__(self) -> None:
        self.data_supported = ['.csv', '.xlsx', '.json']
        self.random_seed = None
        self.MIN_TEST_VAL_RATIO = 0.1 # 最小测试和验证数据集比例
        self.MAX_TEST_VAL_RATIO = 0.3  #最大测试和验证数据集比例
        self.MIN_STRATIFY_SAMPLES = 5  #每个类别的最小样本数，判断是否分层采样依据
        self.MIN_TRAIN_VAL_STRATIFY = 10 # 训练集和验证集分层采样的最小样本数
    # 1.加载数据判断格式是否正确
    def load_data(self, file_path: str) -> pd.DataFrame:

        if not os.path.exists(file_path):
            raise ValueError(f"该路径文件不存在：{file_path}")

        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in self.data_supported:
            raise ValueError(f"不支持文件格式：{file_extension}，仅支持文件格式：{self.data_supported}")

        try:
            if file_extension == '.csv':
                data = pd.read_csv(file_path)
            elif file_extension == '.xlsx':
                data = pd.read_excel(file_path)
            elif file_extension == '.json':
                data = pd.read_json(file_path)
            return data

        except Exception as e:
            raise ValueError(f"文件加载失败：{str(e)}")

    # 2.划分数据集，是否采用分层采样
    def split_dataset(self, data: pd.DataFrame, label_column: str,
                      test_size: float = 0.15, val_size: float = 0.15,
                      use_stratified: bool = True,random_seed: int = None) -> Dict[str, pd.DataFrame]:

        self.random_seed = random_seed

        # 验证数据集
        if test_size < self.MIN_TEST_VAL_RATIO or test_size > self.MAX_TEST_VAL_RATIO:
            raise ValueError(f"测试集比列必须在0.1-0.3之间")
        if val_size < self.MIN_TEST_VAL_RATIO or val_size > self.MAX_TEST_VAL_RATIO:
            raise ValueError(f"验证集必须在0.1-0.3之间")

        # 判断是否采用分层采样
        if use_stratified:
            # 检查是否符合分层采样
            label_series = data[label_column].dropna()
            label_counts = label_series.value_counts()
            # 判断类别数是否小于5，是否适合分层采样
            if label_counts.min() < self.MIN_STRATIFY_SAMPLES:
                use_stratified = False

        #调整后的验证集比列
        val_size_ad = val_size/(1-test_size)
        # 划分测试集
        if use_stratified:
            train_val_data, test_data = train_test_split(
                data,
                test_size=test_size,
                random_state=self.random_seed,
                stratify=data[label_column]
            )
        else:
            train_val_data, test_data = train_test_split(
                data,
                test_size=test_size,
                random_state=self.random_seed
            )
        # 划分训练集和验证集
        if len(train_val_data) < 2:
            # 数据量太小，不划分验证集
            train_data = train_val_data
            val_data = pd.DataFrame()  # 验证集设为空
        else:
            if use_stratified and len(train_val_data) > self.MIN_TRAIN_VAL_STRATIFY:
                train_data, val_data = train_test_split(
                    train_val_data,
                    test_size=val_size_ad,
                    random_state=self.random_seed,
                    stratify=train_val_data[label_column]
                )
            else:
                train_data, val_data = train_test_split(
                    train_val_data,
                    test_size=val_size_ad,
                    random_state=self.random_seed
                )

        datasets = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

        # 输出划分信息
        self._print_split_info(datasets, label_column)

        return datasets

    # 2.1打印数据集划分信息
    def _print_split_info(self, datasets: Dict[str, pd.DataFrame], label_column: str):
        print("\n数据集划分结果")
        # 遍历datasets.values()，计算每个数据集的长度
        total_samples = sum(len(dataset) for dataset in datasets.values())

        for name, data in datasets.items():
            sample_count = len(data)
            ratio = sample_count / total_samples if total_samples > 0 else 0

            if sample_count > 0 and label_column in data.columns:
                label_dist = data[label_column].value_counts().to_dict()
                print(f"{name}: {sample_count} 样本 ({ratio:.1%})，标签分布: {label_dist} ")
            else:
                print(f"{name}: {sample_count} 样本 ({ratio:.1%})")

    #3.加载配置文件参数
    def load_config(self,config_path:str):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"加载配置文件失败：{str(e)}", file=sys.stderr)
            sys.exit(1)
    #4.评估指标
    #4.1安全计算评估指标
    def safe_metric(self, metric_func, y_true, y_pred, **kwargs):
        try:
            return metric_func(y_true, y_pred, **kwargs)
        except Exception as e:
            print(f" 计算{metric_func.__name__}失败：{str(e)}")
            return 0.0

    #4.2绘制混淆矩阵
    def plot_confusion_matrix(self, y_true, y_pred, labels, title):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)

        # 显示数值
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        return plt

    #4.模型评估结果
    def evaluate_model(self, model, X, y, dataset_name, labels):
        if len(X) == 0 or len(y) == 0:
            print(f"\n{dataset_name}：无数据，跳过评估")
            return {}

        y_pred = model.predict(X)
        accuracy = self.safe_metric(accuracy_score, y, y_pred)

        print(f"\n{dataset_name}评估结果：")
        print(f"准确率: {accuracy:.4f}")
        print("\n分类报告：")
        try:
            print(classification_report(y, y_pred, target_names=[str(l) for l in labels]))
        except Exception as e:
            print(f" 生成分类报告失败：{str(e)}")

        # 计算详细指标
        metrics_dict = {
            'accuracy': accuracy,
            'precision': self.safe_metric(precision_score, y, y_pred, average='weighted'),
            'recall': self.safe_metric(recall_score, y, y_pred, average='weighted'),
            'f1': self.safe_metric(f1_score, y, y_pred, average='weighted')
        }

        # 绘制混淆矩阵
        plt = self.plot_confusion_matrix(y, y_pred, labels, f'{dataset_name}混淆矩阵')
        plt.savefig(f'{dataset_name}_confusion_matrix.png')
        plt.close()

        return metrics_dict
