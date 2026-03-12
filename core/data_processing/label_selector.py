import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from enum import Enum
import re
import logging

# 配置日志，方便调试
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProblemType(Enum):
    """问题类型枚举"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    UNKNOWN = "unknown"


class LabelSelector:
    """
    标签选择器，用于标签选择、校验和问题类型检测
    """

    def __init__(self):
        # 明显无关列的关键词
        self.irrelevant_keywords = ['id', 'index', '序号', '编号', '行号']
        # 数值型数据类型
        self.numeric_dtypes = ['int64', 'float64', 'int32', 'float32', 'int16', 'float16']
        # 类别型数据类型
        self.categorical_dtypes = ['object', 'category', 'bool', 'str']

        # 优化日期正则：支持更多常见格式
        # 年份匹配（列名）：如2024年、year2024、2024_year
        self.year_col_pattern = re.compile(r'(\d{4,}|year|年)', re.IGNORECASE)
        # 日期值匹配：支持 2024-01-01、2024/01/01、2024.01.01、2024年01月01日 等
        self.date_value_pattern = re.compile(
            r'^\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}(日)?$|^\d{8}$',  # 补充8位数字日期（20240101）
            re.IGNORECASE
        )
        # 时间值匹配：支持 12:00:00、12:00 等
        self.time_value_pattern = re.compile(r'^\d{1,2}:\d{1,2}(:\d{1,2})?$', re.IGNORECASE)

    def detect_problem_type(self, data: pd.DataFrame, label_column: str) -> Tuple[ProblemType, str]:
        """
        检测问题类型

        Args:
            data: 数据框
            label_column: 标签列名

        Returns:
            (问题类型, 类型说明)
        """
        if label_column not in data.columns:
            raise ValueError(f"标签列不存在: {label_column}")

        label_series = data[label_column].dropna()
        if len(label_series) == 0:
            return ProblemType.UNKNOWN, "标签列全为空值"

        label_dtype = str(label_series.dtype)
        unique_values = label_series.nunique()
        total_values = len(label_series)

        # 数值型标签检测
        if label_dtype in self.numeric_dtypes:
            if unique_values > 10 or (total_values < 100 and unique_values / total_values > 0.5):
                return ProblemType.REGRESSION, "数值型标签，唯一值较多，判定为回归问题"
            else:
                return ProblemType.CLASSIFICATION, "数值型标签，唯一值较少，判定为分类问题"

        # 类别型标签检测
        elif label_dtype in self.categorical_dtypes:
            return ProblemType.CLASSIFICATION, "类别型标签，判定为分类问题"

        return ProblemType.UNKNOWN, f"无法判定问题类型，标签类型: {label_dtype}"

    def validate_label(self, data: pd.DataFrame, label_column: str) -> Dict[str, any]:
        """
        校验标签合法性

        Args:
            data: 数据框
            label_column: 标签列名

        Returns:
            校验结果

        Raises:
            ValueError: 标签校验失败
        """
        if label_column not in data.columns:
            raise ValueError(f"标签列不存在: {label_column}")

        label_series = data[label_column]

        if label_series.isnull().all():
            raise ValueError("标签列全为空值")

        unique_count = label_series.nunique()
        if unique_count == 0:
            raise ValueError("标签列无唯一值")

        problem_type, type_desc = self.detect_problem_type(data, label_column)

        if problem_type == ProblemType.REGRESSION:
            if str(label_series.dtype) not in self.numeric_dtypes:
                raise ValueError(f"回归问题标签必须是数值型，当前类型: {label_series.dtype}")
            if unique_count == 1:
                raise ValueError("回归问题标签只有一个唯一值，无法建模")

        elif problem_type == ProblemType.CLASSIFICATION:
            if unique_count > 100:
                raise ValueError(f"分类问题标签类别数过多({unique_count}类)，建议重新选择标签")

        # 转换numpy类型为Python原生类型
        def to_python_type(value):
            import numpy as np
            if isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                return float(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            else:
                return value
        
        # 转换value_counts结果中的numpy类型
        value_distribution = {}
        for k, v in label_series.value_counts().items():
            value_distribution[to_python_type(k)] = to_python_type(v)
        
        validation_result = {
            'label_column': label_column,  # 标签列名
            'problem_type': problem_type.value,  # 问题类型
            'type_description': type_desc,  # 判断依据
            'label_dtype': str(label_series.dtype),  # 标签数据类型
            'total_samples': int(len(label_series)),  # 标签总样本数量
            'non_null_samples': int(label_series.count()),  # 标签非空样本数量
            'null_samples': int(label_series.isnull().sum()),  # 标签空值数量
            'unique_values': int(unique_count),  # 标签非空值类别数
            'value_distribution': value_distribution  # 每个类别数量
        }

        return validation_result

    def suggest_relevant_columns(self, data: pd.DataFrame, label_column: str) -> List[str]:
        """
        推荐相关列，排除明显无关列

        Args:
            data: 数据框
            label_column: 标签列名

        Returns:
            推荐的相关列列表
        """
        relevant_columns = []
        for col in data.columns:
            if col == label_column:
                continue
            col_lower = col.lower()
            is_irrelevant = any(keyword in col_lower for keyword in self.irrelevant_keywords)
            if not is_irrelevant:
                relevant_columns.append(col)
        return relevant_columns

    def _is_datetime_column(self, data: pd.DataFrame, col: str) -> bool:
        """
        内部方法：判断单列是否为日期时间特征

        Args:
            data: 数据框
            col: 列名

        Returns:
            是否为日期时间特征
        """
        # 1. 原生datetime类型直接判定为日期特征
        if str(data[col].dtype) in ['datetime64[ns]', 'datetime64']:
            return True

        # 2. 非object类型跳过（日期字符串仅存在于object列）
        if str(data[col].dtype) not in ['object']:
            return False

        # 取非空值样本（避免空值干扰）
        non_null_vals = data[col].dropna()
        if len(non_null_vals) == 0:
            return False

        # 3. 列名包含年份关键词（如"2024年"、"year"）
        if self.year_col_pattern.search(col):
            logger.info(f"列[{col}]：列名包含年份关键词，判定为日期特征")
            return True

        # 4. 列值匹配日期/时间格式（抽样检测，提升性能）
        sample_size = min(100, len(non_null_vals))  # 抽样100个值检测，避免全量遍历
        sample_vals = non_null_vals.sample(sample_size, random_state=42).astype(str)

        # 统计匹配比例：超过50%则判定为日期特征
        date_match_count = sum(
            1 for val in sample_vals
            if self.date_value_pattern.match(val) or self.time_value_pattern.match(val)
        )
        if date_match_count / sample_size > 0.5:
            logger.info(f"列[{col}]：{date_match_count}/{sample_size}个样本匹配日期格式，判定为日期特征")
            return True

        return False

    def analyze_features(self, data: pd.DataFrame, label_column: str) -> Dict[str, List[str]]:
        """
        分析特征类型

        Args:
            data: 数据框
            label_column: 标签列名

        Returns:
            特征类型分析结果
        """
        features = self.suggest_relevant_columns(data, label_column)
        analysis = {
            'numeric_features': [],
            'categorical_features': [],
            'text_features': [],
            'datetime_features': []
        }

        for col in features:
            # 优先判断日期时间特征（避免被其他类型覆盖）
            if self._is_datetime_column(data, col):
                analysis['datetime_features'].append(col)
                continue

            col_dtype = str(data[col].dtype)
            unique_values = data[col].nunique()
            total_values = len(data[col])

            # 数值型特征
            if col_dtype in self.numeric_dtypes:
                analysis['numeric_features'].append(col)
            # 类别型/文本特征
            elif col_dtype in self.categorical_dtypes:
                # 文本特征：唯一值过多（>50 或 占样本10%以上）
                if unique_values > max(50, total_values * 0.1):
                    analysis['text_features'].append(col)
                else:
                    analysis['categorical_features'].append(col)
            # 其他类型默认归为文本特征
            else:
                analysis['text_features'].append(col)

        return analysis


if __name__ == "__main__":
    # 测试标签选择器（含日期特征测试）
    from data_loader import DataLoader

    loader = DataLoader()
    selector = LabelSelector()

    try:
        data = loader.load_data("d:\\wh\\AAAproject\\Algorithm\\Data\\个贷违约\\public.csv")
        print("数据加载成功")

        # 测试标签校验
        label_column = "class"
        validation_result = selector.validate_label(data, label_column)
        print(f"\n标签校验结果：")
        for k, v in validation_result.items():
            print(f"  {k}: {v}")

        # 测试特征分析
        features_analysis = selector.analyze_features(data, label_column)
        print(f"\n特征分析结果：")
        for feature_type, features in features_analysis.items():
            print(f"  {feature_type}: {len(features)}个特征")
            if features:
                print(f"    示例: {features[:5]}...")

    except Exception as e:
        logger.error(f"执行错误：{e}", exc_info=True)