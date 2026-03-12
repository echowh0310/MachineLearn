import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    precision_score, recall_score, f1_score
)
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl

class DataLoader:
    """
    数据加载器，支持CSV/Excel/JSON格式数据的读取和验证
    """

    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.json']

    def load_data(self, file_path: str, sheet_name: str = 0) -> pd.DataFrame:
        """
        加载数据文件

        Args:
            file_path: 文件路径

        Returns:
            加载后的数据框

        Raises:
            ValueError: 文件格式不支持或文件不存在
        """
        if not os.path.exists(file_path):
            raise ValueError(f"文件不存在: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {file_ext}，仅支持{self.supported_formats}")

        try:
            if file_ext == '.csv':
                data = pd.read_csv(file_path, low_memory=False)
            elif file_ext == '.xlsx':
                data = pd.read_excel(file_path, sheet_name=sheet_name)
            elif file_ext == '.json':
                data = pd.read_json(file_path)

            return data
        except Exception as e:
            raise ValueError(f"文件读取失败: {str(e)}")

    def detect_outlier_single_col(self, data: pd.DataFrame, col, method="iqr", threshold=1.5, plot_flag=True):
        """
        检测单个数值列的异常值
        :param data: 待处理的DataFrame
        :param col: 单个待检测列名
        :param method: 异常值检测方法，可选"iqr"/"std"/"z_score"
        :param threshold: 检测阈值（IQR=1.5/3；std/z_score=3）
        :param plot_flag: 是否绘制异常值散点图（True/False）
        :return: 该列的异常值信息字典、原始数据副本
        """

        data = data.copy()
        outliers_info_col = {}
        # 列全为空校验
        if data[col].isnull().all():
            print(f"警告：列{col}全为空值，跳过异常值检测")
            return outliers_info_col, data

        # 校验列类型（仅处理数值型）
        if data[col].dtype not in ["int64", "float64"]:
            print(f"警告：列{col}非数值型，跳过异常值检测")
            return outliers_info_col, data

        # 异常值检测逻辑
        lower_bound = None
        upper_bound = None
        is_outliers = np.zeros(len(data), dtype=bool)

        if not isinstance(threshold, (int, float)):
            raise ValueError(f"阈值必须为数值类型，当前：{type(threshold)}")

        if method == "iqr":
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            is_outliers = (data[col] < lower_bound) | (data[col] > upper_bound)

        elif method == "std":
            mean_val = data[col].mean()
            std_val = data[col].std()
            lower_bound = mean_val - threshold * std_val
            upper_bound = mean_val + threshold * std_val
            is_outliers = (data[col] < lower_bound) | (data[col] > upper_bound)

        elif method == "z_score":
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val == 0:
                is_outliers = np.zeros(len(data), dtype=bool)
            else:
                data_z = (data[col] - mean_val) / std_val
                is_outliers = abs(data_z) > threshold
            lower_bound = mean_val - threshold * std_val if std_val != 0 else mean_val
            upper_bound = mean_val + threshold * std_val if std_val != 0 else mean_val

        else:
            raise ValueError("仅支持iqr/std/z_score三种检测方法")

        # 记录该列异常值信息
        outlier_idx = data[is_outliers].index
        outliers_info_col[col] = {
            "count": len(outlier_idx),
            "ratio": len(outlier_idx) / len(data),
            "range": (lower_bound, upper_bound),
            "nulloutliers_range": f"({lower_bound}~{upper_bound})",
            "index": outlier_idx.tolist(),
            "is_outliers": is_outliers.copy(),
            "method": method,
            "threshold": threshold
        }

        # # 可视化该列处理前的异常值散点图
        # if plot_flag:
        #     plt.figure(figsize=(10, 6))
        #     # 正常值
        #     plt.scatter(
        #         data.index[~is_outliers],
        #         data[col][~is_outliers],
        #         color="blue", alpha=0.6,
        #         label=f"正常值（{len(data) - len(outlier_idx)}个）"
        #     )
        #     # 异常值
        #     plt.scatter(
        #         data.index[is_outliers],
        #         data[col][is_outliers],
        #         color="red", alpha=0.8,
        #         label=f"异常值（{len(outlier_idx)}个）"
        #     )
        #     # 上下界线
        #     plt.axhline(y=lower_bound, color="grey", linestyle="--", linewidth=1.5,
        #                 label=f"下界 ({lower_bound:.4f})")
        #     plt.axhline(y=upper_bound, color="black", linestyle="--", linewidth=1.5,
        #                 label=f"上界 ({upper_bound:.4f})")
        #     plt.title(f"{col} - 异常值检测结果（处理前）", fontsize=12)
        #     plt.xlabel("数据行索引")
        #     plt.ylabel(col)
        #     plt.legend(loc="best")
        #     plt.grid(axis="y", alpha=0.3)
        #     plt.tight_layout()
        #     plt.show()

        return outliers_info_col, data

    def valid_data_col(self, data: pd.DataFrame, columns):
        if data is None:
            raise ValueError("data数据集不能为空")
        if columns is None:
            target_columns = data.columns.tolist()
        else:
            if not isinstance(columns, list):
                target_columns = [columns]
            else:
                target_columns = columns.copy()
        # 列名去空格
        target_columns = [col.strip() for col in target_columns if isinstance(col, str)]
        # 过滤掉不存在的列
        valid_columns = [col for col in target_columns if col in data.columns]
        invalid_columns = [col for col in target_columns if col not in data.columns]

        if invalid_columns:
            print(f"以下列不存在数据集中，已跳过：{invalid_columns}")
        if not valid_columns:
            print("无有效列可处理，直接返回原数据")

        return data, valid_columns

    def validate_data(self, data: pd.DataFrame, outliers_method, outliers_threshold) -> Dict[str, Any]:
        """
        验证数据质量

        Args:
            data: 数据框
            outliers_method: 异常值检测方法（iqr/std/z_score）
            outliers_threshold: 异常值检测阈值

        Returns:
            数据质量报告
        """
        if data is None or data.empty:
            raise ValueError("data数据集不能为空或空数据框")

        if len(data.columns) != len(set(data.columns)):
            duplicate_cols = [col for col in data.columns if data.columns.tolist().count(col) > 1]
            print(f"警告：存在重复列名：{duplicate_cols}")

        report = {
            'total_rows': int(len(data)),
            'total_columns': int(len(data.columns)),
            'duplicate_rows': int(data.duplicated().sum()),
            'columns_info': {}
        }
        # 校验数据集和目标列
        columns = data.columns.tolist()
        data, valid_columns = self.valid_data_col(data, columns)
        for col in valid_columns:
            outliers_info_col = {
                'dtype': str(data[col].dtype),
                'non_null_count': int(data[col].count()),
                'null_count': int(data[col].isnull().sum()),
                'unique_values': int(data[col].nunique()),
                'sample_values': data[col].head(5).tolist(),
                'outliers_count': 0,
                'null_outliers_range': None,
                'nulloutliers_range': None,

            }

            # 仅处理数值列
            if str(data[col].dtype) in ['int64', 'float64']:
                if outliers_info_col['non_null_count'] > 0:
                    try:
                        # 异常值检测
                        outliers_info, _ = self.detect_outlier_single_col(
                            data=data,
                            col=col,
                            method=outliers_method,
                            threshold=outliers_threshold,
                            plot_flag=False
                        )
                        if outliers_info and col in outliers_info:
                            outliers_info_col['outliers_count'] = outliers_info[col]['count']
                            outliers_info_col['null_outliers_range'] = outliers_info[col]['range']
                            outliers_info_col['nulloutliers_range'] = outliers_info[col]['nulloutliers_range']

                    except Exception as e:
                        # 检测失败时容错，避免整体流程中断
                        print(f"警告：列{col}异常值检测失败：{str(e)}")
                        outliers_info_col['outliers_count'] = 0
                        outliers_info_col['null_outliers_range'] = None
                        outliers_info_col['nulloutliers_range'] = None

                # 补充数值列统计信息（空值容错）
                outliers_info_col['mean'] = float(data[col].mean()) if not pd.isna(data[col].mean()) else None
                outliers_info_col['median'] = float(data[col].median()) if not pd.isna(data[col].median()) else None
                outliers_info_col['min'] = float(data[col].min()) if not pd.isna(data[col].min()) else None
                outliers_info_col['max'] = float(data[col].max()) if not pd.isna(data[col].max()) else None
            else:
                outliers_info_col['info'] = "文本字段"
                total_non_null = outliers_info_col['non_null_count']
                outliers_info_col['unique_ratio'] = float(
                    data[col].nunique() / total_non_null) if total_non_null > 0 else 0.0

            # 转换numpy类型为原生Python类型（避免序列化问题）
            outliers_info_col['sample_values'] = [self._to_python_type(val) for val in
                                                  outliers_info_col['sample_values']]
            report['columns_info'][col] = outliers_info_col

        return report

    def process_single_col_null(self, process_style, fill_context, col, data: pd.DataFrame):
        """
        处理单个列的缺失值
        参数说明：
        - process_style: 处理方法
        - fill_context: 指定值填充时的填充值（仅fill_cols类型必填）
        - col: 需要处理的单个目标列
        - data: 待处理的DataFrame
        返回：
        - 处理后的DataFrame
        """
        # 校验列是否存在且有效
        if col not in data.columns:
            print(f"列{col}不存在于数据集中，跳过缺失值处理")
            return data

        if data[col].isnull().sum() == 0:
            print(f"列{col}：无缺失值，无需处理")
            return data

        # ========== 单列缺失值处理逻辑 ==========
        if process_style == "missing_all_cols":  # 删除该列包含缺失值的行
            row_count_before = len(data)
            data = data.dropna(subset=[col], axis=0)
            del_count = row_count_before - len(data)
            print(f"列{col}：已删除含缺失值的行 {del_count} 行，剩余行数：{len(data)}")

        elif process_style == "missing_50%_cols":  # 检查该列缺失率>50%则删除列
            missing_ratio = data[col].isnull().sum() / len(data)
            if missing_ratio > 0.5:
                print(f"列{col}：缺失率{missing_ratio:.2%} > 50%，已删除该列")
                data = data.drop(columns=[col])
            else:
                print(f"列{col}：缺失率{missing_ratio:.2%} ≤ 50%，无需删除")

        elif process_style == "mean_cols":  # 均值填充（仅数值型列）
            if data[col].dtype in ["int64", "float64"]:
                mean_value = data[col].mean()
                data[col] = data[col].fillna(mean_value)
                print(f"列{col}：数值型，均值填充，填充值={mean_value:.4f}")
            else:
                print(f"列{col}：非数值型，跳过均值填充")

        elif process_style == "mode_cols":  # 众数填充
            mode_series = data[col].mode()
            if mode_series.empty:
                mode_value = data[col].iloc[0] if not data[col].empty else "未知"
                print(f"列{col}：无众数，使用第一行值填充：{mode_value}")
            else:
                mode_value = mode_series[0]
            data[col] = data[col].fillna(mode_value)
            print(f"列{col}：众数填充，填充值={mode_value}")

        elif process_style == "median_cols":  # 中位数填充（仅数值型列）
            if data[col].dtype in ["int64", "float64"]:
                median_value = data[col].median()
                data[col] = data[col].fillna(median_value)
                print(f"列{col}：数值型，中位数填充，填充值={median_value:.4f}")
            else:
                print(f"列{col}：非数值型，跳过中位数填充")

        elif process_style == "fill_cols":  # 指定值填充
            if fill_context is None:
                raise ValueError(f"列{col}使用指定值填充时，参数fill_context不能为空")
                # 校验填充值类型与列匹配
            if not isinstance(fill_context, type(data[col].dropna().iloc[0]) if not data[col].dropna().empty else str):
                print(f"警告：填充值类型与列{col}不匹配，尝试强制转换")
                try:
                    fill_context = type(data[col].dropna().iloc[0])(fill_context)
                except:
                    raise ValueError(f"填充值{fill_context}无法转换为列{col}的类型{data[col].dtype}")
            data[col] = data[col].fillna(fill_context)
            print(f"列{col}：指定值填充，填充值={fill_context}")

        else:
            raise ValueError(
                f"列{col}：不支持的处理类型：{process_style}，"
                f"可选类型：missing_all_cols/missing_50%_cols/mean_cols/mode_cols/median_cols/fill_cols"
            )

        return data

    def process_outlier_single_col(self, data_processed, col, outliers_info_col,
                                   process_outliers_method, fill_value):
        """
        处理单个数值列的异常值
        :param data_processed: 检测后的DataFrame
        :param col: 单个待处理列名
        :param outliers_info_col: 该列的异常值信息字典
        :param process_outliers_method: 异常值处理方法（del_outlier/save_outlier/fill_outlier）
        :param fill_value: 填充异常值时的自定义值（None则用上下界填充）
        :return: 更新后的该列异常值信息字典、处理后的数据集
        """
        # 1. 基础校验（列存在性 + 数值类型）
        if col not in data_processed.columns:
            print(f"警告：列{col}不存在于数据集，跳过异常值处理")
            return outliers_info_col, data_processed
        if data_processed[col].dtype not in ["int64", "float64"]:
            print(f"警告：列{col}非数值型（{data_processed[col].dtype}），跳过异常值处理")
            return outliers_info_col, data_processed

        # 2. 异常值范围获取
        try:
            # 优先取标准化键null_outliers_range
            outliers_range = outliers_info_col.get("null_outliers_range")
            # 空值/格式校验
            if outliers_range is None:
                print(f"警告：列{col}无异常值范围信息，跳过异常值处理")
                return outliers_info_col, data_processed
            if not isinstance(outliers_range, (list, tuple)) or len(outliers_range) != 2:
                raise ValueError(f"异常值范围需为长度2的列表/元组，当前：{outliers_range}")

            lower_bound, upper_bound = outliers_range
            # 数值类型校验
            if not isinstance(lower_bound, (int, float)) or not isinstance(upper_bound, (int, float)):
                raise ValueError(f"异常值范围需为数值类型，当前：下界={lower_bound}，上界={upper_bound}")

        except Exception as e:
            print(f"警告：列{col}异常值范围解析失败：{str(e)}，跳过异常值处理")
            return outliers_info_col, data_processed

        # 3. 计算异常值索引
        try:
            # 排除空值后计算异常值
            non_null_mask = ~data_processed[col].isnull()
            is_outliers = pd.Series(False, index=data_processed.index)
            is_outliers[non_null_mask] = (data_processed.loc[non_null_mask, col] < lower_bound) | \
                                         (data_processed.loc[non_null_mask, col] > upper_bound)
            outlier_idx = data_processed[is_outliers].index
            outlier_count = len(outlier_idx)

            if outlier_count == 0:
                print(f"列{col}：无异常值，无需处理")
                return outliers_info_col, data_processed

        except Exception as e:
            print(f"警告：列{col}异常值索引计算失败：{str(e)}，跳过异常值处理")
            return outliers_info_col, data_processed

        # 4. 异常值处理逻辑（核心逻辑）
        if process_outliers_method == "del_outlier":
            # 删除异常值行（重置索引）
            data_processed = data_processed[~is_outliers].reset_index(drop=True)
            # 更新异常值信息（注意：原代码此处有bug，应修改outliers_info_col本身，而非outliers_info_col[col]）
            if col in outliers_info_col:
                outliers_info_col[col]["count"] = 0
            else:
                outliers_info_col[col] = {"count": 0}
            print(f"列{col}：已删除{outlier_count}个异常值行，剩余行数：{len(data_processed)}")

        elif process_outliers_method == "save_outlier":
            # 保留异常值，仅打印信息
            print(f"列{col}：保留{outlier_count}个异常值，不做处理")

        elif process_outliers_method == "fill_outlier":
            # 填充异常值（容错逻辑）
            if fill_value is None:
                # 用上下界填充
                fill_lower = (data_processed[col] < lower_bound) & non_null_mask
                fill_upper = (data_processed[col] > upper_bound) & non_null_mask
                data_processed.loc[fill_lower, col] = lower_bound
                data_processed.loc[fill_upper, col] = upper_bound
                print(
                    f"列{col}：用上下界填充{outlier_count}个异常值（下界={lower_bound:.4f}，上界={upper_bound:.4f}）")
            else:
                # 自定义值填充（类型强制转换 + 校验）
                try:
                    fill_value = float(fill_value)  # 强制转换为数值类型
                    data_processed.loc[is_outliers, col] = fill_value
                    print(f"列{col}：用自定义值{fill_value}填充{outlier_count}个异常值")
                except (TypeError, ValueError):
                    raise TypeError(f"列{col}是数值类型，填充值{fill_value}必须为数字")
        else:
            raise ValueError("仅支持del_outlier/save_outlier/fill_outlier三种处理方法")

        return outliers_info_col, data_processed

    def _to_python_type(self, val):
        """将numpy类型转换为原生Python类型"""
        import numpy as np
        if isinstance(val, dict):
            new_dict = {}
            for k, v in val.items():
                new_k = int(k) if isinstance(k, (np.integer, np.int64)) else k
                new_k = float(new_k) if isinstance(new_k, (np.floating, np.float64)) else new_k
                new_dict[new_k] = self._to_python_type(v)
            return new_dict
        elif isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        elif isinstance(val, (np.floating, np.float64, np.float32)):
            return float(val)
        elif isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, dict):
            return {key: self._to_python_type(value) for key, value in val.items()}
        elif isinstance(val, list):
            return [self._to_python_type(item) for item in val]
        else:
            return val

    def basic_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        基础数据清洗，仅去重复行

        Args:
            data: 原始数据框

        Returns:
            清洗后的数据框
        """
        if data.empty:
            print("警告：数据为空，无需清洗")
            return data
        initial_rows = len(data)
        cleaned_data = data.drop_duplicates()
        removed_rows = initial_rows - len(cleaned_data)

        if removed_rows > 0:
            print(f"基础清洗：移除了 {removed_rows} 行重复数据")

        return cleaned_data

    # 评估指标
    # 安全计算评估指标
    def safe_metric(self, metric_func, y_true, y_pred, **kwargs):
        try:
            #长度校验
            if len(y_true) != len(y_pred):
                print(f"指标计算失败：y_true和y_pred长度不一致（{len(y_true)} vs {len(y_pred)}）")
                return 0.0
            return metric_func(y_true, y_pred, **kwargs)
        except Exception as e:
            print(f" 计算{metric_func.__name__}失败：{str(e)}")
            return 0.0

    # 绘制混淆矩阵
    def plot_confusion_matrix(self, y_true, y_pred, labels, title):
        try:
            # 自动适配系统字体
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows
            # plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # Linux
            # plt.rcParams['font.sans-serif'] = ['Heiti TC']  # Mac
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
        except:
            # 使用系统默认支持的字体
            mpl.rcParams['font.family'] = 'sans-serif'
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

    # 模型评估结果
    def evaluate_model(self, model, X, y, dataset_name, labels):
        if len(X) == 0 or len(y) == 0:
            print(f"\n{dataset_name}：无数据，跳过评估")
            return {}

        y_pred = model.predict(X)
        accuracy = round(self.safe_metric(accuracy_score, y, y_pred), 4)

        print(f"\n{dataset_name}评估结果：")
        print(f"准确率: {accuracy:.4f}")
        print("\n分类报告：")
        try:
            print(classification_report(y, y_pred, target_names=[str(l) for l in labels],zero_division=0))
        except Exception as e:
            print(f" 生成分类报告失败：{str(e)}")

        # 计算详细指标
        metrics_dict = {
            'accuracy': accuracy,
            'precision': round(self.safe_metric(precision_score, y, y_pred, average='weighted',zero_division=0), 4),
            'recall': round(self.safe_metric(recall_score, y, y_pred, average='weighted',zero_division=0), 4),
            'f1': round(self.safe_metric(f1_score, y, y_pred, average='weighted',zero_division=0), 4),
            'confusion': confusion_matrix(y, y_pred).tolist()
        }

        # 绘制混淆矩阵
        plt = self.plot_confusion_matrix(y, y_pred, labels, f'{dataset_name}混淆矩阵')
        plt.savefig(f'{dataset_name}_confusion_matrix.png')
        plt.close()

        return metrics_dict


if __name__ == "__main__":
    # 测试数据加载器
    loader = DataLoader()
    try:
        data = loader.load_data("d:\\wh\\AAAproject\\Algorithm\\Data\\个贷违约\\public.csv")
        print(f"数据加载成功，共 {len(data)} 行，{len(data.columns)} 列")

        report = loader.validate_data(data, outliers_method="iqr", outliers_threshold=1.5)
        print(f"数据质量报告：总重复行数 {report['duplicate_rows']}")

        # cleaned_data = loader.basic_cleaning(data)
        # print(f"基础清洗后，共 {len(cleaned_data)} 行")
    except Exception as e:
        print(f"错误：{e}")
