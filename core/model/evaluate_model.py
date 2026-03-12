from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import numpy as np
import logging
from core.data_processing.data_loader import DataLoader
import pandas as pd
from typing import Dict, List, Any, Optional, Union

class ModelEvaluator:
    """模型评估类，封装所有模型评估相关功能"""

    def __init__(self):
        # 初始化数据加载器
        self.loader = DataLoader()
        # 定义日志实例
        self.logger = logging.getLogger("ml_platform_api")

    def _check_data(self, X: Optional[np.ndarray], y: Optional[np.ndarray]) -> bool:
        """校验数据是否有效"""
        if X is None or y is None:
            return False
        return X.size > 0 and y.size > 0

    def extract_feature_importance(self, model, feature_names, model_type):
        """树模型的特征重要性"""
        try:
            importances = model.feature_importances_

            # XGBoost/LightGBM 可选：使用 gain 重要性（需额外处理）
            if model_type in ["xgboost", "lightgbm"] and hasattr(model, "get_booster"):
                # 可扩展：model.get_score(importance_type='gain')
                pass

            # 创建DataFrame
            df = pd.DataFrame({
                '特征': feature_names if len(feature_names) == len(importances)
                else [f'特征{i + 1}' for i in range(len(importances))],
                '重要性': [round(v, 4) for v in importances]
            }).sort_values('重要性', ascending=False)

            self.logger.info(f"{model_type}特征重要性：\n{df.to_string(index=False)}")
            return df.to_dict('records')

        except Exception as e:
            self.logger.warning(f"{model_type}特征重要性提取失败: {str(e)}")
            return []

    def encode(self, obj):
        # 将混淆矩阵部分单独处理
        def format_confusion(matrix):
            # 将每一行转换为字符串，并添加缩进
            formatted_rows = []
            for row in matrix:
                # 将数字转换为固定宽度的字符串，使其对齐
                formatted_row = ", ".join(f"{num:>4}" for num in row)
                formatted_rows.append(f"[ {formatted_row} ]")
            return "[\n      " + ",\n      ".join(formatted_rows) + "\n    ]"

        if isinstance(obj, dict):
            # 递归处理字典
            # 过滤特征重要性字段
            filtered_obj = {k: v for k, v in obj.items() if k != 'feature_importance'}
            items = []
            for key, value in filtered_obj.items():
                if key == 'confusion':
                    # 对混淆矩阵应用特殊格式
                    formatted_value = format_confusion(value)
                    items.append(f'    "{key}": {formatted_value}')
                else:
                    # 对其他值递归编码
                    encoded_value = self.encode(value)
                    items.append(f'    "{key}": {encoded_value}')
            return "{\n  " + ",\n  ".join(items) + "\n}"
        elif isinstance(obj, list):
            # 处理列表
            encoded_items = [self.encode(item) for item in obj]
            return f"[ {', '.join(encoded_items)} ]"
        elif isinstance(obj, (int, float)):
            # 处理数值类型
            return str(obj)
        elif isinstance(obj, str):
            # 处理字符串
            return f'"{obj}"'
        elif obj is None:
            return "null"
        else:
            # 其他类型转为字符串
            return str(obj)

    def evaluatemodel(self, model, x_train, y_train, x_val, y_val, x_test, y_test, all_labels, feature_encoding,
                      problem_type,
                      model_type):
        """
        辅助函数：计算评估指标
        参数补充：
        - problem_type: 任务类型（classification/regression）
        - model_type: 模型类型（random_forest/LR/logistic_regression/svm/knn/decision_tree等）
        返回：
        - metrics: 包含训练/验证/测试集评估指标的字典
        """
        # 初始化空指标字典
        metrics = {
            "train": {},
            "val": {},
            "test": {}
        }

        try:
            # ========== 回归任务评估 ==========
            if problem_type == "regression":
                # 定义回归指标计算函数
                def calc_reg_metrics(y_true, y_pred, data_type=""):
                    """计算回归指标并记录日志"""
                    mae = round(mean_absolute_error(y_true, y_pred), 4)
                    mse = round(mean_squared_error(y_true, y_pred), 4)
                    rmse = round(np.sqrt(mse), 4)
                    r2 = round(r2_score(y_true, y_pred), 4)

                    # 日志输出
                    self.logger.info(f"\n{data_type}模型评估指标：")
                    self.logger.info(f'MAE (平均绝对误差)：{mae}')
                    self.logger.info(f'MSE (均方误差)：{mse}')
                    self.logger.info(f'RMSE (均方根误差)：{rmse}')
                    self.logger.info(f'R² (决定系数)：{r2}')

                    return {
                        "mae": mae,
                        "mse": mse,
                        "rmse": rmse,
                        "r2": r2
                    }

                # 定义回归型评估通用函数
                def evaluate_reg_model(X, y, data_type):
                    """通用回归模型评估函数"""
                    if not self._check_data(X, y):
                        return {}
                    y_pred = model.predict(X)
                    return calc_reg_metrics(y, y_pred, data_type)

                # 线性回归
                if model_type in ["LR", "lasso"]:
                    metrics["train"] = evaluate_reg_model(x_train, y_train, "训练集")
                    metrics["val"] = evaluate_reg_model(x_val, y_val, "验证集")
                    metrics["test"] = evaluate_reg_model(x_test, y_test, "测试集")

                # 随机森林、决策树、梯度提升决策树回归
                elif model_type in ["random_forest", "decision_tree", "gboost"]:
                    metrics["train"] = evaluate_reg_model(x_train, y_train, "训练集")
                    metrics["val"] = evaluate_reg_model(x_val, y_val, "验证集")
                    metrics["test"] = evaluate_reg_model(x_test, y_test, "测试集")

                    # 随机森林回归 - 特征重要性
                    metrics["feature_importance"] = self.extract_feature_importance(
                        model=model,
                        feature_names=feature_encoding,
                        model_type="随机森林回归"
                    )

                elif model_type in ["xgboost", "lightgbm"]:
                    metrics["train"] = evaluate_reg_model(x_train, y_train, "训练集")
                    metrics["val"] = evaluate_reg_model(x_val, y_val, "验证集")
                    metrics["test"] = evaluate_reg_model(x_test, y_test, "测试集")

                elif model_type == "knn":
                    metrics["train"] = evaluate_reg_model(x_train, y_train, "训练集")
                    metrics["val"] = evaluate_reg_model(x_val, y_val, "验证集")
                    metrics["test"] = evaluate_reg_model(x_test, y_test, "测试集")

                elif model_type == "svm":
                    metrics["train"] = evaluate_reg_model(x_train, y_train, "训练集")
                    metrics["val"] = evaluate_reg_model(x_val, y_val, "验证集")
                    metrics["test"] = evaluate_reg_model(x_test, y_test, "测试集")

                elif model_type == "adaboost":
                    metrics["train"] = evaluate_reg_model(x_train, y_train, "训练集")
                    metrics["val"] = evaluate_reg_model(x_val, y_val, "验证集")
                    metrics["test"] = evaluate_reg_model(x_test, y_test, "测试集")

                elif model_type == "mlp":
                    metrics["train"] = evaluate_reg_model(x_train, y_train, "训练集")
                    metrics["val"] = evaluate_reg_model(x_val, y_val, "验证集")
                    metrics["test"] = evaluate_reg_model(x_test, y_test, "测试集")

            # ========== 分类任务评估 ==========
            elif problem_type == "classification":
                # 定义分类模型评估通用函数
                def evaluate_classification_model(X, y, data_type):
                    """通用分类模型评估函数"""
                    if not self._check_data(X, y):
                        return {}
                    return self.loader.evaluate_model(model, X, y, data_type, all_labels)

                # 随机森林分类
                if model_type == "random_forest":
                    # 测试集基础得分
                    if self._check_data(x_test, y_test):
                        RFC_score = model.score(x_test, y_test)
                        self.logger.info(f"random_forest模型在测试集上的得分：{RFC_score}")

                    # 各数据集评估
                    metrics["train"] = evaluate_classification_model(x_train, y_train, "训练集")
                    metrics["val"] = evaluate_classification_model(x_val, y_val, "验证集")
                    metrics["test"] = evaluate_classification_model(x_test, y_test, "测试集")

                    # 特征重要性排序
                    try:
                        # 1.获取特征重要性
                        feature_importances = model.feature_importances_
                        features = feature_encoding

                        # 2. 双重校验：确保特征名和重要性长度一致
                        if len(features) != len(feature_importances):
                            self.logger.warning(
                                f"特征名长度({len(features)})与重要性长度({len(feature_importances)})不匹配，使用默认命名")
                            features = [f'特征{i + 1}' for i in range(len(feature_importances))]

                        # 3. 创建特征重要性DataFrame
                        importance_df = pd.DataFrame({
                            '特征': features,
                            '重要性': [round(imp, 4) for imp in feature_importances]
                        }).sort_values('重要性', ascending=False)

                        # 4. 日志输出
                        self.logger.info("特征重要性排序：\n" + importance_df.to_string(index=False))

                        # 5.将特征重要性加入评估指标字典
                        metrics["feature_importance"] = importance_df.to_dict('records')

                    except Exception as e:
                        self.logger.warning(f"计算特征重要性失败：{str(e)}")
                        metrics["feature_importance"] = []  # 异常时赋值空列表

                # 逻辑回归
                elif model_type == "logistic_regression":
                    metrics["train"] = evaluate_classification_model(x_train, y_train, "训练集")
                    metrics["val"] = evaluate_classification_model(x_val, y_val, "验证集")
                    metrics["test"] = evaluate_classification_model(x_test, y_test, "测试集")

                # 支持向量机
                elif model_type == "svm":
                    metrics["train"] = evaluate_classification_model(x_train, y_train, "训练集")
                    metrics["val"] = evaluate_classification_model(x_val, y_val, "验证集")
                    metrics["test"] = evaluate_classification_model(x_test, y_test, "测试集")

                # K近邻
                elif model_type == "knn":
                    metrics["train"] = evaluate_classification_model(x_train, y_train, "训练集")
                    metrics["val"] = evaluate_classification_model(x_val, y_val, "验证集")
                    metrics["test"] = evaluate_classification_model(x_test, y_test, "测试集")

                # 决策树
                elif model_type == "decision_tree":
                    metrics["train"] = evaluate_classification_model(x_train, y_train, "训练集")
                    metrics["val"] = evaluate_classification_model(x_val, y_val, "验证集")
                    metrics["test"] = evaluate_classification_model(x_test, y_test, "测试集")

                # 朴素贝叶斯（高斯/多元/伯努利）
                elif model_type in ["Ga_naive_bayes", "Be_naive_bayes"]:
                    metrics["train"] = evaluate_classification_model(x_train, y_train, "训练集")
                    metrics["val"] = evaluate_classification_model(x_val, y_val, "验证集")
                    metrics["test"] = evaluate_classification_model(x_test, y_test, "测试集")
                #
                elif model_type in ["xgboost", "lightgbm", "gboost"]:
                    metrics["train"] = evaluate_classification_model(x_train, y_train, "训练集")
                    metrics["val"] = evaluate_classification_model(x_val, y_val, "验证集")
                    metrics["test"] = evaluate_classification_model(x_test, y_test, "测试集")

                elif model_type == "adaboost":
                    metrics["train"] = evaluate_classification_model(x_train, y_train, "训练集")
                    metrics["val"] = evaluate_classification_model(x_val, y_val, "验证集")
                    metrics["test"] = evaluate_classification_model(x_test, y_test, "测试集")

                elif model_type == "mlp":

                    metrics["train"] = evaluate_classification_model(x_train, y_train, "训练集")
                    metrics["val"] = evaluate_classification_model(x_val, y_val, "验证集")
                    metrics["test"] = evaluate_classification_model(x_test, y_test, "测试集")

                # 未识别的分类模型
                else:
                    self.logger.warning(f"未识别的分类模型类型：{model_type}，跳过评估")

            # 未识别的任务类型
            else:
                self.logger.warning(f"未识别的任务类型：{problem_type}，仅返回空指标")

            return metrics


        except Exception as e:
            # 精准捕获当前出错的数据集类型
            error_details = []
            if not self._check_data(x_train, y_train):
                error_details.append("训练集数据为空")
            if not self._check_data(x_val, y_val):
                error_details.append("验证集数据为空")
            if not self._check_data(x_test, y_test):
                error_details.append("测试集数据为空")

            error_msg = "; ".join(error_details) if error_details else "未知原因"
            self.logger.error(f"模型评估失败 - {error_msg}，异常信息：{str(e)}")
            return metrics  # 返回已有部分的指标，而非空字典


# 调用示例
if __name__ == "__main__":
    # 1. 初始化评估器
    evaluator = ModelEvaluator()

    # 2. 调用评估函数（示例参数，需替换为实际训练好的模型和数据）
    # metrics = evaluator.evaluatemodel(
    #     model=trained_model,
    #     x_train=x_train,
    #     y_train=y_train,
    #     x_val=x_val,
    #     y_val=y_val,
    #     x_test=x_test,
    #     y_test=y_test,
    #     all_labels=all_labels,
    #     feature_encoding=feature_encoding,
    #     problem_type="classification",  # 或 "regression"
    #     model_type="random_forest"
    # )

    # 3. 提取特征重要性（示例）
    # feature_importance = evaluator.extract_feature_importance(
    #     model=trained_model,
    #     feature_names=feature_names,
    #     model_type="random_forest"
    # )
    pass