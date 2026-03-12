import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib后端和中文显示
plt.switch_backend('TkAgg')
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def valid_data_col(data, columns):
    """
    校验数据列的有效性
    :param data: 待校验的DataFrame
    :param columns: 指定校验的列名列表（None则校验所有列）
    :return: 原数据、有效列名列表
    """
    if data is None:
        raise ValueError("data数据集不能为空")

    # 处理columns为None的情况（默认校验所有列）
    target_columns = data.columns.tolist() if columns is None else columns.copy()

    # 筛选有效/无效列
    valid_columns = [col for col in target_columns if col in data.columns]
    invalid_columns = [col for col in target_columns if col not in data.columns]

    # 输出提示信息
    if invalid_columns:
        print(f"以下列不存在数据集中，已跳过：{invalid_columns}")
    if not valid_columns:
        print("警告：无有效列可处理")

    return data, valid_columns


def detect_outlier_single_col(data, col, method="iqr", threshold=1.5, plot_flag=True):
    """
    检测单个数值列的异常值（抽离循环后的单列版本）
    :param data: 待处理的DataFrame
    :param col: 单个待检测列名
    :param method: 异常值检测方法，可选"iqr"/"std"/"z_score"
    :param threshold: 检测阈值（IQR=1.5/3；std/z_score=3）
    :param plot_flag: 是否绘制异常值散点图（True/False）
    :return: 该列的异常值信息字典、原始数据副本
    """
    data_processed = data.copy()
    outliers_info_col = {}

    # 校验列类型（仅处理数值型）
    if data_processed[col].dtype not in ["int64", "float64"]:
        print(f"警告：列{col}非数值型，跳过异常值检测")
        return outliers_info_col, data_processed

    # 异常值检测逻辑
    lower_bound = None
    upper_bound = None
    is_outliers = np.zeros(len(data_processed), dtype=bool)

    if method == "iqr":
        q1 = data_processed[col].quantile(0.25)
        q3 = data_processed[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        is_outliers = (data_processed[col] < lower_bound) | (data_processed[col] > upper_bound)

    elif method == "std":
        mean_val = data_processed[col].mean()
        std_val = data_processed[col].std()
        lower_bound = mean_val - threshold * std_val
        upper_bound = mean_val + threshold * std_val
        is_outliers = (data_processed[col] < lower_bound) | (data_processed[col] > upper_bound)

    elif method == "z_score":
        mean_val = data_processed[col].mean()
        std_val = data_processed[col].std()
        data_z = (data_processed[col] - mean_val) / std_val
        is_outliers = abs(data_z) > threshold
        lower_bound = mean_val - threshold * std_val
        upper_bound = mean_val + threshold * std_val

    else:
        raise ValueError("仅支持iqr/std/z_score三种检测方法")

    # 记录该列异常值信息
    outlier_idx = data_processed[is_outliers].index
    outliers_info_col[col] = {
        "count": len(outlier_idx),
        "ratio": len(outlier_idx) / len(data_processed),
        "range": (lower_bound, upper_bound),
        "index": outlier_idx,
        "is_outliers": is_outliers.copy(),
        "method": method,
        "threshold": threshold
    }

    # 可视化该列处理前的异常值散点图
    if plot_flag:
        plt.figure(figsize=(10, 6))
        # 正常值
        plt.scatter(
            data_processed.index[~is_outliers],
            data_processed[col][~is_outliers],
            color="blue", alpha=0.6,
            label=f"正常值（{len(data_processed) - len(outlier_idx)}个）"
        )
        # 异常值
        plt.scatter(
            data_processed.index[is_outliers],
            data_processed[col][is_outliers],
            color="red", alpha=0.8,
            label=f"异常值（{len(outlier_idx)}个）"
        )
        # 上下界线
        plt.axhline(y=lower_bound, color="grey", linestyle="--", linewidth=1.5,
                    label=f"下界 ({lower_bound:.4f})")
        plt.axhline(y=upper_bound, color="black", linestyle="--", linewidth=1.5,
                    label=f"上界 ({upper_bound:.4f})")
        plt.title(f"{col} - 异常值检测结果（处理前）", fontsize=12)
        plt.xlabel("数据行索引")
        plt.ylabel(col)
        plt.legend(loc="best")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    return outliers_info_col, data_processed


def process_outlier_single_col(data_processed, col, outliers_info_col, plot_flag=True,
                               process_outliers_method="save_outlier", fill_value=None):
    """
    处理单个数值列的异常值（抽离循环后的单列版本）
    :param data_processed: 检测后的DataFrame
    :param col: 单个待处理列名
    :param outliers_info_col: 该列的异常值信息字典
    :param plot_flag: 是否绘制可视化图表（True/False）
    :param process_outliers_method: 异常值处理方法，可选"del_outlier"/"save_outlier"/"fill_outlier"
    :param fill_value: 填充异常值时的自定义值（None则用上下界填充）
    :return: 更新后的该列异常值信息字典、处理后的数据集
    """
    # 校验列有效性
    if col not in data_processed.columns or data_processed[col].dtype not in ["int64", "float64"]:
        print(f"警告：列{col}无效或非数值型，跳过异常值处理")
        return outliers_info_col, data_processed

    # 若该列无异常值信息，重新检测
    if col not in outliers_info_col:
        method = "iqr"
        threshold = 1.5
        # 重新计算异常值
        lower_bound = None
        upper_bound = None
        if method == "iqr":
            q1 = data_processed[col].quantile(0.25)
            q3 = data_processed[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            is_outliers = (data_processed[col] < lower_bound) | (data_processed[col] > upper_bound)
        elif method == "std" or method == "z_score":
            mean_val = data_processed[col].mean()
            std_val = data_processed[col].std()
            lower_bound = mean_val - threshold * std_val
            upper_bound = mean_val + threshold * std_val
            is_outliers = (data_processed[col] < lower_bound) | (data_processed[col] > upper_bound)
        else:
            raise ValueError("仅支持iqr/std/z_score三种检测方法")

        outlier_idx = data_processed[is_outliers].index
        outliers_info_col[col] = {
            "count": len(outlier_idx),
            "ratio": len(outlier_idx) / len(data_processed),
            "range": (lower_bound, upper_bound),
            "index": outlier_idx,
            "is_outliers": is_outliers.copy(),
            "method": method,
            "threshold": threshold
        }

    # 获取该列异常值信息
    outlier_idx = outliers_info_col[col]["index"]
    lower_bound, upper_bound = outliers_info_col[col]["range"]
    is_outliers_current = data_processed.index.isin(outlier_idx)
    method = outliers_info_col[col]["method"]
    threshold = outliers_info_col[col]["threshold"]

    # 异常值处理逻辑
    if process_outliers_method == "del_outlier":
        # 删除当前数据集中的异常值行
        data_processed = data_processed[~is_outliers_current].reset_index(drop=True)
        # 更新异常值信息
        outliers_info_col[col]["count"] = 0
        outliers_info_col[col]["ratio"] = 0
        outliers_info_col[col]["index"] = pd.Index([])
        outliers_info_col[col]["is_outliers"] = np.zeros(len(data_processed), dtype=bool)
        print(f"列{col}：已删除{len(outlier_idx)}个异常值行，剩余行数：{len(data_processed)}")

    elif process_outliers_method == "save_outlier":
        # 保留异常值，无操作
        print(f"列{col}：保留{len(outlier_idx)}个异常值，不做处理")

    elif process_outliers_method == "fill_outlier":
        # 填充异常值
        if fill_value is None:
            # 用上下界填充
            fill_lower = data_processed[col] < lower_bound
            fill_upper = data_processed[col] > upper_bound
            data_processed.loc[fill_lower, col] = lower_bound
            data_processed.loc[fill_upper, col] = upper_bound
            print(
                f"列{col}：用上下界填充{len(outlier_idx)}个异常值（下界={lower_bound:.4f}，上界={upper_bound:.4f}）")
        else:
            # 自定义值填充（类型校验）
            if not isinstance(fill_value, (int, float)):
                raise TypeError(f"列{col}是数值类型，填充值{fill_value}必须为数字")
            data_processed.loc[is_outliers_current, col] = fill_value
            print(f"列{col}：用自定义值{fill_value}填充{len(outlier_idx)}个异常值")

    else:
        raise ValueError("仅支持del_outlier/save_outlier/fill_outlier三种处理方法")

    # 可视化该列处理后的结果
    if plot_flag:
        plt.figure(figsize=(10, 6))
        if process_outliers_method == "del_outlier":
            # 删除后无异常值
            plt.scatter(
                data_processed.index,
                data_processed[col],
                color="blue", alpha=0.6,
                label=f"正常值（{len(data_processed)}个）"
            )
            plt.text(0.05, 0.95, "已删除所有异常值", transform=plt.gca().transAxes,
                     bbox=dict(boxstyle="round", facecolor="lightgreen"))
        else:
            # 保留/填充后重新标记异常值
            if method == "iqr":
                q1 = data_processed[col].quantile(0.25)
                q3 = data_processed[col].quantile(0.75)
                iqr = q3 - q1
                new_lower = q1 - threshold * iqr
                new_upper = q3 + threshold * iqr
            else:
                mean_val = data_processed[col].mean()
                std_val = data_processed[col].std()
                new_lower = mean_val - threshold * std_val
                new_upper = mean_val + threshold * std_val

            new_is_outliers = (data_processed[col] < new_lower) | (data_processed[col] > new_upper)
            plt.scatter(
                data_processed.index[~new_is_outliers],
                data_processed[col][~new_is_outliers],
                color="blue", alpha=0.6,
                label=f"正常值/已填充值"
            )
            plt.scatter(
                data_processed.index[new_is_outliers],
                data_processed[col][new_is_outliers],
                color="red", alpha=0.8,
                label=f"未处理异常值（{new_is_outliers.sum()}个）"
            )

            plt.axhline(y=new_lower, color="grey", linestyle="--", linewidth=1.5, label=f"新下界 ({new_lower:.4f})")
            plt.axhline(y=new_upper, color="black", linestyle="--", linewidth=1.5,
                        label=f"新上界 ({new_upper:.4f})")

        # 绘制原始上下界（参考）
        plt.axhline(y=lower_bound, color="orange", linestyle=":", linewidth=1,
                    label=f"原始下界 ({lower_bound:.4f})")
        plt.axhline(y=upper_bound, color="purple", linestyle=":", linewidth=1,
                    label=f"原始上界 ({upper_bound:.4f})")

        plt.title(f"{col} - 异常值{process_outliers_method}结果（处理后）", fontsize=12)
        plt.xlabel("数据行索引")
        plt.ylabel(col)
        plt.legend(loc="best")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    return outliers_info_col, data_processed


# ---------------------- 主函数（负责循环调度） ----------------------
if __name__ == "__main__":
    # 1. 加载数据集（替换为实际路径）
    data = pd.read_csv("D:/wh/AAAproject/Algorithm/Data/个贷违约/train_internet.csv")
    # data = pd.read_csv("D:/wh/AAAproject/Algorithm/Data/irisdata/iris.csv")
    # 2. 定义需要检测的数值列
    detect_outliers_cols = ['monthly_payment', 'house_exist', 'recircle_b', 'recircle_u']
    # detect_outliers_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # 3. 校验列有效性
    data, valid_cols = valid_data_col(data, detect_outliers_cols)
    if not valid_cols:
        exit()  # 无有效列则退出

    # 4. 主循环：逐列检测异常值
    outliers_info = {}
    data_processed = data.copy()
    print("===== 开始逐列检测异常值 =====")
    for col in valid_cols:
        col_outlier_info, data_processed = detect_outlier_single_col(
            data=data_processed,
            col=col,
            method="iqr",
            threshold=3,
            plot_flag=True
        )
        outliers_info.update(col_outlier_info)  # 合并单列结果到总字典
        print("\n===== 异常值统计汇总 =====")
        for col, info in outliers_info.items():
            print(f"{col}：异常值数量={info['count']}，占比={info['ratio']:.2%}，正常范围={info['range']}")

        # 提取该列的异常值信息
        col_info = {col: outliers_info[col]} if col in outliers_info else {}
        col_outlier_info, data_processed = process_outlier_single_col(
            data_processed=data_processed,
            col=col,
            outliers_info_col=col_info,
            plot_flag=True,
            process_outliers_method="del_outlier",  # 处理方法可选：del_outlier/save_outlier/fill_outlier
            fill_value=None
        )
        outliers_info.update(col_outlier_info)  # 更新总字典
        # 6. 绘制处理后的箱线图
        plt.figure(figsize=(12, 6))
        pd.DataFrame(data_processed[col]).boxplot()
        plt.title(f"异常值处理后箱线图（{outliers_info[col]['method'].upper()}法）", fontsize=12)
        plt.xticks(rotation=45)
        plt.ylabel("字段值")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

