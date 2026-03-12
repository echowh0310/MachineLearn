import sys
import matplotlib as mpl
import os

from sklearn.preprocessing import LabelEncoder
from Dataloader import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局配置 =====================
# 配置中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# lightGBM算法


# ===================== 通用函数封装 =====================
def load_config_and_data(model_type):
    """加载配置文件和数据集（通用逻辑）"""
    # 初始化数据加载器
    loader = DataLoader()

    # 解析命令行参数
    config_path = model_type

    # 校验配置文件存在性
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在！")

    # 加载配置
    config = loader.load_config(config_path)

    # 读取核心配置参数
    data_path = config["IRIS_DATA_PATH"]
    label_col = config["IRIS_LABEL_COL"]
    feature_cols = config["IRIS_FEATURE_COLS"]

    # 加载数据集
    data = loader.load_data(data_path)
    print(f"数据集加载成功，行数：{len(data)}，列数：{len(data.columns)}")
    print(f"数据集列名：{data.columns.tolist()}")
    print(f"目标标签列：{label_col}")
    print(f"配置的特征列：{feature_cols}")

    print(f"数据基本信息：\n{data.info(memory_usage='deep')}")
    print(f"缺失值情况：\n{data.isnull().sum()[data.isnull().sum() > 0]}")

    # 分离数值类
    num = ['float64', 'int64']
    num_cols = data.select_dtypes(include=num).columns
    print(f"数值类字段：{num_cols}")
    # 数值类分为连续值 类别 时间

    # 分离字符类
    cat = ["object", "category"]
    cat_cols = data.select_dtypes(include=cat).columns
    print(f"字符类字段{cat_cols}")
    # 字符类分为文本 类别 时间

    return loader, config, data, label_col, feature_cols


# 数据集和列表校验
def valid_data_col(data, columns):
    if data is None:
        raise ValueError("data数据集不能为空")
    if columns is None:
        target_columns = data.columns.tolist()
    else:
        target_columns = columns.copy()

    # 过滤掉不存在的列
    valid_columns = [col for col in target_columns if col in data.columns]
    invalid_columns = [col for col in target_columns if col not in data.columns]

    if invalid_columns:
        print(f"以下列不存在数据集中，已跳过：{invalid_columns}")
    if not valid_columns:
        print("无有效列可处理，直接返回原数据")

    return data, valid_columns


def process_single_col_null(process_style, fill_context, col, data):
    """
    处理单个列的缺失值（核心单列处理逻辑）
    参数说明：
    - process_style: 处理类型标识（同原col_null）
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
        mode_value = mode_series[0] if not mode_series.empty else "未知"
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
        data[col] = data[col].fillna(fill_context)
        print(f"列{col}：指定值填充，填充值={fill_context}")

    else:
        raise ValueError(
            f"列{col}：不支持的处理类型：{process_style}，"
            f"可选类型：missing_all_cols/missing_50%_cols/mean_cols/mode_cols/median_cols/fill_cols"
        )

    return data


def col_null(process_style, fill_context, columns, data):
    """
    多列缺失值处理入口（循环调用单列处理函数）
    参数说明：同原col_null函数
    返回：处理后的DataFrame
    """
    # 校验数据集和目标列
    data, valid_columns = valid_data_col(data, columns)

    if not valid_columns:
        return data

    # 循环处理每个有效列
    print(f"\n开始批量处理 {len(valid_columns)} 列的缺失值：{valid_columns}")
    for col in valid_columns:
        print(f"\n----- 处理列：{col} -----")
        data = process_single_col_null(process_style, fill_context, col, data)

    # ========== 批量处理后验证 ==========
    remaining_missing = data[valid_columns].isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    if len(remaining_missing) > 0:
        print(f"\n批量处理后仍存在缺失值的字段：\n{remaining_missing}")
    else:
        print("\n所有目标列的缺失值已处理完毕")

    return data


# 去重重复行数
def dele_repeat(data, columns=None, keep="first"):
    # 校验必填参数
    data, valid_columns = valid_data_col(data, columns)  # 有效检验
    row_count_before = len(data)
    data_unique = data.drop_duplicates(subset=valid_columns, keep=keep)
    # subset: 这个参数允许在指定基于哪些列来检查重复项，默认为所有列。
    # keep: 控制如何处理重复项，默认值为'first'，表示保留每个重复组的第一个出现。
    # False删除所有重复行

    row_count_after = len(data_unique)
    del_count = row_count_before - row_count_after
    print(f"去重前行数{row_count_before}")
    print(f"去重后行数{row_count_after}")
    print(f"删除重复行数{del_count}")
    print(f"重复率{del_count / row_count_before:.4f}")

    return data_unique


def detect_outlier_single_col(data, col, method="iqr", threshold=1.5 ):
                              # plot_flag=False):
    """
    检测单个数值列的异常值
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

    # # 可视化该列处理前的异常值散点图
    # if plot_flag:
    #     plt.figure(figsize=(10, 6))
    #     # 正常值
    #     plt.scatter(
    #         data_processed.index[~is_outliers],
    #         data_processed[col][~is_outliers],
    #         color="blue", alpha=0.6,
    #         label=f"正常值（{len(data_processed) - len(outlier_idx)}个）"
    #     )
    #     # 异常值
    #     plt.scatter(
    #         data_processed.index[is_outliers],
    #         data_processed[col][is_outliers],
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

    return outliers_info_col, data_processed


def process_outlier_single_col(data_processed, col, outliers_info_col,
                               process_outliers_method="save_outlier", fill_value=None):
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
        outliers_info_col["outliers_count"] = 0
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


def validate_feature_cols(data, feature_cols):
    """特征列有效性校验"""
    valid_feature_cols = []
    feature_indices = []
    for col in feature_cols:
        if col in data.columns:
            valid_feature_cols.append(col)
            feature_indices.append(data.columns.get_loc(col))
        else:
            print(f"特征列 '{col}' 不存在于数据集中，已跳过")
            feature_indices.append(-1)

    # 过滤无效索引并校验有效特征数量
    valid_feature_indices = [idx for idx in feature_indices if idx != -1]
    if len(valid_feature_cols) == 0:
        raise ValueError("无有效特征列，无法训练模型！请检查配置文件中的IRIS_FEATURE_COLS")

    print(f"有效特征列：{valid_feature_cols}")
    print(f"有效特征索引：{valid_feature_indices}")
    return valid_feature_cols, valid_feature_indices


def process_label_encoding(data, label_col):
    """标签编码处理（兼容字符串/数值标签）"""
    y = data[label_col].values
    le = None
    label_mapping = None

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"标签映射关系：{label_mapping}")
    else:
        print("标签列已为数值类型，无需编码")

    # 统计类别数量（不修改原始数据集）
    class_num = len(np.unique(y))
    all_labels = sorted(np.unique(y))
    return y, le, label_mapping, class_num, all_labels


def split_dataset(loader, data, label_col, y, test_size, val_size, use_stratified, random_seed):
    """数据集划分 + 特征/标签提取"""
    # 临时将编码后的标签写入数据（仅用于划分，不污染原始数据）
    data_temp = data.copy()
    data_temp[label_col] = y

    # 划分数据集
    split_data = loader.split_dataset(
        data=data_temp,
        label_column=label_col,
        test_size=test_size,
        val_size=val_size,
        use_stratified=use_stratified,
        random_seed=random_seed
    )

    train_data = split_data["train"]
    val_data = split_data["val"]
    test_data = split_data["test"]

    # 校验训练集非空
    if train_data.empty:
        raise ValueError("训练集为空，无法训练模型！请调整test_size/val_size参数")

    return train_data, val_data, test_data


def extract_features_labels(train_data, val_data, test_data, label_col, valid_feature_indices):
    """提取特征和标签（处理空数据集）"""
    # 提取特征
    x_train = train_data.iloc[:, valid_feature_indices].values if not train_data.empty else np.array([])
    x_val = val_data.iloc[:, valid_feature_indices].values if not val_data.empty else np.array([])
    x_test = test_data.iloc[:, valid_feature_indices].values if not test_data.empty else np.array([])

    # 提取标签
    y_train = train_data[label_col].values if not train_data.empty else np.array([])
    y_val = val_data[label_col].values if not val_data.empty else np.array([])
    y_test = test_data[label_col].values if not test_data.empty else np.array([])

    return x_train, x_val, x_test, y_train, y_val, y_test


# ===================== 主执行流程 =====================
def main(model_type):
    """主函数（统一执行逻辑）"""
    try:
        # 1. 加载配置和数据
        loader, config, data, label_col, feature_cols = load_config_and_data(
            model_type)

        # 处理缺失值函数处理（现在内部会循环处理多列）
        process_style = config["ISNULL_PROCESS"]
        fill_context = config["FILL_PROCESS"]
        columns = config["COLUMNS"]


        data = col_null(process_style=process_style, columns=columns, data=data, fill_context=fill_context)

        # 处理异常值字段
        valid_cols = config["VALID_COLS"]
        threshold = config["THRESHOLD"]
        method = config["METHOD"]
        procee_outliers_method = config["PROCESE_OUTLIERS_METHOD"]
        fill_value = config["FILL_VALUE"]

        outliers_info = {}
        data_processed = data.copy()

        print("===== 开始逐列检测异常值 =====")
        for col in valid_cols:
            #
            plt.figure(figsize=(12, 6))
            pd.DataFrame(data_processed[col]).boxplot(whis=3)
            plt.title(f"异常值处理后箱线图（{col}）", fontsize=12)
            plt.xticks(rotation=45)
            plt.ylabel("字段值")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            #
            col_outlier_info, data_processed = detect_outlier_single_col(
                data=data_processed,
                col=col,
                method=method,
                threshold=threshold
                # plot_flag=False
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
                # plot_flag=False,
                process_outliers_method=procee_outliers_method,  # 处理方法可选：del_outlier/save_outlier/fill_outlier
                fill_value=fill_value
            )
            outliers_info.update(col_outlier_info)  # 更新总字典
            # # 6. 绘制处理后的箱线图
            # plt.figure(figsize=(12, 6))
            # pd.DataFrame(data_processed[col]).boxplot()
            # plt.title(f"异常值处理后箱线图（{outliers_info[col]['method'].upper()}法）", fontsize=12)
            # plt.xticks(rotation=45)
            # plt.ylabel("字段值")
            # plt.grid(axis="y", alpha=0.3)
            # plt.tight_layout()

        # 2. 特征列校验
        # valid_feature_cols, valid_feature_indices = validate_feature_cols(data, feature_cols)
        #
        # # 3. 标签编码
        # y, le, label_mapping, class_num, all_labels = process_label_encoding(data, label_col)
        #
        # # 4. 数据集划分
        # train_data, val_data, test_data = split_dataset(loader, data, label_col, y, test_size, val_size, use_stratified,
        #                                                 random_seed)
        #
        # # 5. 提取特征和标签
        # x_train, x_val, x_test, y_train, y_val, y_test = extract_features_labels(
        #     train_data, val_data, test_data, label_col, valid_feature_indices
        # )

        print("\n程序执行完成，正常退出")
        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\n文件错误：{str(e)}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n参数/数据错误：{str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n程序执行异常：{str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        plt.close('all')
        sys.exit(1)


# ===================== 执行入口 =====================
if __name__ == "__main__":
    # 执行SVM模型：python unified_model.py --config SVMLR.yaml
    # 执行逻辑回归模型：python unified_model.py --config logistic.yaml
    # 可通过修改model_type切换模型
    model_type = "./config/TZ.yaml"  # 可选："svm" / "logistic"
    main(model_type)