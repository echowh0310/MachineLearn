import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
from Dataloader import DataLoader
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier

# 配置中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 初始化数据加载器
loader = DataLoader()

# 定义保存目录和文件名
OUTPUT_DIR = "./model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_DIR = "./model_output/KNN"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "KNN_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "KNN_label_encoder.pkl")
MAPPING_PATH = os.path.join(MODEL_DIR, "KNN_label_mapping.pkl")

try:
    #加载配置文件
    config_path = "./config/KNN.yaml"
    config = loader.load_config(config_path)

    #读取核心配置参数
    data_path = config["IRIS_DATA_PATH"]
    label_col = config["IRIS_LABEL_COL"]
    feature_cols = config["IRIS_FEATURE_COLS"]
    test_size = config["TEST_SIZE"]
    val_size = config["VAL_SIZE"]
    use_stratified = config["USE_STRATIFIED"]
    random_seed = config["RANDOM_SEED"]

    #加载数据集
    data = loader.load_data(data_path)
    print(f" 数据集列名：{data.columns.tolist()}")
    print(f" 目标标签列：{label_col}")
    print(f" 特征列：{feature_cols}")



    # 提取标签并进行编码
    y = data[label_col].values
    le = None  # 初始化编码器
    label_mapping = None  # 初始化映射关系

    if y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y = le.fit_transform(y)
        # 记录标签映射关系
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"标签映射关系：{label_mapping}")
        # 编码后标签写回数据集
        data[label_col] = y
    else:
        print("标签列已为数值类型，无需编码")

    # 统计类别数量
    class_num = len(np.unique(y))
    all_labels = sorted(data[label_col].unique())

    #数据集划分
    # 划分训练/验证/测试集
    split_data = loader.split_dataset(
        data=data,
        label_column=label_col,
        test_size=test_size,
        val_size=val_size,
        use_stratified=use_stratified,
        random_seed=random_seed
    )

    train_data = split_data["train"]
    val_data = split_data["val"]
    test_data = split_data["test"]

    # 提取特征列索引
    valid_feature_cols = []
    feature_indices = []
    for col in feature_cols:
        if col in data.columns:
            valid_feature_cols.append(col)
            feature_indices.append(data.columns.get_loc(col))
        else:
            print(f"特征列 '{col}' 不存在于数据集中，已跳过")
            feature_indices.append(-1)

    # 校验有效特征数量
    if len(valid_feature_cols) == 0:
        raise ValueError(" 无有效特征列，无法训练模型！请检查配置文件中的IRIS_FEATURE_COLS")
    print(f" 有效特征列：{valid_feature_cols}")
    print(f" 特征列索引：{feature_indices}")

    # 特征/标签提取
    valid_feature_indices = [idx for idx in feature_indices if idx != -1]
    # 根据选择特征提取数据集
    x_train = train_data.iloc[:, valid_feature_indices].values if not train_data.empty else np.array([])
    x_val = val_data.iloc[:, valid_feature_indices].values if not val_data.empty else np.array([])
    x_test = test_data.iloc[:, valid_feature_indices].values if not test_data.empty else np.array([])

    # 提取标签
    y_train = train_data[label_col].values if not train_data.empty else np.array([])
    y_val = val_data[label_col].values if not val_data.empty else np.array([])
    y_test = test_data[label_col].values if not test_data.empty else np.array([])

    #模型初始化与训练
    # KNN模型参数
    n_neighbors = config["N_NEIGHBORS"]
    weights = config["WEIGHTS"]
    algorithm = config["ALGORITHM"]
    leaf_size = config["LEAF_SIZE"]
    p = config["P"]
    metric = config["METRIC"]
    metric_params = config["METRIC_PARAMS"]

    # 初始化SVM模型
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        metric=metric,
        metric_params=metric_params
    )

    # 模型训练
    model.fit(x_train, y_train)
    print("\n模型训练完成")


    # 模型评估
    # 在训练集上评估
    train_metrics = loader.evaluate_model(model, x_train, y_train, "训练集", all_labels)

    # 在验证集上评估
    val_metrics = loader.evaluate_model(model, x_val, y_val, "验证集", all_labels)

    # 在测试集上评估
    test_metrics = loader.evaluate_model(model, x_test, y_test, "测试集", all_labels)

    # 输出最终结果汇总
    print("\n=== 模型性能汇总 ===")
    print(f"训练集准确率: {train_metrics.get('accuracy', 0):.4f}")
    print(f"验证集准确率: {val_metrics.get('accuracy', 0):.4f}")
    print(f"测试集准确率: {test_metrics.get('accuracy', 0):.4f}")

    #模型保存
    try:
        # 保存模型
        joblib.dump(model, MODEL_PATH)
        print(f"\n模型已保存到：{MODEL_PATH}")

        # 保存LabelEncoder实例
        if le is not None:
            joblib.dump(le, ENCODER_PATH)
            print(f"标签编码器已保存到：{ENCODER_PATH}")

        # 保存标签映射字典
        if label_mapping is not None:
            joblib.dump(label_mapping, MAPPING_PATH)
            print(f"标签映射关系已保存到：{MAPPING_PATH}")

    except ImportError:
        print("\n警告：无法保存模型/编码器，请安装joblib (pip install joblib)")
    except Exception as e:
        print(f"\n保存模型/编码器失败：{str(e)}")



    print("\n程序执行完成，正常退出")
    sys.exit(0)

except Exception as e:
    print(f"\n 程序执行异常：{str(e)}", file=sys.stderr)
    import traceback

    traceback.print_exc()
    plt.close('all')
    sys.exit(1)