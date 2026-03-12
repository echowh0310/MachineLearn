import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl

from Dataloader import DataLoader
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ====================== 完善中文显示配置 ======================
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = 'sans-serif'

# 初始化数据加载器
loader = DataLoader()

# 定义保存目录和文件名
OUTPUT_DIR = "./model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 模型文件名
MODEL_PATH = os.path.join(OUTPUT_DIR, "random_forest_model.pkl")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
MAPPING_PATH = os.path.join(OUTPUT_DIR, "label_mapping.pkl")
PREPROCESSOR_PATH = os.path.join(OUTPUT_DIR, "preprocessor.pkl")

try:
    # 加载配置文件
    config_path = "RandomForest.yaml"
    config = loader.load_config(config_path)

    # 读取核心配置参数
    data_path = config.get("IRIS_DATA_PATH", "./iris.csv")
    label_col = config.get("IRIS_LABEL_COL", "species")
    feature_cols = config.get("IRIS_FEATURE_COLS", [])
    test_size = config.get("TEST_SIZE", 0.2)
    val_size = config.get("VAL_SIZE", 0.1)
    use_stratified = config.get("USE_STRATIFIED", True)
    random_seed = config.get("RANDOM_SEED", 42)
    encode_cols = config.get("ENCODE_COLS", {})
    class_weight = config.get("CLASS_WEIGHT", None)
    is_standard = config.get("IS_STANDARD", True)
    remainder = config.get("REMAINDER", "passthrough")

    # 加载数据集
    data = loader.load_data(data_path)
    if data.empty:
        raise ValueError("加载的数据集为空，请检查数据路径和文件格式")

    print(f"数据集列名：{data.columns.tolist()}")
    print(f"目标标签列：{label_col}")
    print(f"特征列：{feature_cols}")


    # 列存在性校验
    if label_col not in data.columns:
        raise ValueError(f"标签列 '{label_col}' 不存在于数据集中")
    if not feature_cols:
        raise ValueError("特征列配置为空，请检查配置文件中的IRIS_FEATURE_COLS")

    # 提取标签并进行编码
    y = data[label_col].values
    le = None
    label_mapping = None

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"标签映射关系：{label_mapping}")
        data[label_col] = y
    else:
        print("标签列已为数值类型，无需编码")

    # 统计类别数量
    class_num = len(np.unique(y))
    all_labels = sorted(data[label_col].unique())
    print(f"数据集类别数：{class_num}，类别列表：{all_labels}")

    # 数据集划分
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

    # 校验划分后的数据非空
    for name, subset in [("训练集", train_data), ("验证集", val_data), ("测试集", test_data)]:
        if subset.empty:
            raise ValueError(f"{name}为空，请调整划分比例或检查数据集")

    # ====================== 特征编码逻辑 ======================

    # 1. 筛选需要编码的列
    valid_encode_cols = {}
    onehot_encode_cols = config.get("ONEHOT_ENCODE_COLS", None)
    ordinal_encode_cols = config.get("ORDINAL_ENCODE_COLS", None)

    # 处理独热编码列
    if onehot_encode_cols:
        for col in onehot_encode_cols:
            if col in data.columns and col in train_data.columns:
                valid_encode_cols[col] = "onehot"
            else:
                print(f"编码列 '{col}' 不存在于数据集中，已跳过")
    else:
        print(f"独热编码列为空，已跳过")

    # 处理序列编码列
    if ordinal_encode_cols:
        for col in ordinal_encode_cols:
            if col in data.columns and col in train_data.columns:
                valid_encode_cols[col] = "ordinal"
            else:
                print(f"编码列 '{col}' 不存在于数据集中，已跳过")
    else:
        print(f"序列编码列为空，已跳过")

    # 2. 分离数值特征和需要编码的特征
    numeric_cols = [col for col in feature_cols if col in data.columns and col not in valid_encode_cols]
    onehot_cols = [col for col, typ in valid_encode_cols.items() if typ == "onehot"]
    ordinal_cols = [col for col, typ in valid_encode_cols.items() if typ == "ordinal"]

    print(f"数值特征列：{numeric_cols}")
    print(f"独热编码列：{onehot_cols}")
    print(f"序列编码列：{ordinal_cols}")

    # 校验有效特征列非空
    all_feature_cols = numeric_cols + onehot_cols + ordinal_cols
    if not all_feature_cols:
        raise ValueError("无有效特征列（数值列+编码列），无法训练模型！")

    # 3. 构建预处理流水线
    transformers = []

    # 独热编码
    if onehot_cols:
        onehot_encoder = OneHotEncoder(
            sparse_output=False,
            drop='first',
            handle_unknown='ignore',
            categories='auto'
        )
        transformers.append(('onehot', onehot_encoder, onehot_cols))

    # 序列编码
    if ordinal_cols:
        ordinal_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            categories='auto'
        )
        transformers.append(('ordinal', ordinal_encoder, ordinal_cols))

    # 数值特征：标准化
    if numeric_cols and is_standard:
        transformers.append(('numeric', StandardScaler(), numeric_cols))

    # 4. 初始化预处理器
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder=remainder,
        verbose_feature_names_out=True  # 保留特征名前缀，便于后续匹配
    )

    # ====================== 特征提取与编码 ======================
    X_train = train_data[all_feature_cols]
    X_val = val_data[all_feature_cols]
    X_test = test_data[all_feature_cols]

    # 拟合并转换训练集
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # 提取标签
    y_train = train_data[label_col].values
    y_val = val_data[label_col].values
    y_test = test_data[label_col].values

    print(f"编码后训练集特征维度：{X_train_processed.shape}")
    print(f"编码后验证集特征维度：{X_val_processed.shape}")
    print(f"编码后测试集特征维度：{X_test_processed.shape}")

    # ====================== 模型训练 ======================
    # RandomForest模型参数
    n_estimators = config.get("N_ESTIMATORS", 100)
    max_depth = config.get("MAX_DEPTH", None)

    # 初始化RandomForest模型
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_seed,  # 增加随机种子，保证可复现
        n_jobs=-1  # 并行训练，提升速度
    )

    # 模型训练
    model.fit(X_train_processed, y_train)
    print("\n模型训练完成")

    # ====================== 模型评估 ======================
    # 在训练集上评估
    train_metrics = loader.evaluate_model(model, X_train_processed, y_train, "训练集", all_labels)

    # 在验证集上评估
    val_metrics = loader.evaluate_model(model, X_val_processed, y_val, "验证集", all_labels)

    # 在测试集上评估
    test_metrics = loader.evaluate_model(model, X_test_processed, y_test, "测试集", all_labels)

    # 输出最终结果汇总
    print("\n=== 模型性能汇总 ===")
    print(f"训练集准确率: {train_metrics.get('accuracy', 0):.4f}")
    print(f"验证集准确率: {val_metrics.get('accuracy', 0):.4f}")
    print(f"测试集准确率: {test_metrics.get('accuracy', 0):.4f}")

    # ====================== 模型保存 ======================
    try:
        # 保存模型
        joblib.dump(model, MODEL_PATH)
        print(f"\n模型已保存到：{MODEL_PATH}")

        # 保存预处理器（编码器）
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        print(f"预处理编码器已保存到：{PREPROCESSOR_PATH}")

        # 保存LabelEncoder实例（标签编码）
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

    # ====================== 特征重要性 ======================
    # 获取特征重要性
    feature_importances = model.feature_importances_

    # 从preprocessor中获取处理后的特征名（关键修复：使用get_feature_names_out）
    try:
        encoded_feature_names = preprocessor.get_feature_names_out(all_feature_cols)
    except Exception as e:
        # 兼容旧版本sklearn
        print(f"获取特征名失败：{e}，使用备用命名方案")
        encoded_feature_names = []
        # 1. 独热编码特征名
        if onehot_cols:
            onehot_encoder = preprocessor.named_transformers_['onehot']
            for i, col in enumerate(onehot_cols):
                categories = onehot_encoder.categories_[i]
                for cat in categories[1:]:  # drop='first' 跳过第一个
                    encoded_feature_names.append(f"{col}_{cat}")
        # 2. 序列编码特征名
        encoded_feature_names.extend(ordinal_cols)
        # 3. 数值特征名
        encoded_feature_names.extend(numeric_cols)

    # 最终校验特征名数量
    if len(encoded_feature_names) != len(feature_importances):
        print(f"警告：特征名数量({len(encoded_feature_names)})与重要性数量({len(feature_importances)})不匹配，自动补齐")
        # 补齐或截断
        if len(encoded_feature_names) < len(feature_importances):
            encoded_feature_names += [f"unknown_feature_{i}" for i in
                                      range(len(feature_importances) - len(encoded_feature_names))]
        else:
            encoded_feature_names = encoded_feature_names[:len(feature_importances)]

    # 创建 DataFrame 便于查看
    importance_df = pd.DataFrame({
        '特征': encoded_feature_names,
        '重要性': feature_importances
    }).sort_values('重要性', ascending=False)

    print("\n特征重要性排序：")
    print(importance_df)

    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    # 限制显示前20个特征，避免图表拥挤
    top_n = min(20, len(importance_df))
    top_importance = importance_df.head(top_n)
    plt.barh(top_importance['特征'], top_importance['重要性'])
    plt.xlabel('特征重要性')
    plt.title('随机森林 - 特征重要性（含编码特征）')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    #增加图片保存，避免只显示不保存
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.show()

    print("\n程序执行完成，正常退出")
    sys.exit(0)

except Exception as e:
    print(f"\n程序执行异常：{str(e)}", file=sys.stderr)
    import traceback

    traceback.print_exc()
    plt.close('all')
    sys.exit(1)