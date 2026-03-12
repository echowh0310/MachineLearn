import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from Dataloader import DataLoader
import os
import joblib
from sklearn.metrics import confusion_matrix


# 配置中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 初始化数据加载器
loader = DataLoader()

# 定义保存目录和文件名
OUTPUT_DIR = "./model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_DIR = "./model_output/NB"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "NB_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "NB_label_encoder.pkl")
MAPPING_PATH = os.path.join(MODEL_DIR, "NB_label_mapping.pkl")

try:
    # 加载配置文件
    config_path = "./config/NB.yaml"
    config = loader.load_config(config_path)

    # 读取核心配置参数
    data_path = config["IRIS_DATA_PATH"]
    label_col = config["IRIS_LABEL_COL"]
    feature_cols = config["IRIS_FEATURE_COLS"]
    test_size = config["TEST_SIZE"]
    val_size = config["VAL_SIZE"]
    use_stratified = config["USE_STRATIFIED"]
    random_seed = config["RANDOM_SEED"]

    # 加载数据集
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

    # 模型初始化与训练
    #NB模型参数
    model_type = config["MODEL"]
    alpha = config["ALPHA"]
    fit_prior = config["FIT_PRIOR"]
    class_prior = config["CLASS_PRIOR"]
    binarize = config["BINARIZE"]

    # 初始化NB模型
    if model_type not in ["GaussianNB", "BernoulliNB", "MultinomialNB"]:
        raise ValueError("不支持其他算法，仅支持GaussianNB, BernoulliNB, MultinomialNB")

    if model_type=="GaussianNB":#连续特征
        model = GaussianNB()

    elif model_type=="MultinomialNB":#离散计数特征
        model = MultinomialNB(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior
        )

    elif model_type=="BernoulliNB":#二值离散特征
        model = BernoulliNB(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            binarize=binarize
        )

    # 模型训练
    model.fit(x_train, y_train)
    print("\n模型训练完成")


    NB_socre = model.score(x_test, y_test)
    print(f"模型在测试集上的得分：{NB_socre}")


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

    # 绘制混淆矩阵热力图
    if le is not None:
        class_labels = le.classes_
    else:
        class_labels = all_labels  # 数值类别直接用
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print("\n========== 混淆矩阵 ==========")
    print(cm)

    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels= class_labels, yticklabels=class_labels
    )
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('朴素贝叶斯模型 - 混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plt.show()

    # 模型保存
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

    # 可视化（决策边界+ROC曲线）
    # 仅当特征数=2且类别数≤3时可视化
    # if len(valid_feature_indices) == 2 and class_num <= 3:
    #     h = .02  # 网格步长
    #     # 获取特征范围
    #     X_selected = data.iloc[:, valid_feature_indices].values
    #     x_min, x_max = X_selected[:, 0].min() - 1, X_selected[:, 0].max() + 1
    #     y_min, y_max = X_selected[:, 1].min() - 1, X_selected[:, 1].max() + 1
    #     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #
    #     # 预测网格点类别
    #     Z = Model.predict(np.c_[xx.ravel(), yy.ravel()])
    #     Z = Z.reshape(xx.shape)
    #
    #     # 可视化
    #     plt.figure(figsize=(12, 10))
    #
    #     # 子图1：决策边界 + 测试集样本分布
    #     plt.subplot(2, 1, 1)
    #     plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    #
    #     # 绘制不同类别样本
    #     colors = ['blue', 'red', 'green']
    #     if class_num == 3:
    #         labels = ['Class 0 (setosa)', 'Class 1 (versicolor)', 'Class 2 (virginica)']
    #     else:
    #         labels = ['Class 0', 'Class 1']
    #     for cls in range(class_num):
    #         mask = y_test == cls
    #         if len(x_test) > 0:
    #             plt.scatter(x_test[mask, 0], x_test[mask, 1],
    #                         color=colors[cls], label=labels[cls], edgecolors='k')
    #
    #     plt.title('SVM决策边界')
    #     plt.xlabel(valid_feature_cols[0])
    #     plt.ylabel(valid_feature_cols[1])
    #     plt.legend()
    #     plt.grid(alpha=0.3)
    #
    #     # 子图2：二分类ROC曲线 / 多分类PR曲线
    #     plt.subplot(2, 1, 2)
    #     if len(x_test) == 0:
    #         plt.text(0.5, 0.5, '测试集为空，无法绘制曲线',
    #                  ha='center', va='center', fontsize=12)
    #     else:
    #         # 二分类场景 - ROC曲线
    #         if class_num == 2:
    #             y_pred_proba = Model.predict_proba(x_test)
    #             auc = metrics.roc_auc_score(y_test, y_pred_proba[:, 1])
    #             fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba[:, 1])
    #             plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc:.4f})')
    #             plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    #             plt.xlabel('假阳性率（FPR）')
    #             plt.ylabel('真阳性率（TPR）')
    #             plt.title('ROC曲线（测试集）')
    #             plt.legend(loc="lower right")
    #         # 多分类场景 - PR曲线（宏平均）
    #         elif class_num > 2:
    #             # 计算多分类PR曲线（One-vs-Rest）
    #             y_pred_proba = Model.predict_proba(x_test)
    #             # 对每个类别计算Precision-Recall
    #             precision = dict()
    #             recall = dict()
    #             average_precision = dict()
    #             for i in range(class_num):
    #                 precision[i], recall[i], _ = metrics.precision_recall_curve(
    #                     (y_test == i).astype(int), y_pred_proba[:, i]
    #                 )
    #                 average_precision[i] = metrics.average_precision_score(
    #                     (y_test == i).astype(int), y_pred_proba[:, i]
    #                 )
    #
    #             # 绘制每个类别的PR曲线
    #             colors = ['blue', 'red', 'green'][:class_num]
    #             labels = [f'Class {i}' for i in range(class_num)]
    #             if class_num == 3:
    #                 labels = ['Class 0 (setosa)', 'Class 1 (versicolor)', 'Class 2 (virginica)']
    #
    #             for i, color, label in zip(range(class_num), colors, labels):
    #                 plt.plot(recall[i], precision[i], color=color, lw=2,
    #                          label=f'{label} (AP = {average_precision[i]:.4f})')
    #
    #             # 绘制宏平均PR曲线
    #             precision_macro, recall_macro, _ = metrics.precision_recall_curve(
    #                 label_binarize(y_test, classes=all_labels).ravel(),
    #                 y_pred_proba.ravel()
    #             )
    #             average_precision_macro = metrics.average_precision_score(
    #                 label_binarize(y_test, classes=all_labels), y_pred_proba, average='macro'
    #             )
    #             plt.plot(recall_macro, precision_macro, color='purple', lw=2, linestyle='--',
    #                      label=f'宏平均 (AP = {average_precision_macro:.4f})')
    #
    #             plt.xlabel('召回率（Recall）')
    #             plt.ylabel('精确率（Precision）')
    #             plt.title('PR曲线（多分类，测试集）')
    #             plt.legend(loc="lower left")
    #         plt.grid(alpha=0.3)
    #
    #     plt.tight_layout()
    #     plt.show()

    print("\n程序执行完成，正常退出")
    sys.exit(0)

except Exception as e:
    print(f"\n 程序执行异常：{str(e)}", file=sys.stderr)
    import traceback

    traceback.print_exc()
    plt.close('all')
    sys.exit(1)