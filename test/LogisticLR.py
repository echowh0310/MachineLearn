import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from Dataloader import DataLoader
import sys
import matplotlib as mpl
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
os.environ['OMP_NUM_THREADS'] = '1'
# 配置中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 初始化数据加载
loader = DataLoader()

try:
    # 加载配置文件
    config_path = "LogLR.yaml"
    config = loader.load_config(config_path)

    # 从配置文件读取参数
    data_path = config["IRIS_DATA_PATH"]
    label_column = config["IRIS_LABEL_COL"]
    test_size = config["TEST_SIZE"]
    val_size = config["VAL_SIZE"]
    use_stratified = config["use_stratified"]
    random_seed = config["RANDOM_SEED"]

    # 加载并预处理鸢尾花数据集
    data = loader.load_data(data_path)
    print("数据列名：", data.columns.tolist())

    # 提取特征和标签
    X = data.iloc[:, :-1].values
    y = data[label_column].values  # 标签列

    # 标签列转为数字0/1/2
    if y.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"标签映射关系：{dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 二分类转换：0保留，1/2合并为1
    y_binary = np.where(y == 0, 0, 1)
    data[label_column] = y_binary

    # 划分训练集+验证集+测试集
    split_data = loader.split_dataset(
        data=data,
        label_column=label_column,
        test_size=test_size,
        val_size=val_size,
        use_stratified=use_stratified,
        random_seed=random_seed
    )

    # 提取训练/验证/测试集
    train_data = split_data["train"]
    test_data = split_data["test"]
    val_data = split_data["val"]
    print(f"\n数据集划分完成：")
    print(f"训练集形状：{train_data.shape}，标签分布：0类{sum(train_data[label_column]==0)}个，1类{sum(train_data[label_column]==1)}个")
    print(f"验证集形状：{val_data.shape}，标签分布：0类{sum(val_data[label_column]==0)}个，1类{sum(val_data[label_column]==1)}个")
    print(f"测试集形状：{test_data.shape}，标签分布：0类{sum(test_data[label_column]==0)}个，1类{sum(test_data[label_column]==1)}个")

 # 提取特征和索引位置
    # 从配置文件读取特征列名
    feature_cols = config["IRIS_FEATURE_COLS"]
    # 获取数据集列名列表
    data_columns = data.columns.tolist()
    # 遍历获取每个特征列的索引位置
    feature_indices = []
    for col_name in feature_cols:
        if col_name in data_columns:
            idx = data_columns.index(col_name)
            feature_indices.append(idx)
        else:
            print(f"警告：特征列 '{col_name}' 未在数据集中找到！")
            feature_indices.append(-1)

    # 输出最终结果
    print(f"\n特征列名列表：{feature_cols}")
    print(f"特征列对应的索引位置列表：{feature_indices}")

    #根据选择特征提取数据集
    x_train = train_data.iloc[:, feature_indices].values  # 仅用2个特征训练
    x_test = test_data.iloc[:, feature_indices].values
    x_val = val_data.iloc[:, feature_indices].values

    y_train = train_data[label_column].values
    y_test = test_data[label_column].values
    y_val = val_data[label_column].values

    #逻辑回归模型参数
    max_iter=config["MAX_ITER"]
    solver=config["SOLVER"]
    multi_class=config["MULTI_CLASS"]
    # 逻辑回归模型训练
    model = LogisticRegression(max_iter=max_iter,
                               solver=solver,
                               random_state=random_seed,
                               multi_class=multi_class
                               )
    model.fit(x_train, y_train)
    print("\n模型训练完成")
    print(f"模型截距：{model.intercept_[0]:.4f}")
    print(f"模型系数（2个特征）：{[f'{c:.4f}' for c in model.coef_[0]]} ")

    # 模型评估
    train_acc = model.score(x_train, y_train)
    val_acc = model.score(x_val, y_val) if not val_data.empty else 0.0
    test_acc = model.score(x_test, y_test)
    print(f"\n准确率评估：")
    print(f"训练集准确率：{train_acc:.4f}")
    print(f"验证集准确率：{val_acc:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")

    # 详细分类评估指标（测试集）
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]  # 预测为1类的概率

    # 安全计算指标
    def safe_metric(metric_func, y_true, y_pred, default=0.0):
        try:
            return metric_func(y_true, y_pred)
        except:
            return default

    # 分类指标
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = safe_metric(metrics.precision_score, y_test, y_pred)
    recall = safe_metric(metrics.recall_score, y_test, y_pred)
    f1 = safe_metric(metrics.f1_score, y_test, y_pred)
    auc = safe_metric(metrics.roc_auc_score, y_test, y_pred_proba)
    confusion = metrics.confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("\n模型评估指标（测试集）：")
    print(f'准确率（Accuracy）：{accuracy:.4f}')
    print(f'精确率（Precision）：{precision:.4f}')
    print(f'召回率（Recall）：{recall:.4f}')
    print(f'F1分数：{f1:.4f}')
    print(f'AUC值：{auc:.4f}')
    print(f'混淆矩阵：\n{confusion}')
    print("分类报告:")
    print(class_report)


    # 绘制决策边界（特征维度：2个特征）
    h = .02  # 网格步长
    # 获取所选2个特征的范围
    x_min, x_max = X[:, feature_indices[0]].min() - 1, X[:, feature_indices[0]].max() + 1
    y_min, y_max = X[:, feature_indices[1]].min() - 1, X[:, feature_indices[1]].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 预测网格点的类别（维度匹配：2个特征）
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 可视化
    plt.figure(figsize=(12, 10))

    # 子图1：决策边界 + 测试集样本分布
    plt.subplot(2, 1, 1)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    # 绘制测试集样本
    plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1],
                color='blue', label='Class 0 (setosa)', edgecolors='k')
    plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1],
                color='red', label='Class 1 (non-setosa)', edgecolors='k')
    plt.title('逻辑回归决策边界（特征：花萼长度 vs 花萼宽度）')
    plt.xlabel('花萼长度 (cm)')
    plt.ylabel('花萼宽度 (cm)')
    plt.legend()
    plt.grid(alpha=0.3)

    # 子图2：ROC曲线
    plt.subplot(2, 1, 2)
    if len(np.unique(y_test)) == 2:  # 确保有两个类别
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 随机基线
        plt.xlabel('假阳性率（FPR）')
        plt.ylabel('真阳性率（TPR）')
        plt.title('ROC曲线（测试集）')
        plt.legend(loc="lower right")
    else:
        plt.text(0.5, 0.5, '测试集仅包含单一类别，无法绘制ROC曲线',
                 ha='center', va='center', fontsize=12)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join("./", "kmeans_clusters.png"))
    plt.show()

    print("\n程序执行完成，正常退出")
    sys.exit(0)

except Exception as e:
    print(f"\n程序执行异常：{str(e)}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    plt.close('all')
    sys.exit(1)