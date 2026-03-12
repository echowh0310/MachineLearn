import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sys
from Dataloader import DataLoader
import matplotlib as mpl
from sklearn import metrics

# 配置中文显示
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

loader = DataLoader()

try:
    # 加载配置文件
    config_path = "./config/OneLR.yaml"
    config = loader.load_config(config_path)

    # 从配置文件读取参数
    data_path = config["IRIS_DATA_PATH"]
    label_column = config["IRIS_LABEL_COL"]
    test_size = config["TEST_SIZE"]
    val_size = config["VAL_SIZE"]
    use_stratified = config["use_stratified"]
    random_seed = config["RANDOM_SEED"]

    #  加载数据
    data = loader.load_data(data_path)
    print("数据列名：", data.columns.tolist())

    # 划分数据集
    split_data = loader.split_dataset(
        data=data,
        label_column=label_column,
        test_size=test_size,
        val_size=val_size,
        use_stratified=use_stratified,
        random_seed=random_seed
    )
    # 训练集、验证集和测试集数据
    train_data = split_data["train"]
    test_data = split_data["test"]
    val_data = split_data["val"]

    # 特征数据选择x
    feature_col = config["IRIS_FEATURE_COL"][0] if isinstance(config["IRIS_FEATURE_COL"], (list, tuple)) else config[
        "IRIS_FEATURE_COL"]

    x_train = train_data[feature_col].values.reshape(-1, 1)
    x_test = test_data[feature_col].values.reshape(-1, 1)
    x_val = val_data[feature_col].values.reshape(-1, 1)

    # 标签数据选择（标签为一维数组）
    y_train = train_data[label_column]
    y_test = test_data[label_column]
    y_val = val_data[label_column]

    # 初始化线性回归模型
    model = LinearRegression()

    # 训练模型
    model.fit(x_train, y_train)
    print("\n模型训练完成")
    print(f"模型系数：{model.coef_[0]:.4f}")
    print(f"模型截距：{model.intercept_:.4f}")

    # 验证集评估
    val_score = model.score(x_val, y_val)
    print(f"验证集R²得分：{val_score:.4f}")

    # 测试集评估
    test_score = model.score(x_test, y_test)
    print(f"测试集R²得分：{test_score:.4f}")

    #模型评估
    y_pred = model.predict(x_test)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    MAE = metrics.mean_absolute_error(y_test, y_pred)
    R2 = metrics.r2_score(y_test, y_pred)

    print("\n模型评估指标：")
    print(f'MSE (均方误差)：{MSE:.4f}')
    print(f'RMSE (均方根误差)：{RMSE:.4f}')
    print(f'MAE (平均绝对误差)：{MAE:.4f}')
    print(f'R² (决定系数)：{R2:.4f}')

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color='blue', label='Train Data', alpha=0.6)
    x_range = np.linspace(x_train.min(), x_train.max(), 100).reshape(-1, 1)
    y_pred_line = model.predict(x_range)
    plt.plot(x_range, y_pred_line, color='red', label=f'Fitted Line (coef={model.coef_[0]:.4f})')
    plt.xlabel(feature_col)
    plt.ylabel(label_column)
    plt.title('Linear Regression Fit on Iris Data')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n程序执行完成，正常退出")
    sys.exit(0)  # 0表示正常退出

except Exception as e:
    # 捕获所有异常并打印，避免进程直接崩溃
    print(f"程序执行异常：{str(e)}", file=sys.stderr)
    # 清理绘图资源
    plt.close('all')
    sys.exit(1)  # 异常退出