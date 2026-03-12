import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sys
from Dataloader import DataLoader
import matplotlib as mpl
from sklearn import metrics
import os
# 配置中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
MODEL_DIR = "./model_output/MoreLR"
os.makedirs(MODEL_DIR, exist_ok=True)

# 初始化数据加载
loader = DataLoader()

try:
    # 加载配置文件
    config_path = "./config/MoreLR.yaml"
    config = loader.load_config(config_path)

    # 从配置文件读取参数
    data_path = config["BOSTON_DATA_PATH"]
    test_size = config["TEST_SIZE"]
    val_size = config["VAL_SIZE"]
    random_seed = config["RANDOM_SEED"]
    use_stratified = config["use_stratified"]
    label_column = config["BOSTON_LABEL_COL"]

    # 加载数据
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

    # 提取训练/验证/测试集
    train_data = split_data["train"]
    test_data = split_data["test"]
    val_data = split_data["val"]
    print(f"训练集形状：{train_data.shape}，"
          f"验证集：{val_data.shape}，"
          f"测试集：{test_data.shape}")

    #  特征和标签提取
    x_train = train_data.drop(columns=[label_column]).values
    x_test = test_data.drop(columns=[label_column]).values
    x_val = val_data.drop(columns=[label_column]).values

    # 标签数据选择
    y_train = train_data[label_column].values
    y_test = test_data[label_column].values
    y_val = val_data[label_column].values

    # 模型训练
    # 初始化线性回归模型
    model = LinearRegression()
    # 训练模型
    model.fit(x_train, y_train)
    print("\n模型训练完成")
    print(f"模型截距：{model.intercept_:.4f}")
    print(f"模型系数总数：{len(model.coef_)} ")

    #模型评估
    train_score = model.score(x_train, y_train)
    val_score = model.score(x_val, y_val) if not val_data.empty else np.nan
    test_score = model.score(x_test, y_test)
    print(f"\n训练集R²得分：{train_score:.4f}")
    print(f"验证集R²得分：{val_score:.4f}")
    print(f"测试集R²得分：{test_score:.4f}")

    # 详细评估指标
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
    plt.figure(figsize=(12, 10))

    # 图1：真实值vs预测值折线图
    plt.subplot(2, 1, 1)
    sample_num = min(200, len(y_test))

    plt.plot(range(sample_num), y_test[:sample_num], 'r-', label='真实房价', linewidth=1.5)
    plt.plot(range(sample_num), y_pred[:sample_num], 'b-', label='预测房价', linewidth=1.5)
    plt.xlabel('样本序号')
    plt.ylabel('房价中位数（千美元）')
    plt.title('波士顿房价：真实值 vs 预测值（前{}样本）'.format(sample_num))
    plt.legend()
    plt.grid(alpha=0.3)

    # 图2：真实值vs预测值散点图
    plt.subplot(2, 1, 2)
    plt.scatter(y_test, y_pred, alpha=0.7, color='steelblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=2)
    plt.xlabel('真实房价（千美元）')
    plt.ylabel('预测房价（千美元）')
    plt.title(f'真实值 vs 预测值（R2={R2:.4f}）')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "真实值 vs 预测值.png"))
    plt.show()

    print("\n程序执行完成，正常退出")
    sys.exit(0)

except Exception as e:
    # 捕获所有异常并打印
    print(f"程序执行异常：{str(e)}", file=sys.stderr)
    plt.close('all')
    sys.exit(1)