import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from Dataloader import DataLoader
import pandas as pd
#################################网格搜索  交叉验证############################
# ===================== 基础配置 =====================
# 配置中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 初始化数据加载器
loader = DataLoader()

try:
    # 加载配置文件
    config_path = "SVM.yaml"
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
    print(f" 配置的特征列：{feature_cols}")

    # ===================== 特征列有效性校验 =====================
    # 筛选有效特征列
    valid_feature_cols = []
    feature_indices = []
    for col in feature_cols:
        if col in data.columns:
            valid_feature_cols.append(col)
            feature_indices.append(data.columns.get_loc(col))
        else:
            print(f"特征列 '{col}' 不存在于数据集中，已跳过")

    # 校验有效特征数量
    if len(valid_feature_cols) == 0:
        raise ValueError(" 无有效特征列，无法训练模型！请检查配置文件中的IRIS_FEATURE_COLS")
    print(f" 有效特征列：{valid_feature_cols}")
    print(f" 特征列索引：{feature_indices}")

    # ===================== 数据集划分 =====================
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

    # 打印通用化的标签分布（兼容任意分类数）
    print("\n 数据集划分结果：")
    all_labels = sorted(data[label_col].unique())
    for ds_name, ds in [("训练集", train_data), ("验证集", val_data), ("测试集", test_data)]:
        if ds.empty:
            print(f"{ds_name}：空数据集")
            continue
        # 通用化统计标签分布（支持数字/字符串标签）
        label_dist = ds[label_col].value_counts().to_dict()
        total_samples = len(ds)
        dist_str = ", ".join([f"{k}:{v}个({v / total_samples:.1%})" for k, v in label_dist.items()])
        print(f"{ds_name}：{ds.shape} | 标签分布：{dist_str}")

    # ===================== 特征/标签提取 =====================
    # 提取特征（适配任意数量特征列）
    x_train = train_data.iloc[:, feature_indices].values
    x_val = val_data.iloc[:, feature_indices].values if not val_data.empty else np.array([])
    x_test = test_data.iloc[:, feature_indices].values

    # 提取标签
    y_train = train_data[label_col].values
    y_val = val_data[label_col].values if not val_data.empty else np.array([])
    y_test = test_data[label_col].values

    # ===================== 参数网格搜索配置 =====================
    # 定义参数网格（可根据需求调整）
    param_grid = {
        "C": [0.1, 1, 10, 100, 200],  # 增加更多C值选项
        "kernel": ["linear", "rbf", "poly","sigmoid"],  # 增加多项式核
        "gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1],  # 增加更多gamma选项
        "degree": [2, 3, 4],  # 多项式核的阶数
        "max_iter": [-1, 100, 200, 500],  # -1表示无限制
        "random_state": [random_seed]
    }

    # 配置交叉验证
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed) if use_stratified else 5

    # 初始化SVM模型
    model = SVC()

    # 配置网格搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_strategy,  # 5折交叉验证
        scoring='accuracy',  # 以准确率为评分标准
        refit=True,  # 使用最佳参数重新训练整个数据集
        verbose=2,  # 详细输出
        n_jobs=-1,  # 使用所有CPU核心加速
        error_score='raise'  # 遇到错误时抛出异常
    )

    # ===================== 执行参数搜索 =====================
    print("\n开始网格搜索最佳参数...")
    print(
        f"参数组合总数: {len(grid_search.cv_results_['params']) if hasattr(grid_search, 'cv_results_') else '计算中'}")

    # 执行网格搜索
    grid_search.fit(x_train, y_train)

    # ===================== 输出搜索结果 =====================
    print("\n=== 网格搜索结果 ===")
    print(f"最佳参数组合: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
    print(f"最佳模型索引: {grid_search.best_index_}")

    # 输出所有参数组合的结果
    print("\n所有参数组合的性能排名（前10）：")
    results = pd.DataFrame(grid_search.cv_results_)
    # 按测试分数排序
    results_sorted = results.sort_values(by='mean_test_score', ascending=False).head(10)

    for idx, row in results_sorted.iterrows():
        print(f"\n排名 {idx + 1}:")
        print(f" 参数: {row['params']}")
        print(f" 平均准确率: {row['mean_test_score']:.4f} (+/-{row['std_test_score']:.4f})")
        print(f" 训练时间: {row['mean_fit_time']:.2f}s")

    # 保存所有搜索结果到CSV文件
    results.to_csv('grid_search_results.csv', index=False)
    print("\n所有搜索结果已保存到 grid_search_results.csv")

    # ===================== 使用最佳模型评估 =====================
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    print(f"\n=== 最佳模型评估 ===")

    # 在训练集上评估
    train_metrics = loader.evaluate_model( best_model, x_train, y_train, "训练集", all_labels)

    # 在验证集上评估
    val_metrics = loader.evaluate_model(best_model, x_val, y_val, "验证集", all_labels)

    # 在测试集上评估
    test_metrics = loader.evaluate_model(best_model, x_test, y_test, "测试集", all_labels)

    # ===================== 输出最终结果汇总 =====================
    print("\n=== 模型性能汇总 ===")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"交叉验证最佳准确率: {grid_search.best_score_:.4f}")
    print(f"训练集准确率: {train_metrics.get('accuracy', 0):.4f}")
    print(f"验证集准确率: {val_metrics.get('accuracy', 0):.4f}")
    print(f"测试集准确率: {test_metrics.get('accuracy', 0):.4f}")

    # 保存最佳模型（可选）
    try:
        import joblib

        joblib.dump(best_model, './svm_best_model.pkl')
        print("\n最佳模型已保存到 svm_best_model.pkl")
    except ImportError:
        print("\n警告：无法保存模型，请安装joblib (pip install joblib)")
    except Exception as e:
        print(f"\n保存模型失败：{str(e)}")

    sys.exit(0)

except Exception as e:
    print(f"\n 程序执行异常：{str(e)}", file=sys.stderr)
    import traceback

    traceback.print_exc()
    plt.close('all')
    sys.exit(1)