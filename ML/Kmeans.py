# 先设置环境变量
import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from Dataloader import DataLoader
import joblib
import warnings

# 配置中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 忽略特定警告
warnings.filterwarnings('ignore', category=FutureWarning)
# 额外忽略KMeans的MKL内存泄漏警告
warnings.filterwarnings('ignore', category=UserWarning, message='KMeans is known to have a memory leak')


# 初始化数据加载器
loader = DataLoader()

# 定义保存目录和文件名
OUTPUT_DIR = "./model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_DIR = "./model_output/kmeans"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "kmeans_scaler.pkl")

try:
    # 加载配置文件
    config_path = "./config/Kmeans.yaml"
    config = loader.load_config(config_path)

    # 读取核心配置参数
    data_path = config["IRIS_DATA_PATH"]
    feature_cols = config["IRIS_FEATURE_COLS"]

    # 加载数据集
    data = loader.load_data(data_path)
    print(f" 数据集列名：{data.columns.tolist()}")
    print(f" 特征列：{feature_cols}")

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
    x = data.iloc[:, valid_feature_indices].values if not data.empty else np.array([])

    #数据标准化
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    #确定最佳K值（肘部法则）
    inertia = []
    k_range = range(1, 20)
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(x_scaled)
        inertia.append(km.inertia_)

    #可视化肘部曲线
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('K值')
    plt.ylabel('惯性值（Inertia）')
    plt.title('K-Means肘部法则确定最佳K值')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(MODEL_DIR, "elbow_curve.png"))  # 保存图片
    plt.show()

    # 自动计算最佳K值
    def find_optimal_k(inertia_values, k_range):
        """自适应找肘部点：计算二阶导数，取最大值对应的K"""
        # 一阶导数（惯性值变化量）
        first_deriv = np.diff(inertia_values)
        # 二阶导数（变化量的变化率）
        second_deriv = np.diff(first_deriv)
        # 二阶导数最大值对应的K（+2是因为两次diff后索引偏移）
        optimal_k_idx = np.argmax(second_deriv)
        optimal_k = k_range[optimal_k_idx + 2]
        return optimal_k

    optimal_k = find_optimal_k(inertia, k_range)
    final_k = config["FINAL_K"]
    # 如果自动计算失败，使用设置默认值
    optimal_k = optimal_k if optimal_k else final_k
    print(f"\n自动计算的最佳K值：{optimal_k}")

    #kmeans算法参数
    init = config["INIT"]
    n_init = config["N_INIT"]
    random_state = config["RANDOM_STATE"]

    # 使用最佳K值进行聚类
    kmeans = KMeans(
        n_clusters=optimal_k,
        # n_clusters=3,  #簇数量
        init=init,  #参数指定初始化质心的方法
        n_init=n_init,  #表示使用不同的初始质心运行算法的次数。默认10
        random_state=random_state
    )
    clusters = kmeans.fit_predict(x_scaled)

    # 评估聚类效果
    if optimal_k > 1:  # 轮廓系数要求至少2个簇
        silhouette_avg = silhouette_score(x_scaled, clusters)
        print(f'轮廓系数（Silhouette Score）: {silhouette_avg:.3f}')
    else:
        print("K=1时无法计算轮廓系数")
        silhouette_avg = None

    #获取标签和簇中心
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    print(f" 簇标签：{labels}")
    print(f" 簇中心：{centroids}")

    # 可视化聚类结果
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x[:, 0], x[:, 1],
        c=kmeans.labels_,
        cmap='viridis',
        alpha=0.7,
        label='样本点'
    )
    # 绘制簇中心
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(
        centroids_original[:, 0],
        centroids_original[:, 1],
        s=200, marker='X', c='red',
        label='簇中心'
    )
    plt.xlabel(valid_feature_cols[0])
    plt.ylabel(valid_feature_cols[1])
    plt.title(f'K-Means聚类结果（K={optimal_k}）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(MODEL_DIR, "kmeans_clusters.png"))
    plt.show()


    # 保存模型和标准化器
    joblib.dump(kmeans, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n模型已保存至：{MODEL_PATH}")
    print(f"标准化器已保存至：{SCALER_PATH}")

    print("\n程序执行完成，正常退出")
    sys.exit(0)

except Exception as e:
    print(f"\n 程序执行异常：{str(e)}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    plt.close('all')
    sys.exit(1)