import numpy as np
import matplotlib.pyplot as plt
import sys
from Dataloader import DataLoader as CustomDataLoader
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')
import matplotlib as mpl
# # 中文显示配置
try:
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    mpl.rcParams['axes.unicode_minus'] = False
except:
    pass

# 设置设备：优先使用GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")


# 定义神经网络模型，可配置的全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[64, 32], dropout=0.2):
        # input_dim: 输入特征维度（如4个特征则为4）
        # num_classes: 分类类别数（如鸢尾花3类则为3）
        # hidden_dims: 隐藏层维度列表（默认[64,32]）
        # dropout: 随机失活概率（防止过拟合）
        super(SimpleNN, self).__init__()
        layers = []
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # 隐藏层
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], num_classes))

        # 组装模型
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 初始化数据加载器
loader = CustomDataLoader()

# 定义保存目录和文件名
OUTPUT_DIR = "./model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_DIR = "./model_output/nn"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "pytorch_nn_model.pth")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
MAPPING_PATH = os.path.join(MODEL_DIR, "label_mapping.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")

try:
    # 加载配置文件
    config_path = "./config/NN.yaml"
    config = loader.load_config(config_path)

    # 读取核心配置参数
    data_path = config["IRIS_DATA_PATH"]
    label_col = config["IRIS_LABEL_COL"]
    feature_cols = config["IRIS_FEATURE_COLS"]
    test_size = config["TEST_SIZE"]
    val_size = config["VAL_SIZE"]
    use_stratified = config["USE_STRATIFIED"]
    random_seed = config["RANDOM_SEED"]

    # 设置随机种子
    torch.manual_seed(random_seed) #设置PyTorch随机种子
    np.random.seed(random_seed) #设置NumPy随机种子

    # 加载数据集
    data = loader.load_data(data_path)
    print(f"数据集列名：{data.columns.tolist()}")
    print(f"目标标签列：{label_col}")
    print(f"特征列：{feature_cols}")

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
    print(f"类别数量：{class_num}")

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

    # 筛选有效特征列
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
        raise ValueError("无有效特征列，无法训练模型！请检查配置文件中的IRIS_FEATURE_COLS")
    print(f"有效特征列：{valid_feature_cols}")
    print(f"特征列索引：{feature_indices}")

    # 提取特征列索引
    valid_feature_indices = [idx for idx in feature_indices if idx != -1]

    # 特征/标签提取
    x_train = train_data.iloc[:, valid_feature_indices].values if not train_data.empty else np.array([])
    x_val = val_data.iloc[:, valid_feature_indices].values if not val_data.empty else np.array([])
    x_test = test_data.iloc[:, valid_feature_indices].values if not test_data.empty else np.array([])

    y_train = train_data[label_col].values if not train_data.empty else np.array([])
    y_val = val_data[label_col].values if not val_data.empty else np.array([])
    y_test = test_data[label_col].values if not test_data.empty else np.array([])

    # 特征标准化
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val) if len(x_val) > 0 else x_val
    x_test = scaler.transform(x_test) if len(x_test) > 0 else x_test

    # 创建PyTorch DataLoader
    BATCH_SIZE = 16
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    input_dim = len(valid_feature_indices)
    model = SimpleNN(input_dim=input_dim, num_classes=class_num).to(DEVICE)
    print(f"\n模型结构：")
    print(model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # 学习率调度器

    # 训练参数
    EPOCHS = config["EPOCHS"]
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # 模型训练
    print("\n开始训练模型...")
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        # 计算指标
        train_loss_epoch = train_loss / len(train_loader.dataset)
        val_loss_epoch = val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        train_acc_epoch = train_correct / train_total if train_total > 0 else 0
        val_acc_epoch = val_correct / val_total if val_total > 0 else 0

        # 保存最佳模型
        if val_acc_epoch > best_val_acc and val_total > 0:
            best_val_acc = val_acc_epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'input_dim': input_dim,
                'num_classes': class_num
            }, MODEL_PATH)

        # 记录
        train_losses.append(train_loss_epoch)
        val_losses.append(val_loss_epoch)
        train_accs.append(train_acc_epoch)
        val_accs.append(val_acc_epoch)

        # 学习率调度
        scheduler.step()

        # 打印进度
        if (epoch + 1) % 5 == 0:
            print(
                f'Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.4f}, '
                f'Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}')

    print("\n模型训练完成！")
    print(f"最佳验证集准确率: {best_val_acc:.4f}")

    # 加载最佳模型
    torch.serialization.add_safe_globals([StandardScaler])
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"已加载最佳模型（Epoch: {checkpoint['epoch'] + 1}）")


    # 模型评估函数
    def evaluate_model(model, dataloader, dataset_name, all_labels):
        model.eval()
        y_true = []
        y_pred = []

        if len(dataloader.dataset) == 0:
            print(f"\n{dataset_name}：无数据，跳过评估")
            return {}

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # 计算指标
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n{dataset_name}评估结果：")
        print(f"准确率: {accuracy:.4f}")
        print("\n分类报告：")
        try:
            print(classification_report(y_true, y_pred, target_names=[str(l) for l in all_labels]))
        except Exception as e:
            print(f"生成分类报告失败：{str(e)}")

        # 绘制混淆矩阵
        try:
            plt = loader.plot_confusion_matrix(y_true, y_pred, all_labels, f'{dataset_name}混淆矩阵')
            save_path = os.path.join(OUTPUT_DIR, f'{dataset_name}_confusion_matrix.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()  # 显式关闭

        except Exception as e:
            print(f"绘制{dataset_name}混淆矩阵失败：{str(e)}")

        return {
            'accuracy': accuracy,
            'y_true': y_true,
            'y_pred': y_pred
        }

    # 评估模型
    train_metrics = evaluate_model(model, train_loader, "训练集", all_labels)
    val_metrics = evaluate_model(model, val_loader, "验证集", all_labels)
    test_metrics = evaluate_model(model, test_loader, "测试集", all_labels)

    # 输出最终结果汇总
    print("\n=== 模型性能汇总 ===")
    print(f"训练集准确率: {train_metrics.get('accuracy', 0):.4f}")
    print(f"验证集准确率: {val_metrics.get('accuracy', 0):.4f}")
    print(f"测试集准确率: {test_metrics.get('accuracy', 0):.4f}")

    # 保存编码器和映射关系
    try:
        # 保存标签编码器
        if le is not None:
            joblib.dump(le, ENCODER_PATH)
            print(f"标签编码器已保存到：{ENCODER_PATH}")

        # 保存标签映射字典
        if label_mapping is not None:
            joblib.dump(label_mapping, MAPPING_PATH)
            print(f"标签映射关系已保存到：{MAPPING_PATH}")

        # 保存特征标准化器
        joblib.dump(scaler, SCALER_PATH)
        print(f"特征标准化器已保存到：{SCALER_PATH}")

    except Exception as e:
        print(f"\n保存编码器/标准化器失败：{str(e)}")

    # 可视化训练过程
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练/验证损失曲线')
    plt.legend()
    plt.grid(alpha=0.3)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('训练/验证准确率曲线')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_curve.png'))

    print("\n程序执行完成，正常退出")
    sys.exit(0)

except Exception as e:
    print(f"\n 程序执行异常：{str(e)}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    plt.close('all')
    sys.exit(1)