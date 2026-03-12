# 机器学习

## 项目概述

功能：数据上传、特征处理、模型训练、结果评估。支持多种机器学习算法，包括分类和回归任务，提供API接口。

## 项目结构

```
MachineLearning/
├── Data/                     # 示例数据集
│   ├── Breast/               # 乳腺组织数据集
│   ├── fangjia/              # 房价数据集
│   ├── irisdata/             # 鸢尾花数据集
│   └── 个贷违约/               # 个人贷款违约数据集
├── ML/                       # 机器学习算法脚本实现
│   ├── config/               # 算法配置文件
│   ├── model_output/         # ML算法脚本输出结果
│   ├── Dataloader.py         # 数据加载器
│   ├── Dataprocess.py        # 数据处理
│   ├── DecisionTree.py       # 决策树算法
│   ├── KNN.py                # K近邻算法
│   ├── Kmeans.py             # K-means聚类
│   ├── MoreLinear.py         # 多变量线性回归
│   ├── NB.py                 # 朴素贝叶斯
│   ├── OneLinear.py          # 单变量线性回归
│   ├── RandomForest.py       # 随机森林
│   ├── SVMLR.py              # 支持向量机回归
│   ├── logistic.py           # 逻辑回归
│   └── nn.py                 # 神经网络
├── api/                      # RESTful API
│   ├── logs/                 # 日志文件
│   ├── schemas/              # API数据模型及相关配置参数
│   ├── uploads/              # 上传文件存储
│   └── MLapi.py              # API主程序
├── core/                     # 核心功能模块
│   ├── data_processing/      # 数据处理模块
│   ├── model/                # 模型评估模块
│   └── utils/                # 工具函数
├── templates/                # 前端模板
└── test/                     # 其他文件
```

## API主程序核心功能

### 1. 数据处理
- 支持CSV/Excel/JSON格式数据上传
- 数据质量验证
- 异常值检测和处理（IQR、标准差、Z-score方法）
- 缺失值检测和处理（删除缺失值、均值/中位数/众数填充、指定值填充）
- 重复值删除
- 数据特征编码（独热编码、序数编码）
- 特征标准化、标签编码
- 数据集划分


### 2. 模型训练
- **分类算法**：
  - K近邻(KNN)
  - 支持向量机(SVM)
  - 随机森林
  - 决策树
  - 贝叶斯(GaussianNB、BernoulliNB)
  - 逻辑回归
  - GBDT
  - XGBoost
  - LightGBM
  - AdaBoost
  - 多层感知机(MLP)

- **回归算法**：
  - 线性回归
  - K近邻(KNN)
  - Lasso回归
  - 决策树回归
  - 随机森林回归
  - GBDT
  - XGBoost回归
  - LightGBM回归
  - AdaBoost
  - 支持向量机回归
  - 多层感知机(MLP)


### 3. 模型评估
- 分类评估指标：准确率、精确率、召回率、F1-score、混淆矩阵
- 回归评估指标：MAE、MSE、RMSE、R²
- 特征重要性分析
- 可视化输出（混淆矩阵、特征重要性图）


### 环境要求
- Python 3.7+
- pip 20.0+
- pip install requriements.txt

## 使用方法

### 1. 通过API使用

#### 数据集上传

```
POST /api/v1/data/upload
```

**参数**：
- `file`：待上传的数据集文件（支持csv/xlsx/json）
- `outliers_method`：异常值检测方法
- `outliers_threshold`：异常值检测阈值

**返回**：
- 数据集ID和验证报告

#### 获取数据集列名

```
POST /api/v1/data/{dataset_id}/columns
```

**参数**：
- `dataset_id`：数据集ID

**返回**：
- 数据集ID
- 数据集列名

#### 按列处理异常值、空值

```
POST /api/v1/data/{dataset_id}/columns/process
```

**参数**：
- `dataset_id`：数据集ID
- `columns_config`：列处理配置
- `global_null_process_method`：全局空值处理方法
- `global_outliers_method`：全局异常值检测方法
- `global_outliers_process_method`：全局异常值处理方法

**返回**：
- 处理结果和数据质量报告

#### 所有列统一处理异常值、空值

```
POST /api/v1/data/{dataset_id}/process/all
```

**参数**：
- `dataset_id`：数据集ID
- `cols`：待处理的列名列表
- `null_process_method`：空值处理方法
- `null_fill_value`：空值填充值
- `outliers_process_method`：异常值处理方法
- `outliers_fill_value`：异常值填充值
- `outliers_method`：异常值检测方法
- `outliers_threshold`：异常值检测阈值

**返回**：
- 处理结果和数据质量报告

#### 数据集划分

```
POST /api/v1/data/{dataset_id}/split
```

**参数**：
- `dataset_id`：数据集ID
- `split_params`：数据集划分参数

**返回**：
- 划分结果和各数据集信息


#### 模型训练

```
POST /api/v1/models/{problem_type}/{model_type}/train
```

**参数**：
- `dataset_id`：数据集ID
- `problem_type`：问题类型
- `model_type`：模型类型
- `train_request`：训练请求参数


**返回**：
- 训练结果（模型参数、评估指标、模型保存路径等）

### 2. 直接运行算法脚本

进入ML目录，运行相应的算法脚本(标准数据集)：

```bash
cd ML
python KNN.py
```

算法脚本会读取config目录下的配置文件，加载数据并执行训练和评估。

#### 配置文件

算法配置文件位于`ML/config/`目录下，使用YAML格式。示例配置（KNN.yaml）：

```yaml
IRIS_DATA_PATH: ../Data/irisdata/iris.csv
IRIS_LABEL_COL: species
IRIS_FEATURE_COLS: [sepal_length, sepal_width, petal_length, petal_width]
TEST_SIZE: 0.2
VAL_SIZE: 0.1
USE_STRATIFIED: true
RANDOM_SEED: 42
N_NEIGHBORS: 5
WEIGHTS: uniform
ALGORITHM: auto
LEAF_SIZE: 30
P: 2
METRIC: minkowski
METRIC_PARAMS: null
```

#### 数据格式

支持的数据格式：
- CSV：逗号分隔值文件
- Excel：.xlsx格式文件
- JSON：JSON格式文件


#### 支持的ML算法脚本

##### 分类算法

| 算法名称 | 算法类型 | 适用场景 |
|---------|---------|---------|
| KNN | 分类 | 简单分类任务，数据量适中 |
| SVM | 分类 | 高维数据，复杂决策边界 |
| 随机森林 | 分类 | 大规模数据，抗过拟合 |
| 决策树 | 分类 | 可解释性要求高的场景 |
| 朴素贝叶斯 | 分类 | 文本分类，特征独立假设成立 |
| 逻辑回归 | 分类 | 二分类或多分类任务 |
| MLP | 分类 | 复杂非线性关系 |

##### 回归算法

| 算法名称 | 算法类型 | 适用场景 |
|---------|---------|---------|
| 线性回归 | 回归 | 线性关系建模 |

##### 聚类算法

| 算法名称 | 算法类型 | 适用场景 |
|---------|---------|---------|
| K-means | 聚类 | 无监督学习，数据分组 |

## API文档

### 1. 前端页面路由

**接口地址**：`GET /`

**功能描述**：返回平台前端页面

**响应**：HTML页面

### 2. 数据集上传

**接口地址**：`POST /api/v1/data/upload`

**功能描述**：上传数据集并完成数据质量验证

**请求参数**：
- `file`：文件（form-data），支持csv/xlsx/json格式
- `outliers_method`：异常值检测方法（默认：iqr），可选值：iqr/std/z_score
- `outliers_threshold`：异常值检测阈值（默认：1.5），iqr=1.5/3；std/z_score=3

**响应示例**：
```json
{
  "code": 200,
  "msg": "上传成功",
  "data": {
    "dataset_id": "xxx",
    "original_file_name": "iris.csv",
    "stored_file_name": "xxx.csv",
    "file_size": 4551,
    "file_size_mb": 0.00,
    "file_path": "./uploads/xxx.csv",
    "upload_time": "2026-03-11 10:00:00",
    "validate_report": {
      "total_rows": 150,
      "total_columns": 5,
      "duplicate_rows": 0,
      "columns_info": {
        "sepal_length": {
          "dtype": "float64",
          "non_null_count": 150,
          "null_count": 0,
          "unique_values": 35,
          "sample_values": [5.1, 4.9, 4.7, 4.6, 5.0],
          "outliers_count": 0,
          "mean": 5.843333333333334,
          "median": 5.8,
          "min": 4.3,
          "max": 7.9
        }
      }
    }
  }
}
```

### 3. 获取数据集列名

**接口地址**：`GET /api/v1/data/{dataset_id}/columns`

**功能描述**：根据数据集ID获取数据集的列名列表

**响应示例**：
```json
{
  "dataset_id": "xxx",
  "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
}
```

### 4. 按列自定义处理

**接口地址**：`POST /api/v1/data/{dataset_id}/columns/process`

**功能描述**：支持为不同列配置不同的处理方案，处理空值和异常值

**请求参数**：
```json
{
  "columns_config": [
    {
      "col_name": "sepal_length",
      "null_process_method": "mean_cols",
      "outliers_process_method": "fill_outlier"
    }
  ],
  "global_null_process_method": "mean_cols",
  "global_outliers_method": "iqr",
  "global_outliers_threshold": 1.5,
  "global_outliers_process_method": "fill_outlier"
}
```

**响应示例**：
```json
{
  "code": 200,
  "msg": "处理成功",
  "data": {
    "dataset_id": "xxx",
    "columns": ["sepal_length"],
    "null_process_result": {
      "columns_detail": {
        "sepal_length": {
          "null_count_before": 0,
          "null_count_after": 0,
          "null_processed": 0,
          "null_process_method": "mean_cols"
        }
      },
      "total_null_processed": 0
    },
    "outliers_process_result": {
      "columns_detail": {
        "sepal_length": {
          "original_outliers_count": 0,
          "processed_outliers_count": 0,
          "outliers_processed": 0
        }
      },
      "total_outliers_processed": 0
    }
  }
}
```

### 5. 所有字段统一处理

**接口地址**：`POST /api/v1/data/{dataset_id}/process/all`

**功能描述**：对选择的列名同时进行统一的空值和异常值处理

**请求参数**：
```json
{
  "cols": ["sepal_length", "sepal_width"],
  "null_process_method": "mean_cols",
  "null_fill_value": null,
  "outliers_method": "iqr",
  "outliers_threshold": 1.5,
  "outliers_process_method": "fill_outlier",
  "outliers_fill_value": null
}
```

**响应示例**：
```json
{
  "code": 200,
  "msg": "处理成功",
  "data": {
    "dataset_id": "xxx",
    "columns": ["sepal_length", "sepal_width"],
    "null_process_result": {
      "columns_detail": {
        "sepal_length": {
          "null_count_before": 0,
          "null_count_after": 0,
          "null_processed": 0,
          "null_process_method": "mean_cols"
        },
        "sepal_width": {
          "null_count_before": 0,
          "null_count_after": 0,
          "null_processed": 0,
          "null_process_method": "mean_cols"
        }
      },
      "total_null_processed": 0
    },
    "outliers_process_result": {
      "columns_detail": {
        "sepal_length": {
          "original_outliers_count": 0,
          "processed_outliers_count": 0,
          "outliers_processed": 0
        },
        "sepal_width": {
          "original_outliers_count": 0,
          "processed_outliers_count": 0,
          "outliers_processed": 0
        }
      },
      "total_outliers_processed": 0
    }
  }
}
```

### 6. 数据集划分

**接口地址**：`POST /api/v1/data/{dataset_id}/split`

**功能描述**：数据集划分接口，支持训练/验证/测试集划分，包含特征编码

**请求参数**：
```json
{
  "label_column": "species",
  "feature_cols": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "test_size": 0.2,
  "val_size": 0.1,
  "use_stratified": true,
  "processed_ID": "xxx_processed_20260311_100000",
  "onehot_encode_cols": [],
  "ordinal_encode_cols": [],
  "is_standard": true
}
```

**响应示例**：
```json
{
  "code": 200,
  "msg": "数据集划分成功",
  "data": {
    "dataset_id": "xxx",
    "split_id": "xxx_split_20260311_100000",
    "label_column": "species",
    "data_statistics": {
      "total_size": 150,
      "train_set_size": 105,
      "val_set_size": 15,
      "test_set_size": 30,
      "train_ratio": 0.7,
      "val_ratio": 0.1,
      "test_ratio": 0.2,
      "label_distribution": {
        "0(setosa)": 50,
        "1(versicolor)": 50,
        "2(virginica)": 50
      }
    }
  }
}
```

### 7. 模型训练

**接口地址**：`POST /api/v1/models/{problem_type}/{model_type}/train`

**功能描述**：模型训练接口，支持分类和回归任务，多种算法模型

**请求参数**：
```json
{
  "dataset_id": "xxx",
  "split_id": "xxx_split_20260311_100000",
  "model_params": {
    "n_estimators": 100,
    "max_depth": 5
  }
}
```

**路径参数**：
- `problem_type`：问题类型（classification/regression）
- `model_type`：模型类型（random_forest/xgboost/lightgbm/svm/knn/decision_tree等）

**支持的分类模型**：
- random_forest
- xgboost
- gboost
- lightgbm
- svm
- knn
- decision_tree
- logistic_regression
- Ga_naive_bayes
- Be_naive_bayes
- adaboost
- mlp

**支持的回归模型**：
- random_forest
- xgboost
- lightgbm
- svm
- knn
- decision_tree
- LR
- lasso
- gboost
- adaboost
- mlp

**响应示例**：
```json
{
  "code": 200,
  "msg": "模型训练成功",
  "data": {
    "dataset_id": "xxx",
    "split_id": "xxx_split_20260311_100000",
    "problem_type": "classification",
    "model_type": "random_forest",
    "model_params": {
      "n_estimators": 100,
      "max_depth": 5
    },
    "train_time": {
      "start_time": "2026-03-11 10:05:00",
      "end_time": "2026-03-11 10:05:02",
      "duration_seconds": 2.5
    },
    "metrics": {
      "train": {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      },
      "val": {
        "accuracy": 0.9333,
        "precision": 0.9333,
        "recall": 0.9333,
        "f1": 0.9333
      },
      "test": {
        "accuracy": 0.9667,
        "precision": 0.9667,
        "recall": 0.9667,
        "f1": 0.9667
      }
    },
    "model_save_path": "./uploads/split_datasets/xxx_split_20260311_100000/random_forest_model_20260311_100502.pkl"
  }
}
```

## API基本信息

### 基础URL

```
http://localhost:8005
```

### 环境变量配置

| 环境变量名 | 描述 | 默认值 |
|-----------|------|--------|
| ML_PLATFORM_UPLOAD_DIR | 文件上传目录 | ./uploads |
| ML_PLATFORM_LOG_DIR | 日志文件目录 | ./logs |
| ML_PLATFORM_PORT | API服务端口 | 8005 |
| ML_PLATFORM_MAX_FILE_SIZE | 最大文件上传大小 | 100 * 1024 * 1024 (100MB) |

#### bash 终端
```
# 新建脚本文件
touch set_env.sh
# 编辑脚本
cat > set_env.sh << EOF
#!/bin/bash
# ML平台环境变量配置
export ML_PLATFORM_UPLOAD_DIR="/data/ml/uploads"
export ML_PLATFORM_LOG_DIR="/var/logs/ml_platform"
export ML_PLATFORM_PORT="8080"
export ML_PLATFORM_MAX_FILE_SIZE="209715200"
export ML_PLATFORM_RATE_LIMIT="500/hour"
EOF
# 添加执行权限
chmod +x set_env.sh
# 执行脚本
source set_env.sh

```

### 错误码说明

| 错误码 | 描述 |
|-------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 413 | 文件大小超过限制 |
| 500 | 服务器内部错误 |


## ML算法模型脚本输出

ML算法模型脚本训练完成后，输出结果保存在`model_output/`目录下，包括：
- 模型文件（.pkl或.pth格式）
- 标签编码器
- 特征重要性图
- 混淆矩阵（分类任务）
- 训练曲线

## 日志管理

API日志保存在`api/logs/`目录下，命名格式为`ml_platform_YYYYMMDD.log`，记录平台运行过程中的关键信息和错误。

## 示例数据集

项目提供了多个示例数据集，位于`Data/`目录下：

1. **鸢尾花数据集**：用于分类任务，包含3种鸢尾花的4个特征
2. **房价数据集**：用于回归任务，包含波士顿房价的13个特征
3. **乳腺组织数据集**：用于分类任务，包含6种乳腺组织的9个特征
4. **个人贷款违约数据集**：用于分类任务，包含个人贷款违约预测相关特征，需经过数据处理，特征编码等

## 注意事项

1. 上传文件大小限制为100MB
2. 建议使用CSV格式数据，性能更好
3. 大规模数据集建议先进行采样处理
4. 模型训练时间取决于数据量和算法复杂度
5. 建议定期清理`uploads/`目录下的临时文件

---

**版本**：1.0.0  
**更新日期**：2026-03-11  
**作者**： wanghan