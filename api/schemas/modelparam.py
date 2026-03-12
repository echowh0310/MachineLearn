from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB,MultinomialNB, BernoulliNB
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
# ===================== 模型映射字典 =====================
# 分类模型
CLASSIFICATION_MODELS = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "knn": KNeighborsClassifier,  # K近邻分类
    "decision_tree": DecisionTreeClassifier,  # 决策树分类
    "Ga_naive_bayes": GaussianNB,  #高斯贝叶斯：适配连续数值特征（正态分布）
    # "Mu_naive_bayes": MultinomialNB, #多项式贝叶斯：适配离散计数特征（词频/频次） 不支持特征为负
    "Be_naive_bayes": BernoulliNB, #伯努利贝叶斯：适配二值离散特征（有无/0-1）
    "svm": SVC,
    "gboost": GradientBoostingClassifier,
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier,
    "adaboost": AdaBoostClassifier,
    "mlp": MLPClassifier
}

# 回归模型
REGRESSION_MODELS = {
    "random_forest": RandomForestRegressor, #6.随机森林
    "adaboost": AdaBoostRegressor, #7.AdaBoost
    "knn": KNeighborsRegressor,  # 2.KNN
    "decision_tree": DecisionTreeRegressor,  # 5.决策树回归
    "svm": SVR, #3.SVM回归
    "lasso":Lasso,  #4.Lasso
    "gboost": GradientBoostingRegressor, #8.GBDT回归（梯度提升决策树回归）
    "xgboost": XGBRegressor, #9.XGBoost极限梯度提升
    "lightgbm": LGBMRegressor, #10.基于梯度提升决策树
    "mlp": MLPRegressor, #11.多层感知机
    "LR": LinearRegression #1.线性回归模型//
}

# ===================== 默认模型参数 =====================
DEFAULT_MODEL_PARAMS = {
    # 分类模型默认参数
    "classification": {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 3,
            "min_samples_split": 2,
            "random_state": 42
        },

        "gboost": {
            "n_estimators": 100,
            "max_depth": 3,
            "loss": "log_loss",
            "learning_rate": 0.1,
            "random_state": 42
        },
        "logistic_regression": {
            "max_iter": 1000,
            "penalty": "l2",
            "C": 1.0,
            "random_state": 42
        },
        "knn": {
            "n_neighbors": 5,
            "weights": "uniform",
            "p": 2,
            "metric": "minkowski"
        },
        "decision_tree": {
            "max_depth": None,
            "criterion": 'gini',
            "max_depth": 3 ,
            "min_samples_split": 2,
            "splitter": "best",
            "random_state": 42
        },
        "Ga_naive_bayes": {},
        # "Mu_naive_bayes": {},
        "Be_naive_bayes": {},
        "svm": {
            "kernel": "rbf",
            "gamma": "scale",
            "random_state": 42,
            "probability": False
        },
        "xgboost": {
            "objective": "multi:softmax",  # "binary:logistic"二分类，多分类需动态改为"multi:softmax"
            "n_estimators": 100,
            "learning_rate": 0.1,
            "random_state": 42
        },
        "lightgbm": {
            "n_estimators": 100,
            "objective": "multiclass",  # "binary"二分类，多分类需动态改为"multiclass"
            "learning_rate": 0.1,
            "random_state": 42
        },
        "adaboost": {
            "n_estimators": 50,
            "learning_rate": 1.0,
            "random_state": 42
        },
        "mlp": {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "learning_rate": "adaptive",
            "learning_rate_init": 0.001,
            "max_iter": 300,
            "early_stopping": False,
            "random_state": 42 }
    },
    # 回归模型默认参数
    "regression": {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": None,
            "random_state": 42
        }, #随机森林回归
        "LR": {},  # 线性回归
        "knn": {
            "n_neighbors": 5,
            "weights": "uniform",
            "p": 2
        },  # K近邻回归
        "decision_tree": {
            "max_depth": None,
            "random_state": 42,
            "min_samples_split": 2,
            "min_samples_leaf": 1
        },  # 决策树回归默认参数
        "svm": {
            "kernel": "rbf",
            "gamma": "scale"

        }, #支持向量机回归
        "lasso": {
            "alpha": 3.0,
            "max_iter": 1000,
            "selection": "cyclic"
        }, #岭回归
        "adaboost": {
            "n_estimators": 50,           # 弱学习器数量（默认50）
            "learning_rate": 1.0,         # 学习率（越小需要更多弱学习器）
            "loss": "linear",             # 损失函数（linear=线性损失，适配回归）
            "random_state": 42
        },
        "gboost": {
            "loss": "squared_error",  # 损失函数
            "learning_rate": 0.1,  # 学习率（默认0.1）
            "n_estimators": 100,  # 弱学习器数量
            "subsample": 1.0,  # 每轮训练的样本抽样比例
            "criterion": "friedman_mse",  # 分割准则（friedman_mse更适配梯度提升）
            "max_depth": 3,  # 单树最大深度（默认3，防止过拟合）
            "random_state": 42
        },
        "xgboost": {
            "objective": "reg:squarederror", # 目标函数（平方误差）
            "learning_rate": 0.1,            # 学习率
            "n_estimators": 100,             # 树数量
            "max_depth": 3,                  # 树深度
            "subsample": 1.0,                # 样本抽样比例
            "colsample_bytree": 1.0,         # 特征抽样比例
            "gamma": 0,                      # 分割所需最小损失减少量
            "reg_alpha": 0,                  # L1正则化
            "reg_lambda": 1,                 # L2正则化
            "random_state": 42
        },
    "lightgbm": {
            "objective": "regression",       # 任务类型
            "metric": "mse",                 # 评估指标（均方误差）
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": -1,                 # 树深度（-1=不限制，LightGBM默认）
            "num_leaves": 31,                # 叶子节点数（默认31）
            "subsample": 1.0,                # 样本抽样比例
            "colsample_bytree": 1.0,         # 特征抽样比例
            "random_state": 42
        },
    "mlp": {
            "hidden_layer_sizes": (64, 32),    # 隐藏层结构
            "activation": "relu",            # 激活函数（ReLU适配大部分场景）
            "solver": "adam",                # 优化器（adam=自适应梯度）
            "alpha": 0.0001,                 # L2正则化强度
            "learning_rate": "adaptive",     # 学习率策略（constant=固定、 adaptive=自适应）
            "learning_rate_init": 0.001,     # 初始学习率
            "max_iter": 1000,                 # 最大迭代数
            "random_state": 42,
            "early_stopping": True         # 是否早停（默认False，小数据无需）
        }
    }
}