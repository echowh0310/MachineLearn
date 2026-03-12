from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# ===================== 模型训练接口 =====================
# 训练参数请求模型
class TrainRequest(BaseModel):
    dataset_id: str = Field(..., description="原始数据集ID")
    split_id: str = Field(..., description="使用的划分数据集ID")
    model_params: Optional[Dict[str, Any]] = Field({}, description="模型自定义参数（会覆盖默认参数）")

# 训练结果响应模型
class TrainResponse(BaseModel):
    dataset_id: str = Field(description="原始数据集ID")
    split_id: str = Field(description="使用的划分数据集ID")
    problem_type: str = Field(description="问题类型（classification/regression）")
    model_type: str = Field(description="模型类型（random_forest/xgboost/lightgbm/svm等）")
    model_params: Dict[str, Any] = Field(description="模型参数")
    train_time: Dict[str, Any] = Field(description="训练时间信息")
    data_shape: Dict[str, Any] = Field(description="各数据集形状")
    metrics: Dict[str, Any] = Field(description="评估指标")
    label_mapping: Optional[Dict[str, Any]] = Field({}, description="标签映射关系")
    model_save_path: str = Field(description="模型保存路径")
    message: str = Field(description="操作结果提示")