from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# 数据集列信息响应模型
class DatasetColumnsResponse(BaseModel):
    dataset_id: str = Field(description="数据集唯一标识")
    columns: List[str] = Field(description="数据集列名列表")


# ===================== 按列的异常值、空值处理接口 =====================
# 单列处理配置模型
class ColumnProcessConfig(BaseModel):
    """单列的处理配置"""
    col_name: str = Field(..., description="列名")
    # 空值处理配置
    null_process_method: str = Field(default="mean_cols",
                                     description="空值处理方法：missing_all_cols/missing_50%_cols/mode_cols/median_cols/fill_cols")
    null_fill_value: Optional[float] = Field(default=None, description="空值填充值（仅fill_cols方法生效）")
    # 异常值处理配置
    outliers_method: str = Field(default="iqr", description="异常值检测方法：iqr/std/z_score")
    outliers_threshold: Optional[float] = Field(default=1.5, description="异常值检测阈值")
    outliers_process_method: str = Field(default="save_outlier",
                                         description="异常值处理方法：del_outlier/save_outlier/fill_outlier")
    outliers_fill_value: Optional[float] = Field(default=None, description="异常值填充值（仅fill_outlier方法生效）")
# 多列处理请求模型
class ColumnsProcessRequest(BaseModel):
    """多列处理的请求参数模型"""
    columns_config: List[ColumnProcessConfig] = Field(..., description="所有待处理列的配置列表（支持不同列不同策略）")
    # 全局默认配置
    global_null_process_method: str = Field(default="missing_all_cols", description="全局默认空值处理方法")
    global_outliers_method: str = Field(default="iqr", description="全局默认异常值检测方法")
    global_outliers_threshold: float = Field(default=1.5, description="全局默认异常值检测阈值")
    global_outliers_process_method: str = Field(default="save_outlier", description="全局默认异常值处理方法")


# ===================== 所有字段统一处理接口 =====================
# 所有列统一处理请求模型
class AllColumnsProcessRequest(BaseModel):
    """所有列统一处理的请求参数模型"""
    cols: List[str] = Field(..., description="待处理的列名列表")
    null_process_method: str = Field(
        default="mean_cols",
        description="空值处理方法：missing_all_cols、missing_50%_cols、mode_cols、median_cols、fill_cols"
    )
    null_fill_value: Optional[float] = Field(
        default=3,
        description="空值填充值（仅fill_cols方法生效）"
    )
    outliers_process_method: str = Field(
        default="save_outlier",
        description="异常值处理方法：del_outlier、save_outlier、fill_outlier"
    )
    outliers_fill_value: Optional[float] = Field(
        default=3,
        description="异常值填充值（仅fill_outlier方法生效）"
    )
    outliers_method: str = Field(
        default="iqr",
        description="异常值检测方法，仅支持iqr/std/z_score三种检测方法"
    )
    outliers_threshold: Optional[float] = Field(
        default=1.5,
        description="异常值检测阈值"
    )


# =====================按列的异常值、空值处理接口  所有字段统一处理接口 =====================
# 处理结果响应模型
class ColumnProcessingResponse(BaseModel):
    dataset_id: str = Field(description="数据集唯一标识")
    columns: List[str] = Field(description="处理的列名列表")
    outliers_method: str = Field(description="异常值检测方法，仅支持iqr/std/z_score三种检测方法")
    outliers_threshold: Optional[float] = Field(description="异常值检测阈值", default=None)
    null_process_method: str = Field(description="空值处理方法")
    null_fill_value: Optional[float] = Field(description="空值填充值（仅fill_cols方法生效）", default=None)
    outliers_process_method: str = Field(description="异常值处理方法")
    outliers_fill_value: Optional[float] = Field(description="异常值填充值（仅fill_outlier方法生效）", default=None)
    null_process_result: Dict[str, Any] = Field(description="空值处理结果详情（按列汇总）")
    outliers_process_result: Dict[str, Any] = Field(description="异常值处理结果详情（按列汇总）")
    process_result: Dict[str, Any] = Field(description="整体处理结果汇总")
    validate_report_before: Dict[str, Any] = Field(description="处理前的数据质量报告")
    validate_report_after: Dict[str, Any] = Field(description="处理后的数据质量报告")
    processed_file_path: Optional[str] = Field(description="处理后数据文件的存储路径", default=None)
    processed_ID: Optional[str] = Field(description="处理后数据集ID", default=None)
    processed_time: Optional[str] = Field(description="处理时间戳", default=None)

# ===================== 数据集划分接口 =====================
# 数据集划分请求参数模型
class SplitParams(BaseModel):
    test_size: float = Field(..., ge=0.1, le=0.5, description="测试集比例（0.1~0.5）")
    val_size: float = Field(..., ge=0.0, le=0.4, description="验证集比例（0~0.4）")
    use_stratified: bool = Field(default=True, description="是否分层抽样")
    label_column: str = Field(..., description="标签列名（必须存在于数据集）")
    processed_ID: str = Field(..., description="处理后数据集ID")
    feature_cols: List[str] = Field(..., description="特征列列表")
    onehot_encode_cols: List[str] = Field(default=[], description="需独热编码的列")
    ordinal_encode_cols: List[str] = Field(default=[], description="需序数编码的列")
    class_weight: str = Field(default="balanced", description="类别权重（balanced/none）")
    is_standard: bool = Field(default=False, description="是否标准化特征")
    remainder: str = Field(default="passthrough", description="编码器剩余列处理方式")
# 划分结果响应模型
class SplitDatasetResponse(BaseModel):
    dataset_id: str = Field(description="原始数据集ID")
    processed_ID_used: str = Field(description="实际使用的处理后数据集ID")
    encoded_file_path_used: str = Field(description="实际加载的编码后数据文件路径", default=None)
    processed_file_path_used: str = Field(description="实际加载的处理后数据文件路径", default=None)
    split_id: str = Field(description="划分后的处理数据集ID")
    label_column: str = Field(description="划分使用的标签列")
    split_params: Dict[str, Any] = Field(description="使用的划分参数")
    data_statistics: Dict[str, Any] = Field(description="数据集划分统计信息")
    split_time: str = Field(description="划分完成时间")
    message: str = Field(description="操作结果提示")