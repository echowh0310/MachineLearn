import uuid
import os
import sys
import pandas as pd

# 导入FastAPI相关依赖
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Form, Path, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta

# 添加项目根目录到系统路径，模块搜索
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# 导入目录配置
from api.schemas.env_config import (
    UPLOAD_DIR, PROCESSED_DIR,
    SAVE_DIR, LOG_DIR,
    API_PORT, ORIGIN,
    MAX_FILE_SIZE
)
# 创建必要目录
for dir_path in [UPLOAD_DIR, PROCESSED_DIR, SAVE_DIR, LOG_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

# 导入JWT配置
from api.schemas.jwt_config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, ACCESS_TOKEN_EXPIRE_DAYS

# 导入请求响应模型
from api.schemas.base import BaseResponse
from api.schemas.data import (
    DatasetColumnsResponse,
    ColumnsProcessRequest,
    AllColumnsProcessRequest,
    ColumnProcessingResponse,
    SplitParams,
    SplitDatasetResponse
)
from api.schemas.models import TrainRequest, TrainResponse

# 导入模型参数及映射
from api.schemas.modelparam import CLASSIFICATION_MODELS, REGRESSION_MODELS, DEFAULT_MODEL_PARAMS
# 认证模型
from api.schemas.login import Token, TokenData, UserInDB, LoginRequest
# 导入自定义模块
from core.utils.logger import setup_logger
from core.utils.jsonencoder import CustomJSONEncoder

from core.data_processing.data_loader import DataLoader
from core.data_processing.dataset_splitter import DatasetSplitter
from core.model.evaluate_model import ModelEvaluator
# 从login_model导入密码哈希函数，确保生成和验证使用同一个pwd_context
from core.login.login_model import authenticate_user, create_access_token, get_user, get_password_hash


loader = DataLoader()
dataset_splitter = DatasetSplitter()
model_evaluator = ModelEvaluator()
json_encoder = CustomJSONEncoder()

# 初始化日志器
logger = setup_logger("ml_platform_api",LOG_DIR)

# 元数据存储字典（数据库存储数据）
datasets = {}  # 存储上传的数据集
encoded_datasets = {}  # 存储特征编码结果

# Windows 系统编码适配
if sys.platform == "win32":
    import io
    import locale
    # 1. 强制控制台输出UTF-8编码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
    # 2. 设置系统默认区域编码为UTF-8
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        # 兼容无UTF-8区域设置的系统
        locale.setlocale(locale.LC_ALL, '')



# 用户数据库 - 动态生成密码哈希
from api.schemas.users_db import users_db


#  初始化FastAPI app
app = FastAPI(
    title="机器学习算法平台API",
    description="提供数据集上传、处理、划分和训练等功能的API",
    version="1.0.0"
)

# 认证依赖项
token_auth_scheme = HTTPBearer()


# 验证token
async def decode_and_verify_token(token: str):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception

# 获取当前用户
async def get_current_user(token: str = Depends(token_auth_scheme)):
    return await decode_and_verify_token(token.credentials)

# 配置CORS允许跨域
origins = [
    "http://127.0.0.1:8005",
    "http://localhost:63342"
]

# 统一CORS配置
app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_origins=["*"],#特定域名
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # 方法
    allow_headers=["Content-Type", "Authorization"],  # 常用头部
    expose_headers=["X-Total-Count", "X-Processing-Progress"],
)

# ===================== 路由定义 =====================

from fastapi import Request
from fastapi.templating import Jinja2Templates  # 模板引擎
from fastapi.responses import HTMLResponse
# templates = Jinja2Templates(directory="../templates")
# 使用绝对路径确保模板文件能被正确找到
import os
import jinja2
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")

# 创建Jinja2环境，禁用缓存
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# ===================== 登录接口 =====================
@app.post("/api/login", response_model=Token)
async def login(login_request: LoginRequest):
    """登录接口"""
    # 验证用户
    user = authenticate_user(users_db, login_request.username, login_request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="账号或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 设置token过期时间
    if login_request.remember:
        # 记住我：7天过期
        access_token_expires = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    else:
        # 不记住我：30分钟过期
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # 创建token
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    
    # 返回token信息
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": int(access_token_expires.total_seconds())
    }

# ===================== Token验证接口 =====================
@app.get("/api/verify-token")
async def verify_token(current_user: str = Depends(get_current_user)):
    logger.info(f"username: {current_user}")
    """验证token有效性接口"""
    return {
        "code": 200,
        "msg": "Token is valid",
        "data": {"username": current_user}
    }

# ===================== 根路径处理 =====================
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """根路径，根据认证状态返回登录页面或主应用页面"""
    # 对于根路径访问，直接返回登录页面
    return templates.TemplateResponse("login.html", {"request": request})

# ===================== 主应用页面 =====================
@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request):
    """主应用页面，已登录用户才能访问"""
    # 获取Authorization头
    auth_header = request.headers.get("Authorization")
    token = None
    
    # 检查是否有Bearer令牌
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
    
    # 从查询参数获取token（如果有）
    if not token:
        token = request.query_params.get("token")
    
    # 检查localStorage中的令牌需要通过前端处理，这里只处理请求头和查询参数中的令牌
    if not token:
        return templates.TemplateResponse("login.html", {"request": request})
    
    try:
        # 验证令牌
        await decode_and_verify_token(token)
        # 令牌有效，返回主应用页面，该页面默认显示数据上传界面
        return templates.TemplateResponse("index.html", {"request": request})
    except HTTPException:
        # 令牌无效，返回登录页面
        return templates.TemplateResponse("login.html", {"request": request})

# ===================== 上传数据集接口 =====================
@app.post("/api/v1/data/upload", response_model=BaseResponse)
async def upload_file(
        file: UploadFile = File(..., description="待上传的数据集文件（csv/xlsx/json）"),
        outliers_method: str = Form(default="iqr", description="异常值检测方法：iqr/std/z_score"),
        outliers_threshold: float = Form(default=1.5, description="异常值检测阈值：iqr=1.5/3；std/z_score=3"),
        current_user: str = Depends(get_current_user)
):
    """
    上传数据集并完成数据质量验证
    :param file: 上传文件（支持csv/xlsx/json）
    :param outliers_method: 异常值检测方法
    :param outliers_threshold: 异常值检测阈值
    """
    try:
        # 1. 读取文件内容并验证大小
        file_content = await file.read()
        file_size = len(file_content)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"文件大小超过限制（最大为{MAX_FILE_SIZE / 1024 / 1024:.1f} MB）"
            )
        await file.seek(0)  # 重置文件指针

        # 2. 验证文件格式
        file_ext = os.path.splitext(file.filename)[-1].lower()
        if file_ext not in loader.supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式：{file_ext}，仅支持{loader.supported_formats}"
            )

        # 3. 生成唯一文件名并保存
        file_name = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, file_name)
        try:
            with open(file_path, "wb") as buffer:
                buffer.write(file_content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"文件写入失败：{str(e)}")

        # 4. 加载数据并验证质量（含异常值检测）
        try:
            data = loader.load_data(file_path)
        except ValueError as e:
            # 删除无效文件
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"数据加载失败：{str(e)}")

        validate_report = loader.validate_data(data, outliers_method=outliers_method, outliers_threshold=outliers_threshold)

        # 5. 生成数据集ID和上传时间
        dataset_id = str(uuid.uuid4())
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 6. 存储数据集元信息
        datasets[dataset_id] = {
            "dataset_id": dataset_id,
            "file_name": file_name,
            "file_size": file_size,
            "file_path": file_path,
            "upload_time": upload_time,
            "validate_report": validate_report,
            "outlier_config": {"method": outliers_method, "threshold": outliers_threshold},
            "original_file_name": file.filename
        }
        logger.info(f"数据集上传成功：ID={dataset_id}，文件名={file.filename}（存储为{file_name}）")

        # 7. 返回结果
        return {
            "code": 200,
            "msg": "上传成功",
            "data": {
                "dataset_id": dataset_id,
                "original_file_name": file.filename,
                "stored_file_name": file_name,
                "file_size": file_size,
                "file_size_mb": round(file_size / 1024 / 1024, 2),
                "file_path": file_path,
                "upload_time": upload_time,
                "validate_report": validate_report
            }
        }

    except HTTPException as e:
        logger.error(f"文件上传失败：{e.detail}")
        return {
            "code": e.status_code,
            "msg": e.detail
        }
    except Exception as e:
        logger.error(f"文件上传失败：{str(e)}", exc_info=True)
        return {
            "code": 400,
            "msg": f"文件上传失败：{str(e)}"
        }

# ===================== 获取数据集列名接口 =====================
@app.get("/api/v1/data/{dataset_id}/columns", response_model=DatasetColumnsResponse)
async def get_dataset_columns(
        dataset_id: str,
        current_user: str = Depends(get_current_user)
):
    """
    根据数据集ID获取数据集的列名列表
    :param dataset_id: 数据集唯一标识
    """
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail=f"数据集ID {dataset_id} 不存在")

    try:
        validate_report = datasets[dataset_id]["validate_report"]
        columns = list(validate_report["columns_info"].keys())

        logger.info(f"获取数据集{dataset_id}列名成功，列数：{len(columns)}")
        return {
            "dataset_id": dataset_id,
            "columns": columns
        }
    except Exception as e:
        logger.error(f"获取数据集{dataset_id}列名失败：{str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"获取列名失败：{str(e)}"
        )

# ===================== 按列的异常值、空值处理接口 =====================


@app.post("/api/v1/data/{dataset_id}/columns/process", response_model=BaseResponse)
async def process_columns(
        dataset_id: str,
        process_request: ColumnsProcessRequest = Body(..., description="多列自定义处理配置，支持选择部分列进行处理"),
        current_user: str = Depends(get_current_user)
):
    """
    支持为不同列配置不同的处理方案，处理空值和异常值
    允许选择部分列进行处理，每列可配置独立的处理策略
    :param dataset_id: 数据集ID
    :param process_request: 包含待处理列配置的请求体，仅处理配置中指定的列
    """
    # 1. 基础校验：数据集是否存在
    if dataset_id not in datasets:
        return {
            "code": 404,
            "msg": f"数据集{dataset_id}不存在"
        }

    dataset_meta = datasets[dataset_id]
    file_path = dataset_meta.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return {
            "code": 404,
            "msg": f"数据集文件不存在：{file_path}"
        }

    try:
        # 2. 加载数据集
        data_original = loader.load_data(file_path)  # 原始数据

        # 3. 提取所有特征列+ 校验配置中的列名
        validate_report_before = dataset_meta["validate_report"]
        all_feature_columns = list(validate_report_before["columns_info"].keys())  # 所有特征列
        columns_config = process_request.columns_config

        # 校验配置中的列是否存在
        config_col_names = [cfg.col_name.strip() for cfg in columns_config]
        invalid_cols = [col for col in config_col_names if col not in all_feature_columns]
        if invalid_cols:
            raise HTTPException(
                status_code=400,
                detail=f"配置中包含无效列：{invalid_cols}，数据集所有有效列：{all_feature_columns}"
            )

        # 4. 初始化处理结果容器
        process_result = {
            "total_columns_processed": len(columns_config),
            "rows_before_process": len(data_original),
            "rows_after_process": len(data_original),
            "rows_removed": 0,
            "rows_removed_ratio": 0.0
        }

        null_process_result = {
            "columns_detail": {},  # 按列存储空值处理结果
            "total_null_processed": 0,
            "null_process_summary": {}
        }

        outliers_process_result = {
            "columns_detail": {},  # 按列存储异常值处理结果
            "total_outliers_processed": 0
        }

        # 5. 创建数据副本用于处理
        data_processed = data_original.copy()

        # 6. 按列执行自定义处理
        for idx, col_config in enumerate(columns_config):
            col_name = col_config.col_name.strip()
            # 列类型校验（非数值列仅支持save_outlier）
            col_dtype = validate_report_before["columns_info"].get(col_name, {}).get("dtype", "")
            if col_dtype not in ["int64", "float64"] and col_config.outliers_process_method != "save_outlier":
                logger.info(f"列 {col_name} 是非数值类型（{col_dtype}），仅支持save_outlier异常值处理方法")
                raise HTTPException(
                    status_code=400,
                    detail=f"列 {col_name} 是非数值类型（{col_dtype}），仅支持save_outlier异常值处理方法"
                )

            # 6.1 处理当前列的空值
            # 优先使用列级配置，无则用全局默认
            null_method = col_config.null_process_method or process_request.global_null_process_method
            null_fill_val = col_config.null_fill_value

            # 记录处理前空值数量
            null_count_before = data_processed[col_name].isnull().sum()
            # 执行空值处理
            data_processed = loader.process_single_col_null(
                process_style=null_method,
                fill_context=null_fill_val,
                col=col_name,
                data=data_processed
            )
            # 记录处理后空值数量
            null_count_after = data_processed[col_name].isnull().sum()
            null_processed = null_count_before - null_count_after

            # 保存该列空值处理结果
            null_process_result["columns_detail"][col_name] = {
                "null_count_before": null_count_before,
                "null_count_after": null_count_after,
                "null_processed": null_processed,
                "null_process_method": null_method,
                "null_fill_value": null_fill_val if null_method == "fill_cols" else None
            }
            null_process_result["total_null_processed"] += null_processed

            # 6.2 处理当前列的异常值
            # 优先使用列级配置，无则用全局默认
            outlier_method = col_config.outliers_method or process_request.global_outliers_method
            outlier_threshold = col_config.outliers_threshold or process_request.global_outliers_threshold
            outlier_process_method = col_config.outliers_process_method or process_request.global_outliers_process_method
            outlier_fill_val = col_config.outliers_fill_value

            # 动态检测异常值
            updated_outliers_info_col = loader.validate_data(
                data=data_processed[[col_name]],
                outliers_method=outlier_method,
                outliers_threshold=outlier_threshold
            )["columns_info"][col_name]
            outliers_range = updated_outliers_info_col.get("null_outliers_range")

            if outliers_range is not None:
                # 执行异常值处理
                updated_outliers_info, data_processed = loader.process_outlier_single_col(
                    data_processed=data_processed,
                    col=col_name,
                    outliers_info_col=updated_outliers_info_col,
                    process_outliers_method=outlier_process_method,
                    fill_value=outlier_fill_val
                )

                # 记录异常值处理结果
                original_count = updated_outliers_info_col.get("outliers_count", 0)

                if outlier_process_method == "save_outlier":
                    processed_count = original_count  # 保留异常值，数量不变
                else:
                    # 从更新后的异常值信息中获取处理后的异常值数量
                    processed_count = updated_outliers_info.get(col_name, {}).get("outliers_count", 0)

                outliers_processed = original_count - processed_count

                outliers_process_result["columns_detail"][col_name] = {
                    "original_outliers_count": original_count,
                    "processed_outliers_count": processed_count,
                    "outliers_processed": outliers_processed,
                    "fill_value": outlier_fill_val if outlier_process_method == "fill_outlier" else None,
                    "outliers_method": outlier_method,
                    "outliers_threshold": outlier_threshold,
                    "outliers_process_method": outlier_process_method
                }
                outliers_process_result["total_outliers_processed"] += outliers_processed
            else:
                # 无异常值，记录空结果
                outliers_process_result["columns_detail"][col_name] = {
                    "original_outliers_count": 0,
                    "processed_outliers_count": 0,
                    "outliers_processed": 0,
                    "fill_value": None,
                    "outliers_method": outlier_method,
                    "outliers_threshold": outlier_threshold,
                    "outliers_process_method": outlier_process_method
                }

        # 7. 补充列级配置的响应
        # 7.1 空值处理结果汇总
        remaining_missing = {col: data_processed[col].isnull().sum() for col in config_col_names if
                             data_processed[col].isnull().sum() > 0}
        null_process_result["null_process_summary"]["remaining_missing"] = remaining_missing

        # 7.2 更新总行数统计
        process_result["rows_after_process"] = len(data_processed)
        process_result["rows_removed"] = len(data_original) - len(data_processed)
        process_result["rows_removed_ratio"] = process_result["rows_removed"] / len(data_original) if len(
            data_original) > 0 else 0.0

        # 7.3 重新生成数据质量报告
        validate_report_after = loader.validate_data(
            data=data_processed,
            outliers_method=process_request.global_outliers_method,
            outliers_threshold=process_request.global_outliers_threshold
        )

        # 7.4 保存处理后的数据
        processed_file_path = None
        processed_id = None
        processed_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        safe_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_file_name = f"{dataset_id}_processed_{safe_timestamp}.csv"
        processed_file_path = os.path.join(PROCESSED_DIR, processed_file_name)

        data_processed.to_csv(processed_file_path, index=False, encoding="utf-8-sig")
        if not os.path.exists(processed_file_path):
            raise Exception("处理后文件保存失败")

        processed_id = f"{dataset_id}_processed_{safe_timestamp}"
        # 更新数据集元信息
        datasets[dataset_id].update({
            "processed_ID": processed_id,
            "processed_file_path": processed_file_path,
            "processed_time": processed_time,
            "processing_config": process_request.model_dump(),  # 保存自定义列配置
            "all_feature_columns": all_feature_columns,  # 记录所有特征列
            "validate_report": validate_report_after  # 更新数据集的验证报告
        })

        # 7.5 类型转换
        converted_null_process_result = loader._to_python_type(null_process_result)
        converted_outliers_process_result = loader._to_python_type(outliers_process_result)
        converted_process_result = loader._to_python_type(process_result)
        converted_validate_report_before = loader._to_python_type(validate_report_before)
        converted_validate_report_after = loader._to_python_type(validate_report_after)

        # 8. 构造响应
        processing_result = ColumnProcessingResponse(
            dataset_id=dataset_id,
            columns=config_col_names,  # 返回实际处理的列
            outliers_method=process_request.global_outliers_method,
            outliers_threshold=process_request.global_outliers_threshold,
            null_process_method=process_request.global_null_process_method,
            null_fill_value=None,  # 列级填充值已在columns_detail中
            outliers_process_method=process_request.global_outliers_process_method,
            outliers_fill_value=None,  # 列级填充值已在columns_detail中
            null_process_result=converted_null_process_result,  # 分列空值结果
            outliers_process_result=converted_outliers_process_result,  # 分列异常值结果
            process_result=converted_process_result,
            validate_report_before=converted_validate_report_before,
            validate_report_after=converted_validate_report_after,
            processed_file_path=processed_file_path,
            processed_time=processed_time,
            processed_ID=processed_id
        )

        logger.info(f"数据集{dataset_id}按列处理成功，处理列数：{len(columns_config)}")

        # 9. 返回结果
        return {
            "code": 200,
            "msg": "处理成功",
            "data": processing_result.model_dump()
        }

    except HTTPException as e:
        return {
            "code": e.status_code,
            "msg": e.detail
        }
    except Exception as e:
        error_msg = f"处理失败：{str(e)}"
        logger.error(f"数据集{dataset_id}按列处理失败：{str(e)}", exc_info=True)
        return {
            "code": 500,
            "msg": error_msg
        }

# ===================== 所有字段统一处理接口 =====================


@app.post("/api/v1/data/{dataset_id}/process/all", response_model=BaseResponse)
async def process_all_columns(
        dataset_id: str,
        process_request: AllColumnsProcessRequest = Body(..., description="列统一处理配置，支持选择部分列进行处理"),
        current_user: str = Depends(get_current_user)
):
    """
    对选择的列名同时进行统一的空值和异常值处理
    允许选择部分列进行处理，所有选定列使用相同的处理策略，提高处理效率
    :param dataset_id: 数据集ID
    :param process_request: 包含待处理列列表和统一处理配置的请求体
    """
    # 1. 基础校验：数据集是否存在
    if dataset_id not in datasets:
        return {
            "code": 404,
            "msg": f"数据集{dataset_id}不存在"
        }

    dataset_meta = datasets[dataset_id]
    file_path = dataset_meta.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return {
            "code": 404,
            "msg": f"数据集文件不存在：{file_path}"
        }

    try:
        # 2. 加载数据集
        data_original = loader.load_data(file_path)  # 原始数据

        # 3. 处理列名参数
        cols_list = process_request.cols
        cols_list = list(dict.fromkeys(cols_list))  # 去重且保留顺序
        if not cols_list:
            raise HTTPException(status_code=400, detail="待处理列名不能为空")

        # 4. 校验列名有效性 + 非数值列校验
        validate_report_before = dataset_meta["validate_report"]
        valid_columns = list(validate_report_before["columns_info"].keys()) if validate_report_before["columns_info"] else []

        for col in cols_list:
            if col not in valid_columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"列名 {col} 不存在于数据集，可用列：{valid_columns}"
                )
            # 非数值列仅支持save_outlier
            col_dtype = validate_report_before["columns_info"].get(col, {}).get("dtype", "")
            if col_dtype not in ["int64", "float64"] and process_request.outliers_process_method != "save_outlier":
                raise HTTPException(
                    status_code=400,
                    detail=f"列 {col} 是非数值类型（{col_dtype}），仅支持save_outlier方法"
                )

        # 5. 初始化处理结果
        process_result = {
            "total_columns_processed": len(cols_list),
            "rows_before_process": len(data_original),
            "rows_after_process": len(data_original),
            "rows_removed": 0,
            "rows_removed_ratio": 0.0
        }

        # 初始化空值处理结果
        null_process_result = {
            "columns_detail": {},
            "total_null_processed": 0,
            "null_process_summary": {}
        }

        # 初始化异常值处理结果
        outliers_process_result = {
            "columns_detail": {},
            "total_outliers_processed": 0
        }

        # 6. 创建数据副本
        data_processed = data_original.copy()

        # 7. 批量处理空值
        for col in cols_list:
            if col not in valid_columns:
                continue

            # 记录处理前空值数量
            null_count_before = data_processed[col].isnull().sum()

            # 执行空值处理
            data_processed = loader.process_single_col_null(
                process_style=process_request.null_process_method,
                fill_context=process_request.null_fill_value,
                col=col,
                data=data_processed
            )

            # 记录处理后空值数量
            null_count_after = data_processed[col].isnull().sum()
            null_processed = null_count_before - null_count_after

            # 保存该列空值处理详情
            null_process_result["columns_detail"][col] = {
                "null_count_before": null_count_before,
                "null_count_after": null_count_after,
                "null_processed": null_processed,
                "null_process_method": process_request.null_process_method,
                "null_fill_value": process_request.null_fill_value if process_request.null_process_method == "fill_cols" else None
            }
            null_process_result["total_null_processed"] += null_processed

        # 8. 批量处理异常值（按列处理）
        for col in cols_list:
            # 动态检测异常值
            updated_outliers_info_col = loader.validate_data(
                data=data_processed[[col]],
                outliers_method=process_request.outliers_method,
                outliers_threshold=process_request.outliers_threshold
            )["columns_info"][col]
            outliers_range = updated_outliers_info_col.get("null_outliers_range")

            if outliers_range is None:
                logger.info(f"列{col}无异常值范围信息，跳过处理")
                outliers_process_result["columns_detail"][col] = {
                    "original_outliers_count": 0,
                    "processed_outliers_count": 0,
                    "outliers_processed": 0,
                    "fill_value": None
                }
                continue

            # 处理该列异常值
            updated_outliers_info, data_processed = loader.process_outlier_single_col(
                data_processed=data_processed,
                col=col,
                outliers_info_col=updated_outliers_info_col,
                process_outliers_method=process_request.outliers_process_method,
                fill_value=process_request.outliers_fill_value
            )

            # 记录该列异常值处理详情
            original_count = updated_outliers_info_col.get("outliers_count", 0)
            if process_request.outliers_process_method == "save_outlier":
                processed_count = original_count  # 保留异常值，数量不变
            else:
                processed_count = updated_outliers_info.get(col, {}).get("outliers_count", 0)
            outliers_processed = original_count - processed_count

            outliers_process_result["columns_detail"][col] = {
                "original_outliers_count": original_count,
                "processed_outliers_count": processed_count,
                "outliers_processed": outliers_processed,
                "fill_value": process_request.outliers_fill_value if process_request.outliers_process_method == "fill_outlier" else None
            }
            outliers_process_result["total_outliers_processed"] += outliers_processed

        # 9. 更新处理结果的总行数统计
        process_result["rows_after_process"] = len(data_processed)
        process_result["rows_removed"] = len(data_original) - len(data_processed)
        process_result["rows_removed_ratio"] = (process_result["rows_removed"] / len(data_original)) if len(
            data_original) > 0 else 0.0

        # 10. 重新生成数据质量报告
        updated_validate_report = loader.validate_data(
            data=data_processed,
            outliers_method=process_request.outliers_method,
            outliers_threshold=process_request.outliers_threshold
        )

        # 11. 保存处理后的数据
        processed_file_path = None
        processed_id = None
        processed_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        safe_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_file_name = f"{dataset_id}_processed_{safe_timestamp}.csv"
        processed_file_path = os.path.join(PROCESSED_DIR, processed_file_name)

        data_processed.to_csv(
            processed_file_path,
            index=False,
            encoding="utf-8-sig"
        )

        # 验证文件是否真的保存成功
        if not os.path.exists(processed_file_path):
            raise Exception(f"文件写入后不存在：{processed_file_path}")

        # 生成处理后数据集ID
        processed_id = f"{dataset_id}_processed_{safe_timestamp}"

        # 更新数据集元信息
        datasets[dataset_id]["processed_ID"] = processed_id
        datasets[dataset_id]["processed_file_path"] = processed_file_path
        datasets[dataset_id]["processed_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        datasets[dataset_id]["processing_config"] = {
            "cols": cols_list,
            "null_process_method": process_request.null_process_method,
            "null_fill_value": process_request.null_fill_value,
            "outliers_process_method": process_request.outliers_process_method,
            "outliers_fill_value": process_request.outliers_fill_value,
            "outliers_method": process_request.outliers_method,
            "outliers_threshold": process_request.outliers_threshold
        }
        datasets[dataset_id]["validate_report"] = updated_validate_report

        logger.info(f"处理后数据已保存：{processed_file_path}")

        # 12. 转换所有NumPy类型为Python原生类型
        converted_null_process_result = loader._to_python_type(null_process_result)
        converted_outliers_process_result = loader._to_python_type(outliers_process_result)
        converted_process_result = loader._to_python_type(process_result)
        converted_validate_report_before = loader._to_python_type(validate_report_before)
        converted_validate_report_after = loader._to_python_type(updated_validate_report)

        # 13. 构造响应
        processing_result = ColumnProcessingResponse(
            dataset_id=dataset_id,
            columns=cols_list,
            outliers_method=process_request.outliers_method,
            outliers_threshold=process_request.outliers_threshold,
            null_process_method=process_request.null_process_method,
            null_fill_value=process_request.null_fill_value,
            outliers_process_method=process_request.outliers_process_method,
            outliers_fill_value=process_request.outliers_fill_value,
            null_process_result=converted_null_process_result,
            outliers_process_result=converted_outliers_process_result,
            process_result=converted_process_result,
            validate_report_before=converted_validate_report_before,
            validate_report_after=converted_validate_report_after,
            processed_file_path=processed_file_path,
            processed_time=processed_time,
            processed_ID=processed_id
        )

        logger.info(f"数据集{dataset_id}统一处理完成：处理列={cols_list}，空值方法={process_request.null_process_method}，异常值方法={process_request.outliers_process_method}")

        # 14. 返回结果
        return {
            "code": 200,
            "msg": "处理成功",
            "data": processing_result.model_dump()
        }

    except HTTPException as e:
        return {
            "code": e.status_code,
            "msg": e.detail
        }
    except Exception as e:
        error_msg = f"处理失败：{str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "code": 500,
            "msg": error_msg
        }



# ===================== 数据集划分接口 =====================

@app.post("/api/v1/data/{dataset_id}/split", response_model=BaseResponse)
async def split_dataset(
        dataset_id: str,
        split_params: SplitParams = Body(..., description="数据集划分参数"),
        current_user: str = Depends(get_current_user)
):
    """
    数据集划分接口：优先通过处理后数据集ID加载数据，支持训练/验证/测试集划分
    :param dataset_id: 原始数据集ID
    :param split_params: 划分参数（processed_ID 精准定位处理后数据）
    :return: 划分结果（各数据集大小、比例、标签分布等）
    """
    # ========== 一、数据定位及其校验 ==========
    # 1.校验原始数据集是否存在
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail=f"原始数据集ID {dataset_id} 不存在")

    try:
        # 2. 提取并格式化参数
        label_column = split_params.label_column.strip()
        feature_cols = split_params.feature_cols
        test_size = split_params.test_size
        val_size = split_params.val_size
        use_stratified = split_params.use_stratified #分类任务：数据集分层拆分；回归任务：数据集无需分层
        processed_ID = split_params.processed_ID.strip() if split_params.processed_ID else None
        onehot_encode_cols = split_params.onehot_encode_cols
        ordinal_encode_cols = split_params.ordinal_encode_cols
        class_weight = split_params.class_weight #类别权重
        is_standard = split_params.is_standard
        remainder = split_params.remainder

        #划分参数比例校验
        if test_size + val_size >= 0.9:
            raise HTTPException(
                status_code=400,
                detail=f"测试集比例({test_size}) + 验证集比例({val_size}) = {test_size + val_size} ≥ 0.9，训练集比例不足10%，无法划分"
            )

        # 3. 通过processed_ID精准定位处理后数据
        processed_file_path = None
        if processed_ID:
            # 遍历所有数据集元信息，匹配processed_ID
            matched_meta = None
            for ds_meta in datasets.values():
                if ds_meta.get("processed_ID") == processed_ID:
                    matched_meta = ds_meta
                    break

            if not matched_meta:
                # 列出已生成的processed_ID，方便用户排查
                existing_processed_ids = [ds.get("processed_ID") for ds in datasets.values() if ds.get("processed_ID")]
                raise HTTPException(
                    status_code=404,
                    detail=f"处理后数据集ID {processed_ID} 不存在！已生成的处理后数据集ID列表：{existing_processed_ids}"
                )

            # 校验匹配到的processed_ID是否属于当前原始数据集
            if matched_meta.get("dataset_id") != dataset_id:
                logger.warning(f"处理后数据集ID {processed_ID} 不属于原始数据集 {dataset_id}，仍尝试加载")

            processed_file_path = matched_meta.get("processed_file_path")
            logger.info(f"通过processed_ID {processed_ID} 匹配到文件：{processed_file_path}")
        else:
            # 未指定processed_ID时，使用当前原始数据集的最新处理后文件
            processed_file_path = datasets[dataset_id].get("processed_file_path")
            if not processed_file_path:
                raise HTTPException(
                    status_code=400,
                    detail="未指定processed_ID且原始数据集无处理后文件！请先调用列处理接口生成处理后数据，并传入返回的processed_ID"
                )
            logger.info(f"使用原始数据集 {dataset_id} 的最新处理后文件：{processed_file_path}")

        # 4. 校验处理后文件有效性
        if not processed_file_path:
            raise HTTPException(
                status_code=400,
                detail="未找到处理后数据文件路径！请确认processed_ID正确或重新执行列处理接口"
            )
        if not os.path.exists(processed_file_path):
            raise HTTPException(
                status_code=404,
                detail=f"处理后数据文件不存在：{processed_file_path}，请重新执行列处理接口生成数据"
            )
        if os.path.getsize(processed_file_path) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"处理后数据文件为空：{processed_file_path}"
            )
        # ========== 二、数据加载 字段校验==========
        # 1. 加载处理后数据集
        try:
            raw_data = pd.read_csv(processed_file_path, encoding="utf-8-sig")
            # 重置索引
            raw_data = raw_data.reset_index(drop=True)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"加载处理后数据失败：{str(e)}，文件路径：{processed_file_path}"
            )

        # 2. 校验数据集基本有效性
        if raw_data.empty:
            raise HTTPException(status_code=400, detail="处理后数据集为空，无法划分")
        if len(raw_data) < 10:  # 最小数据量校验，避免划分无意义
            raise HTTPException(status_code=400, detail="数据集行数过少（<10），无法完成有效划分")

        # 3. 校验标签列
        if label_column not in raw_data.columns:
            raise HTTPException(
                status_code=400,
                detail=f"标签列 {label_column} 不存在！数据集有效列：{raw_data.columns.tolist()}"
            )
        if raw_data[label_column].isnull().sum() > 0:
            raise HTTPException(
                status_code=400,
                detail=f"标签列 {label_column} 包含空值（数量：{raw_data[label_column].isnull().sum()}），请先处理空值"
            )
        # 4. 校验特征列
        if feature_cols:
            invalid_feature_cols = [col for col in feature_cols if col not in raw_data.columns]
            if invalid_feature_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"以下特征列不存在：{invalid_feature_cols}，数据集有效列：{raw_data.columns.tolist()}"
                )
        else:
            raise HTTPException(status_code=400, detail="特征列列表不能为空！")
        # ========== 三、特征分类 特征编码（独热编码、序列编码、标准化）==========
        #1. 特征编码
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer
        import numpy as np
        import json

        # 筛选需要编码的列
        valid_encode_cols = {}
        # 处理独热编码列
        if onehot_encode_cols:
            for col in onehot_encode_cols:
                if col in feature_cols and col in raw_data.columns:
                    valid_encode_cols[col] = "onehot"
                else:
                    logger.info(f"编码列 '{col}' 不存在于数据集中，已跳过")
        else:
            logger.info(f"独热编码列为空，已跳过")

        # 处理序列编码列
        if ordinal_encode_cols:
            for col in ordinal_encode_cols:
                if col in feature_cols and col in raw_data.columns:
                    valid_encode_cols[col] = "ordinal"
                else:
                    logger.info(f"编码列 '{col}' 不存在于数据集中，已跳过")
        else:
            logger.info(f"序列编码列为空，已跳过")

        # 2.分离数值特征\未编码文本特征\需要编码的特征
        other_cols = [col for col in feature_cols if col not in valid_encode_cols]  # 筛选出不需要编码的列
        text_cols = []
        numeric_cols = []
        for col in other_cols:
            col_series = raw_data[col].dropna()  # 去除空值避免 dtype 判断错误
            if col_series.empty:
                logger.info(f"警告：列 {col} 全为空值，暂归类为文本列")
                text_cols.append(col)
                continue

            # 判断列类型并分类
            if pd.api.types.is_numeric_dtype(col_series):
                numeric_cols.append(col)  # 数值列
            else:
                text_cols.append(col)   #文本列

        onehot_cols = [col for col, typ in valid_encode_cols.items() if typ == "onehot"]
        ordinal_cols = [col for col, typ in valid_encode_cols.items() if typ == "ordinal"]

        logger.info(f"数值特征列：{numeric_cols}")
        logger.info(f"未编码文本特征列：{text_cols}") #最终会被remainder='drop'丢弃
        logger.info(f"独热编码列：{onehot_cols}")
        logger.info(f"序列编码列：{ordinal_cols}")

        # 校验有效特征列非空
        all_feature_cols = numeric_cols + onehot_cols + ordinal_cols
        if not all_feature_cols:
            raise HTTPException(status_code=400, detail="无有效特征列（数值列+编码列），无法训练模型！")

        # 3.构建预处理流水线
        transformers = []

        # 独热编码
        if onehot_cols:
            onehot_encoder = OneHotEncoder(
                sparse_output=False,
                drop='first',
                handle_unknown='ignore',
                categories='auto'
            )
            transformers.append(('onehot', onehot_encoder, onehot_cols))

        # 序列编码
        if ordinal_cols:
            ordinal_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1,
                categories='auto'
            )
            transformers.append(('ordinal', ordinal_encoder, ordinal_cols))

        # 数值特征, 进行标准化处理(或者直接透传'passthrough')
        if numeric_cols and is_standard:
            transformers.append(('numeric', StandardScaler(), numeric_cols))
        elif numeric_cols and not is_standard:
            transformers.append(('numeric_passthrough', remainder, numeric_cols))

        # 初始化预处理器
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder=remainder,
            verbose_feature_names_out=True  # 保留特征名前缀，便于后续匹配
        )

        # 4.特征提取与编码（先拟合再提取映射）
        X_raw = raw_data[all_feature_cols]
        # 拟合并转换整个数据集
        X_processed = preprocessor.fit_transform(X_raw)

        logger.info(f"编码器剩余列处理方式：remainder = {remainder}")
        feature_names = preprocessor.get_feature_names_out()
        logger.info(f"处理后的特征名称: {list(feature_names)}")

        # ========== 正确提取各特征列的编码映射关系 ==========
        # 先定义存储编码映射的字典
        feature_encoding_mapping = {}  # 存储各特征列的编码映射
        # （1）. 提取独热编码的映射关系（从preprocessor中获取拟合后的编码器）
        if onehot_cols:
            # 从ColumnTransformer中获取拟合后的onehot编码器
            fitted_onehot_encoder = preprocessor.named_transformers_['onehot']
            # 遍历每个独热编码列
            for col_idx, col_name in enumerate(onehot_cols):
                # 拟合后才有categories_属性
                original_categories = fitted_onehot_encoder.categories_[col_idx]
                # 跳过drop='first'的第一个类别
                if len(original_categories) > 1:
                    original_categories = original_categories[1:]
                # 构建映射：原始类别 → 编码后的列名
                encoded_col_prefix = f"onehot__{col_name}_"
                col_mapping = {
                    str(cat): f"{encoded_col_prefix}{str(cat)}"  # 统一转字符串避免类型问题
                    for cat in original_categories
                }
                feature_encoding_mapping[col_name] = col_mapping

        # （2）. 提取序数编码的映射关系
        if ordinal_cols:
            # 从ColumnTransformer中获取拟合后的ordinal编码器
            fitted_ordinal_encoder = preprocessor.named_transformers_['ordinal']
            # 遍历每个序数编码列
            for col_idx, col_name in enumerate(ordinal_cols):
                original_categories = fitted_ordinal_encoder.categories_[col_idx]
                encoded_values = list(range(len(original_categories)))
                # 构建映射：原始类别 → 编码值
                col_mapping = {
                    str(cat): val for cat, val in zip(original_categories, encoded_values)
                }
                # 补充未知值的映射
                col_mapping["UNKNOWN"] = -1
                feature_encoding_mapping[col_name] = col_mapping

        # (3). 数值列无编码映射，按状态标记
        for col_name in numeric_cols:
            if is_standard:
                feature_encoding_mapping[col_name] = "数值列（无编码，标准化）"
            else:
                feature_encoding_mapping[col_name] = "数值列（无编码，透传）"
        # ========== 正确提取各特征列的编码映射关系 ==========

        # 5.将编码后的特征合并回原始数据集，使用编码器生成的特征名
        # 构建编码后的特征DataFrame
        feature_names = preprocessor.get_feature_names_out(all_feature_cols)
        X_processed_df = pd.DataFrame(
            X_processed,
            columns=feature_names,  # 如：onehot__gender_男、ordinal__education_0
            index=raw_data.index
        )

        # 删除原始特征列，保留标签列和其他列
        raw_data_encoded = raw_data.drop(columns=all_feature_cols)  # 删除原始特征列
        raw_data_encoded = pd.concat([raw_data_encoded, X_processed_df], axis=1)  # 合并编码后的特征

        # 记录编码后的特征列，确认非空
        logger.info(f"数值列标准化开关：is_standard = {is_standard}")
        logger.info(f"特征列编码映射关系：{feature_encoding_mapping}")
        logger.info(f"编码后的特征列：{X_processed_df.columns.tolist()}")
        logger.info(f"编码后的特征矩阵形状：{X_processed.shape}")

        if X_processed.shape[1] == 0:
            raise HTTPException(
                status_code=400,
                detail=f"特征编码后维度为0！请检查：\n1. 特征列是否存在\n2. 编码器配置是否正确"
            )
        # ========== 四、标签编码 数据划分==========
        #1. 提取标签并进行编码
        y = raw_data[label_column].values
        le = None
        label_mapping = None
        data_encoded = raw_data_encoded.copy()
        logger.info(f"标签列名：{label_column}")
        logger.info(f"标签列原始类型：{raw_data[label_column].dtype}")
        logger.info(f"标签列前5个值：{raw_data[label_column].head().tolist()}")

        # 步骤1.判断分类（离散）还是回归（连续）
        # 判定规则：
        #  非数值类型 → 需要编码
        #数值类型但唯一值数量少（< 总样本数的10%）且为整数 → 判定为分类标签，需要编码
        is_categorical = False
        # 先判断是否为非数值类型
        if not pd.api.types.is_numeric_dtype(y):
            is_categorical = True
        else:
            # 数值类型但可能是分类标签（如用数字编码的类别）
            unique_count = len(np.unique(y))
            total_count = len(y)
            # 分类标签特征：唯一值少（<10%）且为整数
            if (unique_count / total_count < 0.1) and np.all(np.mod(y, 1) == 0):
                is_categorical = True

        # 步骤2：根据是否为分类标签执行编码逻辑
        if is_categorical:
            # 分类标签：执行LabelEncoder编码
            le = LabelEncoder()
            label_colum_type = "标签列可能为分类标签"
            # 对全局标签进行编码（处理未知类别）
            try:
                y_encoded = le.fit_transform(y)
            except Exception as e:
                # 处理包含NaN的情况
                y_filled = np.where(pd.isna(y), "UNKNOWN", y)
                le.fit(y_filled)
                y_encoded = le.transform(y_filled)
                logger.warning(f"标签列包含空值，已填充为'UNKNOWN'后编码：{str(e)}")

            # 构建标签映射关系（原始值 → 编码值）
            label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            logger.info(f"标签映射关系：{label_mapping}")
            logger.info(f"标签编码完成，共 {len(label_mapping)} 个类别")

            # 替换编码后数据集中的标签列
            data_encoded[label_column] = y_encoded
        else:
            label_colum_type="标签列可能为连续数值类型"
            use_stratified = False
            # 真正的数值标签（如回归任务）：无需编码
            logger.info("标签列为连续数值类型，无需编码（回归任务）")
            y_encoded = y
            # 构建空的映射关系
            label_mapping = None
            # 数值标签直接赋值
            data_encoded[label_column] = y_encoded

        # 统计类别数量（兼容分类/回归场景）
        if is_categorical:
            class_num = len(le.classes_)
            all_labels = le.classes_.tolist()
        else:
            class_num = -1  # 回归任务标记为-1
            all_labels = []

        logger.info(f"数据集类别数：{class_num}，类别列表：{all_labels}")


        # 2. 分层划分专属校验
        if use_stratified:
            # 如果是回归任务（连续数值标签），不允许使用分层划分
            if not is_categorical:
                logger.info(f"标签列 {label_column} 是连续数值类型，自动禁用分层划分")
                use_stratified = False
            else:
                # 分类任务，进行分层划分校验
                label_dtype = raw_data[label_column].dtype
                # 支持的分层标签类型（分类任务）
                supported_dtypes = [int, str, bool, "int64", "str","object", "bool","float64"]
                if str(label_dtype) not in supported_dtypes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"分层划分仅支持分类标签（int/str/bool），当前标签列 {label_column} 类型：{label_dtype}"
                    )
                #float类型需为整数，校验
                if str(label_dtype) == "float64":
                    label_vals = raw_data[label_column].dropna().unique()
                    if not all(val.is_integer() for val in label_vals):
                        logger.info(f"float类型标签列 {label_column} 包含非整数值，无法作为分类标签进行分层划分")
                        raise HTTPException(
                            status_code=400,
                            detail=f"float类型标签列 {label_column} 包含非整数值，无法作为分类标签进行分层划分"
                        )


                # 校验标签类别数量
                label_unique_count = raw_data[label_column].nunique()
                if label_unique_count == 1:
                    raise HTTPException(
                        status_code=400,
                        detail=f"标签列 {label_column} 仅包含1个类别（{raw_data[label_column].unique()[0]}），无法分层划分！\n请关闭use_stratified或更换标签列"
                    )

                # 校验每个类别样本量是否足够划分
                label_counts = raw_data[label_column].value_counts()
                min_label_count = label_counts.min()
                if min_label_count < 5:  # 每个类别至少5个样本才能保证分层有效性
                    raise HTTPException(
                        status_code=400,
                        detail=f"标签列 {label_column} 中存在样本量过少的类别（最小：{min_label_count}），无法分层划分！\n类别分布：{label_counts.to_dict()}"
                    )

        # 3. 执行数据集划分
        try:
            split_datasets = dataset_splitter.split_dataset(
                data=data_encoded,  # 使用特征+标签都编码后的数据集
                label_column=label_column,
                test_size=test_size,
                val_size=val_size,
                use_stratified=use_stratified
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"数据集划分失败：{str(e)}\n建议检查：\n1. 划分比例是否合理（test/val需在0.1-0.3之间）\n2. 标签列是否符合分层要求\n3. 数据集样本量是否充足"
            )

        # 计算划分统计信息
        train_data = split_datasets["train"]
        val_data = split_datasets["val"]
        test_data = split_datasets["test"]
        total_size = len(raw_data)

        # ========== 五、数据保存 元信息==========
        # 1.直接提取划分后的标签
        # 提取划分后的特征和标签（编码后的特征列）
        feature_cols_encoded = [col for col in feature_names if col in train_data.columns]

        X_train = train_data[feature_cols_encoded].values
        X_val = val_data[feature_cols_encoded].values
        X_test = test_data[feature_cols_encoded].values
        y_train = train_data[label_column].values
        y_val = val_data[label_column].values
        y_test = test_data[label_column].values

        logger.info(f"编码后训练集特征维度：{X_train.shape}")
        logger.info(f"编码后验证集特征维度：{X_val.shape}")
        logger.info(f"编码后测试集特征维度：{X_test.shape}")


        # 2.保存预处理后的数据集
        # 生成唯一标识
        split_id = f"{dataset_id}_split_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_path = os.path.join(SAVE_DIR, split_id)
        os.makedirs(save_path, exist_ok=True)

        # 保存特征矩阵+标签数组
        np.savez(
            os.path.join(save_path, "processed_data.npz"),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test
        )
        ##########################
        #处理标签分布
        # 使用编码后的标签分布（与实际划分数据一致）
        if is_categorical:
            # 编码后的标签分布
            label_distribution_encoded = pd.Series(y_encoded).value_counts().to_dict()
            # 映射回原始标签，便于阅读（保留编码后数字+原始文本）
            label_distribution = {
                f"{code}({label})": count
                for label, code in label_mapping.items()
                for c, count in label_distribution_encoded.items()
                if c == code
            }
        else:
            # 回归任务：统计数值分布（均值/中位数/范围）
            label_distribution = {
                "mean": float(np.mean(y_encoded)),
                "median": float(np.median(y_encoded)),
                "min": float(np.min(y_encoded)),
                "max": float(np.max(y_encoded))
            }
        label_distribution = {
            int(k) if isinstance(k, (np.integer, np.int64)) else k: v
            for k, v in label_distribution.items()
        }
        ###############################

        # 3.保存元信息（特征列名、标签映射、预处理流水线等）
        meta_info = {
            "split_id": split_id,
            "dataset_id": dataset_id,
            "label_column": label_column,
            "label_mapping": label_mapping,
            "feature_encoding_mapping": feature_encoding_mapping,  # 保存特征编码映射
            "feature_encoding": X_processed_df.columns.tolist(),
            "all_feature_cols": all_feature_cols,
            "numeric_cols": numeric_cols,
            "onehot_cols": onehot_cols,
            "ordinal_cols": ordinal_cols,
            "split_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_shape": {
                "train": (X_train.shape[0], X_train.shape[1]),
                "val": (X_val.shape[0], X_val.shape[1]),
                "test": (X_test.shape[0], X_test.shape[1])
            }
        }

        #保存编码器
        import joblib
        # 训练接口中，在特征编码后保存预处理流水线
        preprocessor_save_path = os.path.join(save_path, "preprocessor.pkl")
        joblib.dump(preprocessor, preprocessor_save_path)  # 保存拟合好的流水线

        # 同时保存标签编码器（LabelEncoder）
        if le:
            le_save_path = os.path.join(save_path, "label_encoder.pkl")
            joblib.dump(le, le_save_path)

        # 更新元信息，记录保存路径
        meta_info["preprocessor_path"] = preprocessor_save_path
        meta_info["label_encoder_path"] = le_save_path if le else None

        meta_info = loader._to_python_type(meta_info)
        with open(os.path.join(save_path, "meta_info.json"), "w", encoding="utf-8") as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=4)

        # 4.保存为DataFrame
        # 训练集：特征列（编码后）+ 标签列
        train_df = pd.DataFrame(X_train, columns=feature_cols_encoded)
        train_df[label_column] = y_train
        train_df.to_csv(os.path.join(save_path, "train_data.csv"), index=False, encoding="utf-8-sig")

        # 验证集：特征列（编码后）+ 标签列
        val_df = pd.DataFrame(X_val, columns=feature_cols_encoded)
        val_df[label_column] = y_val
        # 验证集非空时保存
        if not val_data.empty:
            val_df.to_csv(os.path.join(save_path, "val_data.csv"), index=False, encoding="utf-8-sig")
        else:
            logger.info("验证集为空，不保存val_data.csv")

        # 测试集：特征列（编码后）+ 标签列
        test_df = pd.DataFrame(X_test, columns=feature_cols_encoded)
        test_df[label_column] = y_test
        # 测试集非空时保存
        if not test_data.empty:
            test_df.to_csv(os.path.join(save_path, "test_data.csv"), index=False, encoding="utf-8-sig")
        else:
            logger.info("测试集为空，不保存test_data.csv")

        # 5.更新全局datasets元信息
        # 记录保存路径，供训练接口读取
        datasets[dataset_id]["split_info"] = {
            "split_id": split_id,
            "save_path": save_path,
            "meta_info_path": os.path.join(save_path, "meta_info.json"),
            "processed_data_path": os.path.join(save_path, "processed_data.npz"),
            "label_mapping": label_mapping,
            "feature_encoding_mapping": feature_encoding_mapping,
            "feature_encoding": X_processed_df.columns.tolist(),
            "all_feature_cols": all_feature_cols
        }

        #使用划分后的DataFrame计算样本量
        train_size = len(train_data)
        val_size = len(val_data) if not val_data.empty else 0
        test_size = len(test_data) if not test_data.empty else 0

        # 重新计算总样本
        actual_total = train_size + val_size + test_size
        train_ratio = round(train_size / actual_total, 4) if actual_total > 0 else 0.0
        val_ratio = round(val_size / actual_total, 4) if actual_total > 0 else 0.0
        test_ratio = round(test_size / actual_total, 4) if actual_total > 0 else 0.0

        # 6.构建返回结果
        split_result = SplitDatasetResponse(
            dataset_id=dataset_id,
            processed_ID_used=processed_ID or datasets[dataset_id].get("processed_ID"),
            processed_file_path_used=processed_file_path,
            split_id=split_id,
            label_column=label_column,
            split_params=split_params.model_dump(),
            data_statistics=loader._to_python_type({
                "total_size": total_size,
                "train_set_size": len(train_data),
                "val_set_size": len(val_data),
                "test_set_size": len(test_data),
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "label_column": label_column,
                "label_colum_type": label_colum_type,
                "label_distribution": label_distribution,
                "encoded_label_mapping": label_mapping,
                "feature_encoding_mapping": feature_encoding_mapping,  # 返回特征编码映射
                "feature_encoding": X_processed_df.columns.tolist(),
                "feature_dim_after_encoding": {
                    "train": X_train.shape,
                    "val": X_val.shape,
                    "test": X_test.shape
                }
            }),
            split_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            message="数据集划分+特征编码成功，已保存预处理数据供训练使用"
        )

        # 保存划分结果到元信息
        datasets[dataset_id]["split_result"] = loader._to_python_type(split_result)
        datasets[dataset_id]["split_data"] = loader._to_python_type({
            "train": train_data.to_dict(orient="records"),
            "val": val_data.to_dict(orient="records") if not val_data.empty else [],
            "test": test_data.to_dict(orient="records") if not test_data.empty else []
        })
        datasets[dataset_id]["split_label_column"] = label_column
        datasets[dataset_id]["split_processed_ID"] = split_result.processed_ID_used

        logger.info(
            f"数据集处理完成 | 原始ID：{dataset_id} | 标签列：{label_column} | 总样本：{total_size} | "
            f"训练集：{len(train_data)}({train_ratio:.1%}) | 验证集：{len(val_data)}({val_ratio:.1%}) | 测试集：{len(test_data)}({test_ratio:.1%}) | "
            f"预处理数据保存路径：{save_path}"
        )

        return {
            "code": 200,
            "msg": "数据集划分成功",
            "data": split_result.model_dump()
        }

    except HTTPException as e:
        return {
            "code": e.status_code,
            "msg": e.detail
        }
    except Exception as e:
        logger.error(f"数据集{dataset_id}处理异常：{str(e)}", exc_info=True)
        return {
            "code": 500,
            "msg": f"服务器内部错误：{str(e)}"
        }


# ===================== 模型训练接口 =====================
import joblib
import json
import numpy as np

@app.post("/api/v1/models/{problem_type}/{model_type}/train", response_model=BaseResponse)
async def train_model(
        problem_type: str = Path(..., description="问题类型（classification/regression）"),
        model_type: str = Path(..., description="模型类型（random_forest/xgboost/lightgbm/svm等）"),
        train_request: TrainRequest = Body(..., description="训练请求参数"),
        current_user: str = Depends(get_current_user)
):
    """
    模型训练接口：支持分类和回归任务，多种算法模型
    - 支持XGBoost、LightGBM、支持向量机、随机森林、逻辑回归、KNN、决策树、朴素贝叶斯等算法
    - 算法参数支持自定义设置，不设置采用默认参数
    - 训练评价结果包括（回归任务：MAE、MSE、RMSE、R2等； 分类任务：含准确率、精确率、召回率等）
    - 问题类型（classification/regression）
    - 模型类型（random_forest/xgboost/lightgbm/svm/knn/decision_tree等）
    - 请求参数（dataset_id, split_id, model_params）
    - 训练结果（模型参数、评估指标、模型保存路径等）
    """
    # 1. 基础参数校验
    if problem_type not in ["classification", "regression"]:
        return {
            "code": 400,
            "msg": f"不支持的问题类型 {problem_type}，仅支持 classification 和 regression"
        }

    # 2. 根据问题类型选择模型映射
    model_mapping = CLASSIFICATION_MODELS if problem_type == "classification" else REGRESSION_MODELS

    # 3. 校验模型类型
    if model_type not in model_mapping:
        return {
        "code": 400,
        "msg": f"{problem_type}任务不支持{model_type}模型，支持的类型：{list(model_mapping.keys())}"
    }


    dataset_id = train_request.dataset_id
    split_id = train_request.split_id
    model_params = train_request.model_params or {}
    label_col = datasets[dataset_id]["split_label_column"]

    # 4. 校验数据集是否存在
    if dataset_id not in datasets:
        return {
            "code": 404,
            "msg": f"数据集 ID {dataset_id} 不存在"
        }

    # 5. 校验split_id是否匹配
    split_info = datasets[dataset_id].get("split_info")
    if not split_info:
        return {
            "code": 400,
            "msg": f"数据集 {dataset_id} 尚未完成划分，请先执行数据集划分接口"
        }

    if split_info.get("split_id") != split_id:
        return {
            "code": 404,
            "msg": f"数据集 {dataset_id} 不存在划分 ID {split_id}，当前有效划分 ID：{split_info.get('split_id')}"
        }

    # 6. 获取预处理文件路径并校验
    processed_data_path = split_info.get("processed_data_path")
    meta_info_path = split_info.get("meta_info_path")
    logger.info(f"元信息已保存至：{meta_info_path}")


    if not processed_data_path or not os.path.exists(processed_data_path):
        return {
            "code": 404,
            "msg": f"预处理数据文件不存在：{processed_data_path}，请重新执行数据集划分接口"
        }

    if not meta_info_path or not os.path.exists(meta_info_path):
        return {
            "code": 404,
            "msg": f"元信息文件不存在：{meta_info_path}，请重新执行数据集划分接口"
        }

    try:
        # 7. 加载预处理数据（编码后的特征+标签）
        logger.info(f"开始加载预处理数据：{processed_data_path}")
        data = np.load(processed_data_path)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        # 校验数据有效性
        if X_train.size == 0 or y_train.size == 0:
            return {
                "code": 400,
                "msg": "训练集数据为空，无法训练模型"
            }

        if len(X_train) != len(y_train):
            return {
                "code": 400,
                "msg": f"训练集特征和标签数量不匹配：特征{len(X_train)}，标签{len(y_train)}"
            }

        logger.info(f"数据加载成功 | 训练集：{X_train.shape} | 验证集：{X_val.shape} | 测试集：{X_test.shape}")

        # 8. 加载元信息（标签映射、特征信息等）
        with open(meta_info_path, "r", encoding="utf-8") as f:
            meta_info = json.load(f)
        label_mapping = meta_info.get("label_mapping", {})

        # 9. 合并模型参数（默认参数 + 自定义参数）
        final_model_params = DEFAULT_MODEL_PARAMS[problem_type][model_type].copy()
        if model_params:
            final_model_params.update(model_params)
        logger.info(f"使用模型参数：{final_model_params}")

        # 10. 初始化并训练模型
        model_class = model_mapping[model_type]
        model = model_class(**final_model_params)

        train_start_time = datetime.now()
        logger.info(f"开始训练{model_type}模型 | 数据集ID：{dataset_id} | 划分ID：{split_id}")
        model.fit(X_train, y_train)
        train_end_time = datetime.now()
        train_duration = (train_end_time - train_start_time).total_seconds()

        metrics = {}
        all_labels = sorted(np.unique(y_train))
        feature_encoding = split_info.get("feature_encoding")
        logger.info(f'编码后特征列：{feature_encoding}')

        # 11. 模型评估
        # 合并所有指标
        metrics = model_evaluator.evaluatemodel(
            model=model,
            x_train=X_train,
            y_train=y_train,
            x_val=X_val,
            y_val=y_val,
            x_test=X_test,
            y_test=y_test,
            all_labels=all_labels,
            problem_type=problem_type,
            model_type=model_type,
            feature_encoding=feature_encoding
        )
        all_metrics = metrics

        # 使用自定义编码器格式化输出,格式化混淆矩阵，可视化混淆矩阵优化
        formatted_output = f'模型评估指标：\n{json_encoder.encode(all_metrics)}'
        # 输出到日志
        logger.info(formatted_output)

        # 12. 保存模型文件
        save_path = split_info.get("save_path", "./models")
        os.makedirs(save_path, exist_ok=True)  # 确保保存目录存在
        model_filename = f"{model_type}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_save_path = os.path.join(save_path, model_filename)

        # 保存模型
        joblib.dump(model, model_save_path)
        logger.info(f"模型已保存至：{model_save_path}")


        # 13. 构建返回结果
        train_result = TrainResponse(
            dataset_id=dataset_id,
            split_id=split_id,
            problem_type=problem_type,
            model_type=model_type,
            model_params=final_model_params,
            train_time={
                "start_time": train_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": train_end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(train_duration, 2)
            },
            data_shape={
                "train": (X_train.shape[0], X_train.shape[1]),
                "val": (X_val.shape[0], X_val.shape[1]) if X_val.size > 0 else (0, 0),
                "test": (X_test.shape[0], X_test.shape[1]) if X_test.size > 0 else (0, 0)
            },
            metrics=all_metrics,
            label_mapping=label_mapping,
            model_save_path=model_save_path,
            meta_info_path=meta_info_path,
            message="模型训练成功"
        )

        # 14. 记录训练结果到全局数据集元信息
        if "train_results" not in datasets[dataset_id]:
            datasets[dataset_id]["train_results"] = []
        datasets[dataset_id]["train_results"].append({
            "model_type": model_type,
            "split_id": split_id,
            "model_params": final_model_params,
            "metrics": all_metrics,
            "model_save_path": model_save_path,
            "train_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # 15. 返回结果
        return {
            "code": 200,
            "msg": "模型训练成功",
            "data": train_result.model_dump()
        }

    except HTTPException as e:
        logger.error(f"训练失败：{str(e)}")
        return {
            "code": e.status_code,
            "msg": e.detail
        }
    except Exception as e:
        logger.error(f"训练失败：{str(e)}", exc_info=True)
        return {
            "code": 500,
            "msg": f"模型训练失败：{str(e)}"
        }


# ========== 模型预测接口(字段) ==========
from api.schemas.models import PredictRequest
@app.post("/api/v1/models/{problem_type}/{model_type}/predict", response_model=BaseResponse)
async def predict_from_json(
        problem_type: str = Path(..., description="问题类型（classification/regression）"),
        model_type: str = Path(..., description="模型类型（random_forest/xgboost/lightgbm/svm等）"),
        predict_request: PredictRequest = Body(..., description="预测请求参数"),
        current_user: str = Depends(get_current_user)
):
    """
    模型预测接口：支持分类和回归任务的模型预测
    - 加载训练好的模型和预处理器
    - 对输入数据进行预处理
    - 执行模型预测
    - 返回预测结果

     - problem_type: 问题类型（classification/regression）
     - model_type: 模型类型（random_forest/xgboost/lightgbm/svm等）
     - predict_request: 预测请求参数（model_path, meta_info_path, input_data）
     - return: 预测结果（预测值、置信度等）
    """
    # 1. 基础参数校验
    if problem_type not in ["classification", "regression"]:
        return {
            "code": 400,
            "msg": f"不支持的问题类型 {problem_type}，仅支持 classification 和 regression"
        }

    try:
        # 2. 提取预测参数
        input_data = predict_request.input_data
        model_path = predict_request.model_path.replace('"', '').replace("\\", "/")
        meta_info_path =  predict_request.meta_info_path.replace('"', '').replace("\\", "/")



        # 3. 校验文件路径
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"模型文件不存在：{model_path}")

        if not os.path.exists(meta_info_path):
            raise HTTPException(status_code=404, detail=f"元信息文件不存在（特征编码器、标签映射）：{meta_info_path}")

        # 4.加载元信息文件
        meta_info = None
        if meta_info_path and os.path.exists(meta_info_path):
            with open(meta_info_path, "r", encoding="utf-8") as f:
                meta_info = json.load(f)

        preprocessor_path = meta_info.get("preprocessor_path")

        # 5. 加载模型和预处理器
        logger.info(f"开始加载模型：{model_path}")
        model = joblib.load(model_path)

        logger.info(f"开始加载预处理器：{preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)

        # 6. 处理输入数据
        # 将输入数据转换为DataFrame
        input_df = pd.DataFrame(input_data)

        # 7. 执行预处理
        logger.info(f"开始预处理输入数据")
        input_processed = preprocessor.transform(input_df)

        # 8. 执行预测
        logger.info(f"开始执行预测")
        predict_start_time = datetime.now()

        if hasattr(model, 'predict_proba') and problem_type == "classification":
            # 分类任务，获取预测概率和预测标签
            y_pred_proba = model.predict_proba(input_processed)
            y_pred = model.predict(input_processed)
        else:
            # 回归任务或不支持概率预测的分类模型，仅获取预测值
            y_pred = model.predict(input_processed)
            y_pred_proba = None

        predict_end_time = datetime.now()
        predict_duration = (predict_end_time - predict_start_time).total_seconds()


        # 9. 处理预测结果
        result = {
            "predictions": y_pred.tolist(),
            "predict_duration_seconds": round(predict_duration, 4),
            "model_type": model_type,
            "problem_type": problem_type
        }

        # 添加概率信息
        if y_pred_proba is not None:
            result["probabilities"] = y_pred_proba.tolist()

        # 添加标签映射
        if meta_info:
            result["label_mapping"] = meta_info["label_mapping"]
            label_mapping = meta_info["label_mapping"]
            logger.info(f"标签映射：{label_mapping}")
            # 将预测结果转换为原始标签
            try:
                # 检查label_mapping的结构，处理不同格式
                if isinstance(label_mapping, dict):
                    # 检查是否是嵌套字典
                    has_dict_values = any(isinstance(v, dict) for v in label_mapping.values())

                    if has_dict_values:
                        # 处理嵌套字典情况，例如：{"label1": {"code": 0}, "label2": {"code": 1}}
                        logger.warning(f"标签映射是嵌套字典，无法直接创建反向映射: {label_mapping}")
                        # 可以根据实际情况提取嵌套值，这里暂时跳过
                        result["predictions_original"] = y_pred.tolist()
                    else:
                        # 正常情况，创建反向映射
                        reverse_label_mapping = {v: k for k, v in label_mapping.items()}
                        result["predictions_original"] = [reverse_label_mapping.get(pred, pred) for pred in
                                                          y_pred.tolist()]
                        label_reverse = result["predictions_original"]
                        logger.info(f"预测结果: {label_reverse}")
                else:
                    # 非字典类型，跳过反向映射
                    logger.warning(f"标签映射不是字典类型: {type(label_mapping)}")
                    result["predictions_original"] = y_pred.tolist()
                    logger.info(f"预测结果: {y_pred.tolist()}")
            except TypeError as e:
                # 捕获类型错误，确保预测流程不中断
                logger.error(f"创建反向标签映射失败: {str(e)}")
                logger.error(f"标签映射结构: {label_mapping}")
                result["predictions_original"] = y_pred.tolist()

        logger.info(f"预测完成，共处理 {len(input_data)} 条数据，耗时 {predict_duration:.4f} 秒")

        # 10. 构造响应
        return {
            "code": 200,
            "msg": "预测成功",
            "data": result
        }

    except HTTPException as e:
        logger.error(f"预测失败：{e.detail}")
        return {
            "code": e.status_code,
            "msg": e.detail
        }
    except Exception as e:
        logger.error(f"预测失败：{str(e)}", exc_info=True)
        return {
            "code": 500,
            "msg": f"预测失败：{str(e)}"
        }




# ========== 模型预测接口(文件) ==========
@app.post("/api/v1/models/{problem_type}/{model_type}/file_predict")
async def predict_from_file(
        problem_type: str = Path(..., description="问题类型（classification/regression）"),
        model_type: str = Path(..., description="模型类型（random_forest/xgboost/lightgbm/svm等）"),
        split_id: str = Form(..., description="划分数据集ID"),
        model_path: str = Form(..., description="模型文件路径"),
        meta_info_path: str = Form(..., description="元信息文件路径"),
        file: UploadFile = File(..., description="预测文件：csv / xlsx / xls"),
        current_user: str = Depends(get_current_user)
):
    if problem_type not in ["classification", "regression"]:
        return {
            "code": 400,
            "msg": f"不支持的问题类型 {problem_type}，仅支持 classification 和 regression"
        }

    try:
        # 1. 读取上传的文件
        filename = file.filename
        if filename.endswith(".csv"):
            input_df = pd.read_csv(file.file)
        elif filename.endswith((".xlsx", ".xls")):
            input_df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="仅支持 csv / xlsx / xls 格式")

        model_path = model_path.replace('"', '').replace("\\", "/")
        meta_info_path = meta_info_path.replace('"', '').replace("\\", "/")

        # 2. 校验路径
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"模型文件不存在：{model_path}")
        if not os.path.exists(meta_info_path):
            raise HTTPException(status_code=404, detail=f"元信息文件不存在：{meta_info_path}")

        # 3. 加载模型 预处理
        with open(meta_info_path, "r", encoding="utf-8") as f:
            meta_info = json.load(f)
        preprocessor_path = meta_info.get("preprocessor_path")
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        input_processed = preprocessor.transform(input_df)

        # 4. 预测
        start = datetime.now()
        if hasattr(model, "predict_proba") and problem_type == "classification":
            logger.info(f"批量分类预测")
            y_pred = model.predict(input_processed)
            y_pred_proba = model.predict_proba(input_processed)
        else:
            logger.info(f"批量回归预测")
            y_pred = model.predict(input_processed)
            y_pred_proba = None
        duration = round((datetime.now() - start).total_seconds(), 4)

        # 5. 构造结果
        if hasattr(model, "predict_proba") and problem_type == "classification":
            output_df = input_df.copy()
            output_df["predict_result"] = y_pred.tolist()
            output_df["predict_proba"] = y_pred_proba.tolist()
        else:
            output_df = input_df.copy()
            output_df["predict_result"] = y_pred.tolist()


        #标签映射
        result = {}
        if meta_info:
            result["label_mapping"] = meta_info["label_mapping"]
            label_mapping = meta_info["label_mapping"]
            logger.info(f"标签映射：{label_mapping}")
            # 将预测结果转换为原始标签
            try:
                # 检查label_mapping的结构，处理不同格式
                if isinstance(label_mapping, dict):
                    # 检查是否是嵌套字典
                    has_dict_values = any(isinstance(v, dict) for v in label_mapping.values())

                    if has_dict_values:
                        # 处理嵌套字典情况，例如：{"label1": {"code": 0}, "label2": {"code": 1}}
                        logger.warning(f"标签映射是嵌套字典，无法直接创建反向映射: {label_mapping}")
                    else:
                        # 正常情况，创建反向映射
                        reverse_label_mapping = {v: k for k, v in label_mapping.items()}
                        output_df["predict_result"] = [reverse_label_mapping.get(pred, pred) for pred in
                                                          y_pred.tolist()]
                        logger.info(f"预测结果映射已写入")
                else:
                    # 非字典类型，跳过反向映射
                    logger.warning(f"标签映射不是字典类型: {type(label_mapping)}")
            except TypeError as e:
                # 捕获类型错误，确保预测流程不中断
                logger.error(f"创建反向标签映射失败: {str(e)}")
                logger.error(f"标签映射结构: {label_mapping}")

        # 6. 保存结果
        save_path = os.path.join(SAVE_DIR, split_id)
        os.makedirs(save_path, exist_ok=True)
        predict_result_path = os.path.join(save_path, f"{model_type}_predict_result.csv")
        output_df.to_csv(predict_result_path, index=False, encoding="utf-8-sig")
        logger.info(f"批量预测保存至{predict_result_path}")

        # 7. 返回
        result = {
            "predict_duration_seconds": duration,
            "split_id": split_id,
            "model_type": model_type,
            "problem_type": problem_type,
            "predict_result_path": predict_result_path,
            "label_mapping": label_mapping,

        }
        logger.info(f"预测完成，共处理 {len(output_df)} 条数据，耗时 {duration:.4f} 秒")
        return {
            "code": 200,
            "msg": "预测成功",
            "data": result
        }


    except HTTPException as e:
        return {"code": e.status_code, "msg": e.detail}
    except Exception as e:
        logger.error(f"错误：{str(e)}", exc_info=True)
        return {"code": 500, "msg": f"预测失败：{str(e)}"}

# ===================== 主函数 =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=ORIGIN, port=API_PORT)
