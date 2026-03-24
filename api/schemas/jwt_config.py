import os
# 读取环境变量，无则用默认值
UPLOAD_DIR = os.getenv("ML_PLATFORM_UPLOAD_DIR", "./uploads")
PROCESSED_DIR = os.path.join(UPLOAD_DIR, "processed")
SAVE_DIR = os.path.join(UPLOAD_DIR, "split_datasets")
LOG_DIR = os.getenv("ML_PLATFORM_LOG_DIR", "./logs")
API_PORT = int(os.getenv("ML_PLATFORM_PORT", 8006))
MAX_FILE_SIZE = int(os.getenv("ML_PLATFORM_MAX_FILE_SIZE", 100 * 1024 * 1024))  # 100MB
RATE_LIMIT = os.getenv("ML_PLATFORM_RATE_LIMIT", "100/minute")
ORIGIN =os.getenv("ML_PLATFORM_RATE_ORIGIN", "0.0.0.0")