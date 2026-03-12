import os
import sys
import logging
from datetime import datetime

def setup_logger(logger_name: str = "ml_platform_api", log_dir: str = "./logs") -> logging.Logger:
    """
    初始化日志器，确保处理器不重复添加，禁用日志向上传播
    :param logger_name: 日志器名称
    :param log_dir: 日志文件存储目录
    :return: 配置好的日志器
    """
    # 1. 获取日志器
    logger = logging.getLogger(logger_name)

    # 2. 禁用日志向上传播到根日志器
    logger.propagate = False

    # 3. 检查是否已配置过处理器，避免重复添加
    if logger.handlers:
        return logger  # 已配置过，直接返回

    # 4. 确保日志目录存在，避免文件写入失败
    os.makedirs(log_dir, exist_ok=True)

    # 5. 配置文件处理器
    log_filename = os.path.join(log_dir, f"ml_platform_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(
        filename=log_filename,
        mode='a',
        encoding='utf-8',  # 明确指定文件编码
        errors='replace'  # 编码错误时替换而非报错
    )

    # 6. 配置控制台处理器
    stream_handler = logging.StreamHandler(sys.stdout)

    # 7. 统一日志格式
    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(log_formatter)
    stream_handler.setFormatter(log_formatter)

    # 8. 设置日志级别并添加处理器
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

# 初始化全局日志器
logger = setup_logger("ml_platform_api", log_dir="./logs")

# 测试代码
if __name__ == "__main__":
    logger.info("日志系统初始化完成")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")