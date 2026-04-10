"""
统一日志配置模块
"""
import logging
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    创建并配置 logger

    Args:
        name: logger 名称，通常使用 __name__
        level: 日志级别，默认 INFO

    Returns:
        配置好的 logger 实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # 格式化器
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


# 创建项目根 logger
app_logger = setup_logger("app")

# 创建各模块 logger
config_logger = setup_logger("app.config")
retriever_logger = setup_logger("app.retriever")
rag_logger = setup_logger("app.rag_engine")
llm_logger = setup_logger("app.llm")
embedding_logger = setup_logger("app.embedding")
reranker_logger = setup_logger("app.reranker")
redis_logger = setup_logger("app.redis")
milvus_logger = setup_logger("app.milvus")
es_logger = setup_logger("app.es")
