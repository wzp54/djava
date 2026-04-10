# cache.py
import os
import redis
from app.logger import redis_logger

# 从环境变量读取 Redis 配置
REDIS_HOST = os.getenv("REDIS_HOST", "115.191.31.132")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "123456")

redis_logger.info(f"初始化 Redis 客户端: {REDIS_HOST}:{REDIS_PORT}")

# 1. 建立基础的 Redis 连接
redis_conn = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True
)


def get_cache(query):
    """从 Redis 直接获取缓存的回答"""
    # 给 Key 加一个前缀，方便在 AnotherRedisDesktopManager 里分类查看
    cache_key = f"rag_qa:{query}"

    # 直接根据 Key 拿 Value
    try:
        result = redis_conn.get(cache_key)
        if result:
            redis_logger.debug(f"缓存命中: {query[:30]}...")
        else:
            redis_logger.debug(f"缓存未命中: {query[:30]}...")
        return result
    except Exception as e:
        redis_logger.error(f"Redis 获取缓存失败: {e}")
        return None


def set_cache(query, answer):
    """将新的问答对存入 Redis"""
    cache_key = f"rag_qa:{query}"

    try:
        # 直接以字符串形式存储，还可以顺手加个过期时间（比如 7 天）
        # ex=604800 表示 7 天后这条缓存自动清理，防止垃圾数据堆积
        redis_conn.set(cache_key, answer, ex=604800)
        redis_logger.info(f"缓存写入成功: {query[:30]}..., 答案长度: {len(answer)}")
    except Exception as e:
        redis_logger.error(f"Redis 设置缓存失败: {e}")