import requests

from app.logger import embedding_logger
from app.model_api import (
    build_bearer_headers,
    resolve_embedding_api_key,
    resolve_embedding_model,
    resolve_embedding_url,
)

EMBEDDING_URL = resolve_embedding_url()
EMBEDDING_MODEL = resolve_embedding_model("text-embedding-v4")
EMBEDDING_HEADERS = build_bearer_headers(resolve_embedding_api_key())


def _request_embeddings(texts):
    response = requests.post(
        EMBEDDING_URL,
        headers=EMBEDDING_HEADERS,
        json={
            "model": EMBEDDING_MODEL,
            "input": {"texts": texts},
            "parameters": {},
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["output"]["embeddings"]]


def get_embedding(text):
    embedding_logger.debug(f"获取单个文本 embedding: {text[:50]}...")
    try:
        result = _request_embeddings([text])[0]
        embedding_logger.info(f"成功获取 embedding，向量维度: {len(result)}")
        return result
    except Exception as exc:
        embedding_logger.error(f"获取 embedding 失败: {exc}")
        raise


def get_embeddings_batch(texts):
    embedding_logger.info(f"批量获取 embedding，文本数量: {len(texts)}")
    try:
        result = _request_embeddings(texts)
        embedding_logger.info(f"批量 embedding 成功，返回 {len(result)} 个向量")
        return result
    except Exception as exc:
        embedding_logger.error(f"批量获取 embedding 失败: {exc}")
        raise
