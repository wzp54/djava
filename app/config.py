import os
from urllib.parse import urlparse

from elasticsearch import Elasticsearch

from app.logger import config_logger

ES_HOST_RAW = os.getenv("ES_HOST", "115.191.31.132").rstrip("/")
ES_PORT = os.getenv("ES_PORT", "9200")
ES_SCHEME = os.getenv("ES_SCHEME", "")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "3531014")

if "://" in ES_HOST_RAW:
    parsed = urlparse(ES_HOST_RAW)
    ES_SCHEME = parsed.scheme or ES_SCHEME
    ES_HOST = parsed.hostname or "115.191.31.132"
    if parsed.port:
        ES_PORT = str(parsed.port)
else:
    ES_HOST = ES_HOST_RAW

if not ES_SCHEME:
    ES_SCHEME = "https"

MILVUS_HOST = os.getenv("MILVUS_HOST", "115.191.31.132")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "javaguide_chunks")

config_logger.info(f"初始化 Elasticsearch: {ES_SCHEME}://{ES_HOST}:{ES_PORT}")

es = Elasticsearch(
    hosts=[f"{ES_SCHEME}://{ES_HOST}:{ES_PORT}"],
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=False,
    ssl_show_warn=False,
)


def get_es_client():
    return es
