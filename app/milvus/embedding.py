import os

import requests


def _get_non_empty_env(*names, default=""):
    for name in names:
        value = os.getenv(name)
        if value is not None:
            value = value.strip()
            if value:
                return value
    return default


EMBEDDING_URL = _get_non_empty_env(
    "EMBEDDING_API_URL",
    default=(
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
        "text-embedding/text-embedding"
    ),
).rstrip("/")
EMBEDDING_MODEL = _get_non_empty_env("EMBEDDING_MODEL", default="text-embedding-v4")
EMBEDDING_API_KEY = _get_non_empty_env(
    "EMBEDDING_API_KEY", "LLM_API_KEY", "DASHSCOPE_API_KEY", default=""
)
EMBEDDING_HEADERS = {"Content-Type": "application/json"}
if EMBEDDING_API_KEY:
    EMBEDDING_HEADERS["Authorization"] = f"Bearer {EMBEDDING_API_KEY}"


def _request_embeddings(texts):
    response = requests.post(
        EMBEDDING_URL,
        headers=EMBEDDING_HEADERS,
        json={
            "model": EMBEDDING_MODEL,
            "input": {"texts": texts},
            "parameters": {},
        },
    )
    response.raise_for_status()
    return [item["embedding"] for item in response.json()["output"]["embeddings"]]


def get_embedding(text):
    return _request_embeddings([text])[0]


def get_embeddings(texts):
    return _request_embeddings(texts)
