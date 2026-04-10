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


RERANKER_URL = _get_non_empty_env(
    "RERANKER_API_URL",
    default=(
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/"
        "text-rerank/text-rerank"
    ),
).rstrip("/")
RERANKER_MODEL = _get_non_empty_env("RERANKER_MODEL", default="gte-rerank-v2")
RERANKER_API_KEY = _get_non_empty_env(
    "RERANKER_API_KEY", "LLM_API_KEY", "DASHSCOPE_API_KEY", default=""
)
RERANKER_HEADERS = {"Content-Type": "application/json"}
if RERANKER_API_KEY:
    RERANKER_HEADERS["Authorization"] = f"Bearer {RERANKER_API_KEY}"


def rerank(query, documents, top_n=3):
    response = requests.post(
        RERANKER_URL,
        headers=RERANKER_HEADERS,
        json={
            "model": RERANKER_MODEL,
            "input": {
                "query": query,
                "documents": documents,
            },
        },
    )
    response.raise_for_status()
    raw_results = response.json()["output"]["results"]
    normalized_results = [
        {
            "index": item["index"],
            "score": item.get("score", item.get("relevance_score", 0)),
            "relevance_score": item.get("relevance_score", item.get("score", 0)),
        }
        for item in raw_results
    ]
    normalized_results.sort(key=lambda item: item["relevance_score"], reverse=True)
    return normalized_results[:top_n]
