import requests

from app.logger import reranker_logger
from app.model_api import (
    build_bearer_headers,
    resolve_reranker_api_key,
    resolve_reranker_model,
    resolve_reranker_url,
)

RERANKER_URL = resolve_reranker_url()
RERANKER_MODEL = resolve_reranker_model("gte-rerank-v2")
RERANKER_HEADERS = build_bearer_headers(resolve_reranker_api_key())


def rerank(query, documents, top_n=3):
    reranker_logger.info(
        f"Rerank 请求: query={query}, documents数量={len(documents)}, top_n={top_n}"
    )
    try:
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
            timeout=60,
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
        results = normalized_results[:top_n]
        reranker_logger.info(f"Rerank 完成，返回 {len(results)} 条结果")
        return results
    except Exception as exc:
        reranker_logger.error(f"Rerank 请求异常: {exc}")
        raise
