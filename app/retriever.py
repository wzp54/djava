from app.config import get_es_client
from app.embedding import get_embedding
from app.logger import retriever_logger

def _keyword_search(query_text, top_k=5):
    retriever_logger.debug(f"关键词检索: query={query_text}, top_k={top_k}")
    es = get_es_client()
    response = es.search(
        index="javaguide_children",
        body={
            "query": {"match": {"content": query_text}},
            "size": top_k,
            "_source": ["id", "content", "file_path"]
        }
    )
    hits = [hit["_source"] for hit in response["hits"]["hits"]]
    retriever_logger.info(f"关键词检索完成，返回 {len(hits)} 条结果")
    return hits

def _vector_search(query_vector, top_k=5):
    retriever_logger.debug(f"向量检索: top_k={top_k}")
    es = get_es_client()
    response = es.search(
        index="javaguide_children",
        body={
            "knn": {
                "field": "vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 50
            },
            "_source": ["id", "content", "file_path"]
        }
    )
    hits = [hit["_source"] for hit in response["hits"]["hits"]]
    retriever_logger.info(f"向量检索完成，返回 {len(hits)} 条结果")
    return hits

def hybrid_retrieve(query, top_k=5):
    retriever_logger.info(f"混合检索开始: query={query}")
    # 文本词检索
    kw_docs = _keyword_search(query, top_k)
    # 先转词向量再词向量检索
    query_vec = get_embedding(query)
    vec_docs = _vector_search(query_vec, top_k)
    # 去重合并
    unique_docs = {}
    for doc in kw_docs + vec_docs:
        if doc["content"] not in unique_docs:
            unique_docs[doc["content"]] = doc
    merge_list = list(unique_docs.values())
    retriever_logger.info(f"混合检索完成，返回 {len(merge_list)} 条去重结果")
    return merge_list