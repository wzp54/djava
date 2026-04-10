from pymilvus import connections, Collection
from app.config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME
from app.embedding import get_embedding

_conn = None


def _get_collection():
    """获取 Milvus collection（单例连接）"""
    global _conn
    if _conn is None:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        _conn = True
    collection = Collection(COLLECTION_NAME)
    collection.load()
    return collection


def _vector_search(query_vector, top_k=10):
    """向量检索"""
    collection = _get_collection()

    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["id", "content", "file_path", "parent_content"]
    )

    docs = []
    for hit in results[0]:
        docs.append({
            "id": hit.entity.get("id"),
            "content": hit.entity.get("content"),
            "file_path": hit.entity.get("file_path"),
            "parent_content": hit.entity.get("parent_content"),
            "score": hit.score
        })
    return docs


def hybrid_retrieve(query, top_k=10):
    """
    混合检索（目前纯向量）
    返回: [{"id", "content", "file_path", "parent_content", "score"}, ...]
    """
    query_vec = get_embedding(query)
    docs = _vector_search(query_vec, top_k)
    return docs