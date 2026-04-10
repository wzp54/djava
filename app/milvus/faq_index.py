from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from uuid import uuid1
from app.config import MILVUS_HOST, MILVUS_PORT
from app.embedding import get_embedding
from app.logger import milvus_logger

FAQ_COLLECTION = "javaguide_faq"

def _connect():
    milvus_logger.debug(f"连接 Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

def init_faq_collection():
    """建表，只需执行一次"""
    milvus_logger.info("初始化 FAQ 集合")
    _connect()
    if utility.has_collection(FAQ_COLLECTION):
        milvus_logger.info("FAQ 集合已存在")
        return

    fields = [
        FieldSchema(name="qid", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="query", dtype=DataType.VARCHAR, max_length=8000),
        FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=32000),
        FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=1024),
    ]
    schema = CollectionSchema(fields)
    collection = Collection(FAQ_COLLECTION, schema)
    collection.create_index("vec", {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 256}
    })
    milvus_logger.info("FAQ 集合创建完成")

def load_faq(faq_data):
    """
    导入 FAQ 数据
    faq_data: [{"query": "...", "answer": "..."}, ...]
    """
    milvus_logger.info(f"开始导入 FAQ 数据，共 {len(faq_data)} 条")
    _connect()
    collection = Collection(FAQ_COLLECTION)

    batch_qids = []
    batch_queries = []
    batch_answers = []
    batch_vecs = []

    for i, item in enumerate(faq_data):
        vec = get_embedding(item["query"])
        batch_qids.append(str(uuid1()))
        batch_queries.append(item["query"][:7900])
        batch_answers.append(item["answer"][:31900])
        batch_vecs.append(vec)

        if len(batch_qids) >= 32:
            collection.insert([batch_qids, batch_queries, batch_answers, batch_vecs])
            milvus_logger.info(f"已导入 {i + 1} 条")
            batch_qids, batch_queries, batch_answers, batch_vecs = [], [], [], []

    if batch_qids:
        collection.insert([batch_qids, batch_queries, batch_answers, batch_vecs])

    collection.flush()
    milvus_logger.info(f"FAQ 导入完成，共 {len(faq_data)} 条")

def search_faq(query, top_k=1):
    """FAQ 检索，返回最匹配的问答对和分数"""
    milvus_logger.debug(f"FAQ 检索: query={query}, top_k={top_k}")
    _connect()
    collection = Collection(FAQ_COLLECTION)
    collection.load()

    query_vec = get_embedding(query)
    results = collection.search(
        data=[query_vec],
        anns_field="vec",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["query", "answer"]
    )

    hits = []
    for hit in results[0]:
        hits.append({
            "query": hit.entity.get("query"),
            "answer": hit.entity.get("answer"),
            "score": hit.score
        })
    milvus_logger.info(f"FAQ 检索完成，返回 {len(hits)} 条结果")
    return hits

if __name__ == "__main__":
    import json

    # 第一步：建表
    init_faq_collection()

    # 第二步：导入数据
    with open("faq_data.json", "r", encoding="utf-8") as f:
        faq_data = json.load(f)
    load_faq(faq_data)