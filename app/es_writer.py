from elasticsearch import helpers
from app.embedding import get_embeddings_batch
from app.config import get_es_client

def bulk_index_parents(parents):
    """批量写入父文档"""
    es = get_es_client()

    actions = [
        {
            "_index": "javaguide_parents",
            "_id": p["id"],
            "_source": {
                "source_file": p["source_file"],  # 改这里
                "title": p["title"],
                "content": p["content"]
            }
        }
        for p in parents
    ]

    helpers.bulk(es, actions)
    print(f"父文档写入完成: {len(parents)} 条")

def bulk_index_children(children, batch_size=32):
    """批量向量化子文档并写入 ES"""
    es = get_es_client()

    total = len(children)
    success_count = 0

    for i in range(0, total, batch_size):
        batch = children[i:i + batch_size]

        # 批量获取向量
        texts = [c["content"] for c in batch]
        vectors = get_embeddings_batch(texts)

        # 构建 bulk 请求
        actions = []
        for j, child in enumerate(batch):
            actions.append({
                "_index": "javaguide_children",
                "_id": child["id"],  # 改这里，用 id 不是 child_id
                "_source": {
                    "parent_id": child["parent_id"],
                    "content": child["content"],
                    "chunk_index": child["chunk_index"],
                    "file_path": child["file_path"],
                    "vector": vectors[j]
                }
            })

        helpers.bulk(es, actions)
        success_count += len(batch)
        print(f"进度: {success_count}/{total}")

    return success_count