from pymilvus import connections, Collection
from app.embedding import get_embeddings

# 连接配置
MILVUS_HOST = "115.191.31.132"
MILVUS_PORT = 19530
COLLECTION_NAME = "javaguide_chunks"


def get_collection():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    return Collection(COLLECTION_NAME)


def bulk_index_chunks(parents, children, batch_size=32):
    """
    将父文档和子文档写入 Milvus
    parents: [{"id": ..., "file_path": ..., "title": ..., "content": ...}, ...]
    children: [{"id": ..., "parent_id": ..., "file_path": ..., "content": ...}, ...]
    """
    collection = get_collection()

    # 建立 parent_id -> parent_content 映射
    parent_map = {p["id"]: p["content"] for p in parents}

    # 分批处理
    total = len(children)
    for i in range(0, total, batch_size):
        batch = children[i:i + batch_size]

        # 准备数据
        ids = []
        parent_ids = []
        file_paths = []
        contents = []
        parent_contents = []
        texts_for_embedding = []

        for child in batch:
            ids.append(child["id"])
            parent_ids.append(child["parent_id"])
            file_paths.append(child["file_path"])
            contents.append(child["content"][:7900])  # 截断防超长

            # 获取父文档内容
            pc = parent_map.get(child["parent_id"], "")
            parent_contents.append(pc[:31900])

            texts_for_embedding.append(child["content"])

        # 批量获取向量
        vectors = get_embeddings(texts_for_embedding)

        # 写入 Milvus
        collection.insert([
            ids,
            parent_ids,
            file_paths,
            contents,
            parent_contents,
            vectors
        ])

        print(f"已写入 {min(i + batch_size, total)}/{total}")

    collection.flush()
    print(f"✅ 全部写入完成，共 {total} 条")