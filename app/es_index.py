from app.config import es

parents_mapping = {
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "smartcn"},
            "content": {"type": "text", "index": False},
            "source_file": {"type": "keyword"}
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    }
}

children_mapping = {
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "content": {"type": "text", "analyzer": "smartcn"},
            "parent_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "file_path": {"type": "keyword"},
            "vector": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine"
            }
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    }
}


def create_indices():
    # 删除旧索引
    if es.indices.exists(index="javaguide_parents"):
        es.indices.delete(index="javaguide_parents")
    if es.indices.exists(index="javaguide_children"):
        es.indices.delete(index="javaguide_children")

    # 创建新索引
    es.indices.create(index="javaguide_parents", body=parents_mapping)
    es.indices.create(index="javaguide_children", body=children_mapping)
    print("索引创建成功！")


if __name__ == "__main__":
    create_indices()