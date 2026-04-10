from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# 连接 Milvus
connections.connect(host="115.191.31.132", port=19530)

collection_name = "javaguide_chunks"

# 删除旧的（如果存在）
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"已删除旧 Collection: {collection_name}")

# 定义字段
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
    FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8000),
    FieldSchema(name="parent_content", dtype=DataType.VARCHAR, max_length=32000),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
]

# 创建 Collection
schema = CollectionSchema(fields, description="JavaGuide RAG")
collection = Collection(collection_name, schema)

# 创建向量索引
collection.create_index(
    field_name="vector",
    index_params={
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 256}
    }
)

print(f"✅ Collection '{collection_name}' 创建成功!")
print(f"字段: {[f.name for f in fields]}")