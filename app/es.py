from elasticsearch import Elasticsearch
import json
es = Elasticsearch(
    hosts=["https://115.191.31.132:9200"],
    basic_auth=("elastic", "3531014"),
    verify_certs=False,
    ssl_show_warn=False
)

# 直接拿 ID 为 1 的数据
# 数据可以是任意的 Python 字典
# doc = {
#     "title": "JavaGuide 面试突击版",
#     "content": "HashMap 的底层原理是数组+链表...",
#     "tags": ["Java", "集合"],
#     "views": 1024
# }
#
# # 动作：往 'my_knowledge_base' 索引里存数据
# # id="1" 是可选的。如果你不写，ES 会自动生成一个乱码 ID (如 xU7s9...)
# # 在 RAG 中，建议用 MD5 作为 ID，防止重复
# resp = es.index(index="my_knowledge_base", id="1", document=doc)
#
# print(f"结果: {resp['result']}") # 输出: created 或 updated
# 直接拿 ID 为 1 的数据
def print_separator(title):
    print(f"\n{'=' * 20} {title} {'=' * 20}")


# === 第一步：看有哪些索引 (相当于 SQL 的 SHOW TABLES) ===
print_separator("现有索引列表")
# cat API 是专门给人看的，v=True 显示表头
indices = es.cat.indices(v=True, format="json")

for idx in indices:
    # 打印 索引名、文档数量、占用空间
    print(f"[{idx['index']}] \t文档数: {idx['docs.count']} \t占用: {idx['store.size']}")

# === 第二步：看索引里的数据长什么样 (相当于 SQL 的 SELECT * LIMIT 1) ===
target_index = "javaguide_children"  # 你想看哪个索引，就填哪个

print_separator(f"索引 [{target_index}] 的第一条数据")

try:
    # size=1 表示只取一条，explain=False 不显示打分过程
    resp = es.search(index=target_index, size=1)

    if resp['hits']['hits']:
        first_doc = resp['hits']['hits'][0]
        print("【元数据 (Metadata)】:")
        print(f"ID: {first_doc['_id']}")
        print(f"Score: {first_doc['_score']}")

        print("\n【真实数据 (_source)】:")
        # json.dumps 让输出更漂亮
        print(json.dumps(first_doc['_source'], indent=2, ensure_ascii=False))
    else:
        print("空空如也，里面没数据。")
except Exception as e:
    print(f"查询失败: {e}")