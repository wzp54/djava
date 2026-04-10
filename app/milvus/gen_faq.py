import json
from pymilvus import connections, Collection
from app.config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME
from qwen_LLM import generate_raw  # 需要你加一个不带 RAG 的纯生成函数

SYSTEM_PROMPT = '我们在做一个 Java 技术知识库助手。请根据以下技术文档内容，输出开发者可能问的问题，并给出答案。格式为json:[{"query":"","answer":""}]，不要给多余的内容，不要解释'

def gen_faq():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(COLLECTION_NAME)
    collection.load()

    # 遍历所有文档内容
    results = collection.query(expr='id != ""', output_fields=["content"], limit=16384)

    all_qas = []
    for item in results:
        content = item["content"]
        if len(content) < 50:  # 太短的跳过
            continue
        try:
            prompt = f"以下为技术文档内容：{content}"
            # 这里需要你的千问直接生成，不走 RAG
            response = generate_raw(SYSTEM_PROMPT, prompt)
            qas = json.loads(response)
            all_qas.extend(qas)
            print(f"生成 {len(qas)} 条 FAQ")
        except Exception as e:
            print(f"跳过: {e}")
            continue

    with open("faq_data.json", "w", encoding="utf-8") as f:
        json.dump(all_qas, f, ensure_ascii=False, indent=2)
    print(f"共生成 {len(all_qas)} 条 FAQ，已保存到 faq_data.json")

if __name__ == "__main__":
    gen_faq()