from retriever import hybrid_retrieve
from try_reraken import rerank
from qwen_LLM import generate

def start_rag(query):
    # 1. 检索
    print("--> 正在检索知识库...")
    candidates = hybrid_retrieve(query, top_k=10)

    if not candidates:
        return "知识库中未找到相关内容。"

    # 2. 精排
    print("--> 正在进行深度排序...")
    docs_texts = [doc["content"] for doc in candidates]
    rerank_results = rerank(query, docs_texts, top_n=3)

    top_chunks = []
    for r in rerank_results:
        top_chunks.append(candidates[r["index"]])

    # 3. 生成
    print("--> 正在呼叫通义千问大脑...")
    final_answer = generate(query, top_chunks)

    return final_answer

if __name__ == "__main__":
    answer = start_rag("Spring Boot 自动装配原理是什么？")
    print("\n最终答案：\n", answer)