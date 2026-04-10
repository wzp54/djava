# rag_engine.py
from app.retriever import hybrid_retrieve
from app.try_reraken import rerank
from app.qwen_sglang import generate
from app.my_redis.cache import get_cache, set_cache
from app.milvus.faq_index import search_faq
from app.qwen_sglang_stream import generate_stream
from app.logger import rag_logger

FAQ_THRESHOLD = 0.82

# ==========================================
# 🌟 RAG 高级 Query 改写 Prompt
# ==========================================
STRATEGY_PROMPT = """你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。
1. 直接检索：适用于查询意图明确，需要从知识库中检索特定信息的问题。
2. 假设问题检索：适用于查询较为抽象，直接检索效果不佳的问题。
3. 子查询检索：将复杂的用户查询拆分为多个简单的子查询，最多生成3个。适用于涉及多个实体的问题。
4. 回溯问题检索：将复杂查询转化为更基础、易于检索的问题。
根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。"""

HYDE_PROMPT = """假设你是用户，想了解以下问题，请生成一个简短的假设答案。只输出假设答案即可。
问题: {query}
假设答案:"""

SUBQUERY_PROMPT = """将以下复杂查询分解为多个简单子查询，最多生成3个子查询。只输出子查询的问题，每行一个。
问题: {query}
子查询:"""

BACKTRACKING_PROMPT = """将以下复杂查询简化为一个更简单的问题，只输出简化问题即可。
问题: {query}
简化问题:"""


def adaptive_query_rewrite(query, history):
    """
    🌟 自适应 Query 改写：根据问题复杂度自动选择 HyDE、子查询或回溯改写
    返回一个 list，里面包含一个或多个改写后的 query
    """
    try:
        # 1. 大模型自主选择策略
        strategy = generate(STRATEGY_PROMPT.format(query=query), chunks=[], history=history).strip()

        # 2. 根据策略进行改写裂变
        if "假设问题" in strategy:
            queries = [generate(HYDE_PROMPT.format(query=query), chunks=[], history=history).strip()]
        elif "子查询" in strategy:
            sub_res = generate(SUBQUERY_PROMPT.format(query=query), chunks=[], history=history)
            queries = [q.strip() for q in sub_res.split("\n") if q.strip()][:3]  # 最多取3个子问题
        elif "回溯问题" in strategy:
            queries = [generate(BACKTRACKING_PROMPT.format(query=query), chunks=[], history=history).strip()]
        else:
            queries = [query]  # 默认直接检索

        rag_logger.info(f"原始问题: '{query}'")
        rag_logger.info(f"命中策略: '{strategy}'")
        rag_logger.info(f"改写结果: {queries}")

        return queries

    except Exception as e:
        rag_logger.warning(f"策略改写失败，降级为原问题: {e}")
        return [query]


def reorder_lost_in_the_middle(chunks):
    """解决 Long Context 下的 Lost in the middle 问题"""
    if not chunks or len(chunks) < 3:
        return chunks
    reordered = [None] * len(chunks)
    left, right = 0, len(chunks) - 1
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            reordered[left] = chunk
            left += 1
        else:
            reordered[right] = chunk
            right -= 1
    return reordered


def router_check(query, history):
    """Agent 意图识别：优化了 Prompt 和判断逻辑，去除了历史记录干扰"""
    prompt = f"""
你是一个意图识别助手。请判断用户的最新问题是否需要检索【计算机科学、软件工程、编程、IT技术】相关的专业知识库。

规则：
- 如果需要检索专业知识，只能输出 "true"
- 如果是闲聊、打招呼、通用常识，只能输出 "false"
- 绝对不要输出任何其他解释或标点符号！

例子：
用户：Python里怎么实现多线程？
助手：true

用户：今天天气真不错啊。
助手：false

用户：帮我写一段冒泡排序的代码。
助手：true

用户：你能帮我做什么？
助手：false

用户的最新问题：{query}
助手："""
    try:
        decision = generate(prompt, chunks=[], history=[])
        decision_text = decision.strip().lower()
        rag_logger.debug(f"路由模型原始输出: '{decision_text}'")
        return "true" in decision_text
    except Exception as e:
        rag_logger.warning(f"意图识别发生错误，默认走大模型直答: {e}")
        return False


# 🌟 核心引擎
async def run_hyperknow_rag(query: str, history: list):
    """
    接收用户问题，执行完整 RAG 流程，并源源不断地 yield 纯文本字符串
    """
    # 0. 查缓存
    cached_answer = get_cache(query)
    if cached_answer:
        yield cached_answer
        return

    # 0.5 查 FAQ
    faq_results = search_faq(query, top_k=1)
    if faq_results and faq_results[0]["score"] > FAQ_THRESHOLD:
        faq_answer = faq_results[0]["answer"]
        try:
            set_cache(query, faq_answer)
        except Exception:
            pass
        yield faq_answer
        return

    # 1. 意图识别：是不是IT专业问题
    if_need_search = router_check(query, history)
    chunks_to_use = []
    prefix_message = ""

    if if_need_search:
        # 🌟 2. 核心新增：自适应 Query 改写，拿到一个查询列表
        rewrite_queries = adaptive_query_rewrite(query, history)

        all_candidates = []
        # 🌟 3. 多路海量召回：遍历所有改写后的查询，分别去搜
        for q in rewrite_queries:
            # 这里给每个子问题捞 top_5 即可，避免总数太大爆显存
            candidates = hybrid_retrieve(q, top_k=5)
            if candidates:
                all_candidates.extend(candidates)

        # 🌟 4. 去重与原问题重排
        if all_candidates:
            # 简单的根据文档内容去重（防止多个子查询搜到同一篇文档）
            unique_docs_map = {doc["content"]: doc for doc in all_candidates}
            unique_docs = list(unique_docs_map.values())
            docs_texts = list(unique_docs_map.keys())

            # 【关键】使用“原始 query”进行统一的终极重排序 (Rerank)
            rerank_results = rerank(query, docs_texts, top_n=5)

            # 提取排序最高的 top_n 篇文档
            top_chunks = [unique_docs[r["index"]] for r in rerank_results]
            # 解决大模型遗忘问题
            chunks_to_use = reorder_lost_in_the_middle(top_chunks)
        else:
            prefix_message = "抱歉，我的知识库中未找到相关具体内容。基于我的通用知识，"
            yield prefix_message

    # 5. 生成流式回答
    final_answer = prefix_message
    try:
        async for chunk in generate_stream(query, chunks=chunks_to_use, history=history):
            final_answer += chunk
            yield chunk
    except Exception as e:
        yield f"\n[生成报错: {str(e)}]"

    # 6. 写缓存
    try:
        if final_answer.strip():
            set_cache(query, final_answer)
    except Exception:
        pass