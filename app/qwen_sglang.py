from openai import OpenAI

from app.logger import llm_logger
from app.model_api import resolve_llm_api_key, resolve_llm_base_url, resolve_llm_model

LLM_BASE_URL = resolve_llm_base_url()
LLM_API_KEY = resolve_llm_api_key()
LLM_MODEL = resolve_llm_model("qwen-turbo")

llm_logger.info(f"初始化阿里百炼 LLM 客户端: {LLM_BASE_URL}")

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)


def _build_prompt(query, chunks):
    if not chunks:
        return query
    context = "\n\n---\n\n".join(chunk["content"] for chunk in chunks)
    return (
        "请基于以下参考资料回答问题。\n\n"
        f"参考资料：\n{context}\n\n"
        f"问题：{query}\n"
        "答案："
    )


def generate(query, chunks=None, history=None):
    chunks = chunks or []
    history = history or []

    llm_logger.info(
        f"LLM 生成请求: query={query}, chunks数量={len(chunks)}, history长度={len(history)}"
    )

    messages = [
        {
            "role": "system",
            "content": "你是专业的计算机学习助手 Hyperknow-AI，请专业、清晰地回答问题。",
        }
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": _build_prompt(query, chunks)})

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0,
        )
        result = response.choices[0].message.content or ""
        llm_logger.info(f"LLM 生成完成，回复长度: {len(result)}")
        return result
    except Exception as exc:
        llm_logger.error(f"LLM 生成失败: {exc}")
        raise
