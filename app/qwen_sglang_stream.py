from openai import AsyncOpenAI

from app.logger import llm_logger
from app.model_api import resolve_llm_api_key, resolve_llm_base_url, resolve_llm_model

LLM_BASE_URL = resolve_llm_base_url()
LLM_API_KEY = resolve_llm_api_key()
LLM_MODEL = resolve_llm_model("qwen-turbo")

client = AsyncOpenAI(
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


async def generate_stream(query, chunks=None, history=None):
    chunks = chunks or []
    history = history or []

    llm_logger.info(f"LLM 流式生成请求: query={query}, chunks数量={len(chunks)}")

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
        stream = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        llm_logger.info("LLM 流式生成完成")
    except Exception as exc:
        llm_logger.error(f"LLM 流式生成失败: {exc}")
        raise
