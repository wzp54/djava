from openai import OpenAI

from app.model_api import resolve_llm_api_key, resolve_llm_base_url, resolve_llm_model

LLM_BASE_URL = resolve_llm_base_url()
LLM_API_KEY = resolve_llm_api_key()
LLM_MODEL = resolve_llm_model("qwen-turbo")

client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)


def _build_rag_prompt(query, chunks):
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

    messages = [
        {
            "role": "system",
            "content": "你是专业的计算机学习助手 Hyperknow-AI，请专业、清晰地回答问题。",
        }
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": _build_rag_prompt(query, chunks)})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def generate_raw(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""
