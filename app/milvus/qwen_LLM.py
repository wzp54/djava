import os

from openai import OpenAI


def _get_non_empty_env(*names, default=""):
    for name in names:
        value = os.getenv(name)
        if value is not None:
            value = value.strip()
            if value:
                return value
    return default


LLM_BASE_URL = _get_non_empty_env(
    "LLM_API_BASE_URL",
    default="https://dashscope.aliyuncs.com/compatible-mode/v1",
).rstrip("/")
LLM_API_KEY = _get_non_empty_env(
    "LLM_API_KEY",
    "DASHSCOPE_API_KEY",
    "OPENAI_API_KEY",
    default="EMPTY",
)
LLM_MODEL = _get_non_empty_env("LLM_MODEL", default="qwen-turbo")

client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)


def generate(query, chunks):
    context = "\n\n---\n\n".join(chunk["content"] for chunk in chunks)
    prompt = (
        "请基于以下参考资料回答问题。\n\n"
        f"参考资料：\n{context}\n\n"
        f"用户问题：{query}\n"
        "答案："
    )
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
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
