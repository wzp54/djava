import os


def _get_non_empty_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            value = value.strip()
            if value:
                return value
    return default


def resolve_dashscope_api_key() -> str:
    return _get_non_empty_env("DASHSCOPE_API_KEY", default="")


def build_bearer_headers(api_key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def resolve_llm_base_url() -> str:
    return _get_non_empty_env(
        "LLM_API_BASE_URL",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/")


def resolve_llm_api_key() -> str:
    return _get_non_empty_env(
        "DASHSCOPE_API_KEY",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        default="EMPTY",
    )


def resolve_llm_model(default_model: str = "qwen-turbo") -> str:
    return _get_non_empty_env("LLM_MODEL", default=default_model)


def resolve_embedding_url() -> str:
    return _get_non_empty_env(
        "EMBEDDING_API_URL",
        default=(
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
            "text-embedding/text-embedding"
        ),
    ).rstrip("/")


def resolve_embedding_api_key() -> str:
    return _get_non_empty_env(
        "DASHSCOPE_API_KEY",
        "EMBEDDING_API_KEY",
        "LLM_API_KEY",
        default="",
    )


def resolve_embedding_model(default_model: str = "text-embedding-v4") -> str:
    return _get_non_empty_env("EMBEDDING_MODEL", default=default_model)


def resolve_reranker_url() -> str:
    return _get_non_empty_env(
        "RERANKER_API_URL",
        default=(
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/"
            "text-rerank/text-rerank"
        ),
    ).rstrip("/")


def resolve_reranker_api_key() -> str:
    return _get_non_empty_env(
        "DASHSCOPE_API_KEY",
        "RERANKER_API_KEY",
        "LLM_API_KEY",
        default="",
    )


def resolve_reranker_model(default_model: str = "gte-rerank-v2") -> str:
    return _get_non_empty_env("RERANKER_MODEL", default=default_model)


def resolve_judge_base_url() -> str:
    return _get_non_empty_env(
        "JUDGE_API_BASE_URL",
        "LLM_API_BASE_URL",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/")


def resolve_judge_api_key() -> str:
    return _get_non_empty_env(
        "DASHSCOPE_API_KEY",
        "JUDGE_API_KEY",
        "LLM_API_KEY",
        default="",
    )


def resolve_judge_model(default_model: str = "qwen-plus") -> str:
    return _get_non_empty_env("JUDGE_MODEL", default=default_model)


def resolve_judge_group_id() -> str:
    return _get_non_empty_env("JUDGE_GROUP_ID", default="")
