# Hyperknow-AI RAG Demo

一个基于 `FastAPI + Chainlit + Elasticsearch + Milvus + Redis + 阿里云百炼` 的 RAG 问答系统原型。

项目目标是搭建一条完整的智能问答链路：前端对话、后端流式输出、Embedding 检索、Rerank 重排、LLM 生成，以及基于 LLM-as-a-Judge 的评测脚本。

## 项目亮点

- 基于 `FastAPI` 提供流式问答接口
- 基于 `Chainlit` 提供简单对话前端
- 支持 `阿里云百炼` 统一接入：
  - 生成模型
  - Embedding
  - Rerank
  - Judge
- 支持 FAQ 命中、缓存命中、知识库检索、重排后生成
- 支持评测脚本，用于对 RAG 输出进行自动打分

## 技术栈

- 后端：`FastAPI`、`Uvicorn`
- 前端：`Chainlit`
- 模型能力：`DashScope / 阿里云百炼`
- 检索存储：`Elasticsearch`、`Milvus`
- 缓存：`Redis`
- Python SDK：`openai`、`requests`、`httpx`

## 项目结构

```text
.
├─ app/
│  ├─ main.py                 # FastAPI 入口
│  ├─ rag_engine_qurey.py     # RAG 主流程
│  ├─ retriever.py            # ES 检索
│  ├─ embedding.py            # 百炼 Embedding 适配
│  ├─ try_reraken.py          # 百炼 Rerank 适配
│  ├─ qwen_sglang.py          # 百炼 LLM 调用
│  ├─ qwen_sglang_stream.py   # 百炼流式 LLM 调用
│  ├─ model_api.py            # 模型统一配置
│  ├─ my_redis/               # Redis 缓存
│  └─ milvus/                 # FAQ / Milvus 相关脚本
├─ frontend/
│  └─ web_ui.py               # Chainlit 前端
├─ rag_evaluator.py           # 评测脚本
├─ rag_eval_langgraph/        # LangGraph 版评测脚本
├─ .env                       # 本地环境配置
└─ pyproject.toml
```

## 系统流程

```text
用户提问
   ↓
Chainlit 前端
   ↓
FastAPI /api/chat/stream
   ↓
缓存命中 → 直接返回
   ↓
FAQ 命中 → 直接返回
   ↓
路由判断是否需要检索
   ↓
Embedding → 检索 → Rerank
   ↓
阿里百炼 LLM 流式生成
   ↓
返回前端
```

## 当前默认端口

- 后端：`http://127.0.0.1:8000`
- 前端：`http://127.0.0.1:8001`
- 健康检查：`http://127.0.0.1:8000/health`

## 环境配置

当前版本只需要维护一个百炼 Key：

```env
DASHSCOPE_API_KEY=你的阿里云百炼APIKey
```

关键配置项位于 `.env`：

```env
LLM_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-turbo

EMBEDDING_API_URL=https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding
EMBEDDING_MODEL=text-embedding-v4

RERANKER_API_URL=https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
RERANKER_MODEL=gte-rerank-v2

JUDGE_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
JUDGE_MODEL=qwen-plus
```

数据库/中间件相关配置也在 `.env` 中维护：

- `ES_HOST` / `ES_PORT` / `ES_SCHEME`
- `MILVUS_HOST` / `MILVUS_PORT`
- `REDIS_HOST` / `REDIS_PORT`

## 安装依赖

推荐使用 Poetry：

```bash
poetry install
```

如果你已经有本地 Python 环境，也可以直接用现有解释器安装依赖。

## 启动方式

### 1) 启动后端

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --env-file .env
```

### 2) 启动前端

```bash
python -m chainlit run frontend/web_ui.py --host 0.0.0.0 --port 8001
```

### 3) 打开页面

```text
http://127.0.0.1:8001
```

## API 说明

### `POST /api/chat/stream`

请求体：

```json
{
  "query": "什么是 Java？",
  "history": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，我是 Hyperknow-AI"}
  ]
}
```

返回形式：`SSE` 流式输出

### `GET /health`

用于健康检查：

```json
{
  "status": "ok"
}
```

## 评测脚本

### 普通版

```bash
python rag_evaluator.py
```

### LangGraph 版

```bash
python rag_eval_langgraph/rag_evaluator.py
```

评测维度包括：

- `faithfulness`
- `answer_relevance`
- `overall_effectiveness`

## 已完成的改造

- 将生成模型统一切换为阿里云百炼
- 将 Embedding 统一切换为阿里云百炼
- 将 Rerank 统一切换为阿里云百炼
- 将 Judge 统一切换为阿里云百炼
- 支持只维护一个 `DASHSCOPE_API_KEY`
- 修复 Elasticsearch `http/https` 协议配置问题

## 已知问题

当前项目更偏“可演示原型”，而不是“完整生产系统”，主要还有这些待优化点：

- 多轮上下文能力不稳定，第二轮/第三轮对话仍有优化空间
- 当前并不是强制所有问题都走知识库，部分分支会直接调用大模型
- 前端后端地址仍有一定硬编码
- 缺少更完善的自动化测试和部署脚本
- README、架构图、脱敏配置仍可继续完善

