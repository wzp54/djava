import json
import logging
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 这个导入完全没问题！
from app.rag_engine_qurey import run_hyperknow_rag
from app.logger import app_logger

# 配置 uvicorn 日志
logging.getLogger("uvicorn").setLevel(logging.INFO)

app = FastAPI()

app_logger.info("FastAPI 应用启动")

# 允许跨域（前端 Chainlit 需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    history: list = []

async def sse_formatter(query: str, history: list):
    """把 rag_engine 吐出来的纯文本，包装成 SSE 格式"""
    app_logger.info(f"收到聊天请求: query={query}, history长度={len(history)}")
    try:
        async for text_chunk in run_hyperknow_rag(query, history):
            yield f"data: {json.dumps({'token': text_chunk}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'complete': True})}\n\n"
    except Exception as e:
        app_logger.error(f"SSE 流式响应异常: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

# 🌟 核心排错点：必须是 @app.post，绝对不能是 @app.get，也不能漏掉！
@app.post("/api/chat/stream")
async def chat_endpoint(request: ChatRequest):
    app_logger.info(f"API 调用: /api/chat/stream, query={request.query}")
    return StreamingResponse(
        sse_formatter(request.query, request.history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    app_logger.info("启动 Uvicorn 服务器，端口: 8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)