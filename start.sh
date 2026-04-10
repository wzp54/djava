#!/bin/bash

# 启动 FastAPI 后端
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!

# 等一下让 uvicorn 先绑定端口
sleep 2

# 启动 Chainlit 前端
cd /code/frontend
chainlit run web_ui.py --host 0.0.0.0 --port 8001 &
CHAINLIT_PID=$!

# 等任意一个退出
wait $UVICORN_PID $CHAINLIT_PID