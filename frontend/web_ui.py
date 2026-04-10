import chainlit as cl
import httpx  # 🌟 必须用这个异步库
import json

API_URL = "http://127.0.0.1:8000"


@cl.on_chat_start
async def start():
    await cl.Message("👋 我是 Hyperknow-AI 学习助手，随时为你解答专业问题！").send()
    cl.user_session.set("history", [])


@cl.on_message
async def handle_msg(message: cl.Message):
    # 先在页面上建一个空的聊天气泡
    msg = cl.Message(content="")
    await msg.send()

    history = cl.user_session.get("history", [])
    answer = ""

    # 使用异步 httpx 发起流式请求
    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                    "POST",
                    f"{API_URL}/api/chat/stream",
                    json={"query": message.content, "history": history},
                    timeout=120.0
            ) as response:

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            chunk = json.loads(data_str)

                            # 接收到结束信号，跳出循环
                            if chunk.get("complete"):
                                break

                            token = chunk.get("token", "")
                            if token:
                                answer += token

                                # ==========================================
                                # 🎨 核心 UI 魔法：折叠思考过程
                                # ==========================================
                                # 将 <think> 替换为 HTML 的折叠标签 <details>
                                # 将 </think> 替换为闭合标签，并加上华丽的分割线
                                display_text = answer.replace(
                                    "<think>", "<details>\n<summary>💡 <b>AI 深度思考过程 (点击展开/折叠)</b></summary>\n\n<br>\n\n"
                                ).replace(
                                    "</think>", "\n\n</details>\n\n---\n\n"
                                )

                                # 直接全量更新内容，防止渲染截断
                                msg.content = display_text
                                await msg.update()




                        except json.JSONDecodeError:
                            pass

        except httpx.ReadTimeout:
            msg.content += "\n\n[⚠️ 请求超时：大模型思考时间过长]"
            await msg.update()
        except Exception as e:
            msg.content += f"\n\n[⚠️ 请求后端失败: {str(e)}]"
            await msg.update()

    # 保存历史记录
    if answer:
        history.append({"role": "user", "content": message.content})
        history.append({"role": "assistant", "content": answer})
        cl.user_session.set("history", history)