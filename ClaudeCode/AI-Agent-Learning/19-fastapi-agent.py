"""
FastAPI + Agent 部署 —— 把 AI 助手变成线上服务
================================================
之前你的 Agent 只能在终端里跑，面试官看不到。
现在用 FastAPI 把它做成 REST API，全世界都能访问！

架构变化：
  之前: 终端输入 → Agent.chat() → print 到终端
  现在: HTTP 请求 → FastAPI → Agent.chat() → JSON 响应

FastAPI 是 Python 最流行的 Web 框架，特点：
  - 自动生成 API 文档（/docs），面试官可以直接在网页上试
  - 原生支持异步（async/await），和你的 Agent 完美配合
  - 类型注解自动校验请求参数（Pydantic）

启动方式：
  python 19-fastapi-agent.py
  然后访问 http://127.0.0.1:8000/docs 看自动生成的 API 文档！

安装依赖：
  pip install fastapi uvicorn sse-starlette
"""

import sys
import os
import io
import json
from datetime import datetime
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

load_dotenv()


# ============================================================
# 第一部分: Agent 核心（和你之前写的 Agent 一模一样）
# ============================================================

DS = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

SYSTEM_PROMPT = "你是有用的助手，用中文回答，简洁清晰，控制在200字以内。"


# ---- 工具函数 ----

def get_current_time() -> str:
    """获取当前的日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_python_code(code: str) -> str:
    """执行 Python 代码并返回结果"""
    old_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured
    try:
        local_vars = {}
        exec(code, {}, local_vars)
        sys.stdout = old_stdout
        output = captured.getvalue().strip()
        if output:
            return output
        if local_vars:
            return str(list(local_vars.values())[-1])
        return "执行完成"
    except Exception as e:
        sys.stdout = old_stdout
        return f"出错：{e}"


# ---- 工具注册表（熟悉的配方！） ----

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "执行 Python 代码。需要计算、数据处理时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python 代码"},
                },
                "required": ["code"],
            },
        },
    },
]

TOOL_MAP = {
    "get_current_time": get_current_time,
    "run_python_code": run_python_code,
}


def agent_chat(user_message: str, history: list[dict] = None) -> str:
    """非流式聊天：一次返回全部结果"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = DS.chat.completions.create(
        model="deepseek-chat", messages=messages, tools=TOOLS,
    )
    reply = response.choices[0].message

    # 工具调用循环
    while reply.tool_calls:
        messages.append({
            "role": "assistant",
            "content": reply.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in reply.tool_calls
            ],
        })
        for tc in reply.tool_calls:
            args = json.loads(tc.function.arguments)
            result = TOOL_MAP[tc.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

        response = DS.chat.completions.create(
            model="deepseek-chat", messages=messages,
        )
        reply = response.choices[0].message

    return reply.content


def agent_chat_stream(user_message: str, history: list[dict] = None):
    """流式聊天：逐字返回（SSE）"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    # 第一阶段：工具检测（非流式，因为需要完整 tool_calls）
    response = DS.chat.completions.create(
        model="deepseek-chat", messages=messages, tools=TOOLS,
    )
    reply = response.choices[0].message

    if reply.tool_calls:
        messages.append({
            "role": "assistant",
            "content": reply.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in reply.tool_calls
            ],
        })
        for tc in reply.tool_calls:
            args = json.loads(tc.function.arguments)
            result = TOOL_MAP[tc.function.name](**args)
            yield f"data: [工具] {tc.function.name}\n\n"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    # 第二阶段：流式输出最终回答
    stream = DS.chat.completions.create(
        model="deepseek-chat", messages=messages, stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            yield f"data: {text}\n\n"

    yield "data: [DONE]\n\n"


# ============================================================
# 第二部分: FastAPI 应用 —— 把 Agent 暴露为 REST API
# ============================================================

app = FastAPI(
    title="AI Agent API",
    description="DeepSeek 驱动的 AI 助手，支持工具调用和流式输出",
    version="1.0.0",
)


# ---- Pydantic 模型：定义请求和响应格式 ----

class ChatRequest(BaseModel):
    message: str = Field(description="用户消息")
    history: list[dict] = Field(default_factory=list, description="对话历史")


class ChatResponse(BaseModel):
    reply: str = Field(description="AI 回复")
    model: str = Field(default="deepseek-chat")


class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: dict


# ---- 端点 1: 首页 ----

@app.get("/", response_class=HTMLResponse)
async def home():
    """欢迎页面"""
    return """
    <html>
    <head><title>AI Agent API</title>
    <style>body{font-family:sans-serif;max-width:700px;margin:50px auto;padding:20px;}
    h1{color:#6C47FF}a{color:#6C47FF}</style></head>
    <body>
    <h1>🤖 AI Agent API</h1>
    <p>DeepSeek 驱动的智能助手，支持工具调用 + 流式输出</p>
    <ul>
      <li><a href='/docs'>📖 交互式 API 文档</a> — 可以直接在网页上试！</li>
      <li><a href='/tools'>🔧 查看可用工具</a></li>
      <li><code>POST /chat</code> — 对话（非流式）</li>
      <li><code>POST /chat/stream</code> — 对话（流式打字机效果）</li>
    </ul>
    <p><small>项目 19 · 把 Agent 部署到线上 · 面试直接给链接演示</small></p>
    </body>
    </html>
    """


# ---- 端点 2: 健康检查 ----

@app.get("/health")
async def health():
    """康检查：确认服务在运行"""
    return {"status": "ok", "model": "deepseek-chat"}


# ---- 端点 3: 查看可用工具 ----

@app.get("/tools", response_model=list[ToolInfo])
async def list_tools():
    """列出 Agent 可用的所有工具

    相当于把 TOOL_REGISTRY 暴露给外界。
    面试时访问 /tools 就能看到 Agent 有哪些能力。
    """
    return [
        ToolInfo(
            name=t["function"]["name"],
            description=t["function"]["description"],
            parameters=t["function"]["parameters"],
        )
        for t in TOOLS
    ]


# ---- 端点 4: 对话（非流式） ----

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """发送消息，获取 AI 回复。

    curl 测试:
      curl -X POST http://127.0.0.1:8000/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "现在几点了？"}'
    """
    try:
        reply = agent_chat(req.message, req.history)
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- 端点 5: 对话（流式 SSE） ----

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """发送消息，流式返回 AI 回复（Server-Sent Events）。

    curl 测试:
      curl -X POST http://127.0.0.1:8000/chat/stream \
        -H "Content-Type: application/json" \
        -d '{"message": "计算 1 加到 100"}' \
        --no-buffer
    """
    return StreamingResponse(
        agent_chat_stream(req.message, req.history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )


# ============================================================
# 启动说明
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("AI Agent API 启动中...")
    print("=" * 60)
    print()
    print("📍 访问地址:")
    print("   API 文档:  http://127.0.0.1:8000/docs")
    print("   工具列表:  http://127.0.0.1:8000/tools")
    print("   健康检查:  http://127.0.0.1:8000/health")
    print()
    print("🧪 快速测试:")
    print('   curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d \'{"message":"你好"}\'')
    print()
    print("面试时把这个链接发给面试官，现场演示！")
    print("=" * 60)

    uvicorn.run(app, host="127.0.0.1", port=8000)
