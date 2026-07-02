"""
🌟 AI 助手 v3.0 云端版 —— 零依赖，一键部署 Railway
=====================================================
和本地版的区别：去掉了 Ollama 依赖，纯 CPU + DeepSeek API。
免费服务器 512MB 内存就能跑。

集成功能：
  记忆系统    — 短期 token 窗口 + 长期文本存储 + 自动摘要
  RAG 知识库  — BM25 关键词检索（不需要 Ollama 向量）
  流式输出    — SSE 打字机效果
  工具调用    — 时间 / Python 执行 / 知识库搜索
  FastAPI     — REST API + /docs 自动文档

部署到 Railway：
  1. 把这个文件推送到 GitHub
  2. Railway 关联仓库，自动检测 Dockerfile
  3. 设置环境变量 DEEPSEEK_API_KEY
  4. 完成！

启动：
  uvicorn personal_assistant_v3_cloud:app --host 0.0.0.0 --port 8000
"""

import sys
import os
import io
import json
import re
from datetime import datetime
from collections import OrderedDict

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import jieba
from rank_bm25 import BM25Okapi

load_dotenv()

DS = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

# ============================================================
# 知识库
# ============================================================

KNOWLEDGE_BASE = [
    "人工智能（Artificial Intelligence）是计算机科学的分支，旨在模拟人类智能。",
    "机器学习是人工智能的核心方法，让计算机从数据中学习规律。",
    "机器学习的三种主要范式：监督学习、无监督学习和强化学习。",
    "深度学习使用多层神经网络来提取数据的层次化特征表示。",
    "卷积神经网络（CNN）特别擅长处理图像数据。",
    "循环神经网络（RNN）用于处理序列数据，如时间序列和自然语言。",
    "Transformer架构通过自注意力机制（Self-Attention）并行处理序列。",
    "大语言模型（LLM）如DeepSeek、GPT基于Transformer架构，参数量达千亿级。",
    "Python是AI和数据分析领域最流行的编程语言。",
    "强化学习中，智能体（Agent）通过与环境交互获得奖励信号来学习最优策略。",
    "GPU比CPU更适合矩阵运算，能大幅加速深度学习训练。",
    "过拟合指模型在训练集表现好但在新数据上表现差，需要正则化来解决。",
    "向量数据库（如Chroma）专门存储和检索高维向量，是RAG系统的核心组件。",
    "FastAPI是Python高性能Web框架，支持异步处理和自动生成API文档。",
    "MCP协议（Model Context Protocol）是AI工具集成的开放标准。",
    "LangGraph是LangChain的升级版Agent框架，用图结构定义工作流。",
]


# ============================================================
# 记忆系统（纯文本版，不依赖 Ollama）
# ============================================================

class MemoryManager:
    def __init__(self, max_turns: int = 20):
        self.messages: list[dict] = []
        self.long_term: OrderedDict = OrderedDict()  # key → value
        self.summary = ""
        self.max_turns = max_turns

    def store_fact(self, key: str, value: str):
        self.long_term[key] = value
        if len(self.long_term) > 50:
            self.long_term.popitem(last=False)

    def recall(self, query: str, n: int = 3) -> list[str]:
        # 简单关键词匹配（不需要向量）
        results = []
        for k, v in self.long_term.items():
            if any(w in k + v for w in jieba.cut(query)):
                results.append(f"{k}: {v}")
        return results[:n]

    def summarize(self, msgs: list[dict]) -> str:
        dialogue = ""
        for m in msgs:
            role = "用户" if m["role"] == "user" else "助手"
            dialogue += f"{role}: {str(m.get('content', ''))[:300]}\n"
        resp = DS.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "把对话压缩为简洁摘要，100字以内。"},
                {"role": "user", "content": f"摘要：\n{dialogue}"},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_turns * 2:
            old = self.messages[:4]
            self.summary += " | " + self.summarize(old)
            self.messages = self.messages[4:]

    def build_context(self, user_msg: str) -> list[dict]:
        system = "你是智能助手，用中文回答，简洁有帮助。"
        if self.summary:
            system += f"\n\n[历史摘要]\n{self.summary}"
        recalled = self.recall(user_msg)
        if recalled:
            system += "\n\n[相关记忆]\n" + "\n".join(f"- {r}" for r in recalled)
        return [{"role": "system", "content": system}] + self.messages

    def extract_and_store(self, user_msg: str, reply: str):
        prompt = f"提取一条值得记住的事实。没有就答'无'。\n用户: {user_msg[:200]}\n助手: {reply[:200]}"
        resp = DS.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        result = resp.choices[0].message.content.strip()
        if result != "无" and len(result) > 3:
            self.store_fact(f"关于用户", result)


# ============================================================
# RAG 引擎（BM25 关键词版，不需要 Ollama）
# ============================================================

class RAGEngine:
    def __init__(self, docs: list[str]):
        self.docs = docs
        tokenized = [list(jieba.cut(d)) for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        scores = self.bm25.get_scores(list(jieba.cut(query)))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [self.docs[i] for i, s in ranked[:top_k] if s > 0]

    def answer(self, question: str) -> str:
        chunks = self.search(question, top_k=3)
        context = "\n\n".join(chunks)
        resp = DS.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "根据资料回答。没有就说不知道。100字以内。"},
                {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content


# ============================================================
# 工具
# ============================================================

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_python_code(code: str) -> str:
    old, cap = sys.stdout, io.StringIO()
    sys.stdout = cap
    try:
        local_vars = {}
        exec(code, {}, local_vars)
        sys.stdout = old
        out = cap.getvalue().strip()
        return out or str(list(local_vars.values())[-1]) if local_vars else "OK"
    except Exception as e:
        sys.stdout = old
        return f"错误：{e}"


TOOLS = [
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前日期和时间", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "run_python_code", "description": "执行Python代码", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "search_knowledge", "description": "搜索AI知识库", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]

TOOL_MAP = {"get_current_time": get_current_time, "run_python_code": run_python_code}

# ============================================================
# 全局初始化
# ============================================================

memory = MemoryManager()
rag = RAGEngine(KNOWLEDGE_BASE)


# ============================================================
# Agent 核心
# ============================================================

def agent_chat_stream(user_msg: str):
    memory.add_message("user", user_msg)
    ctx = memory.build_context(user_msg)
    ctx.append({"role": "user", "content": user_msg})

    resp = DS.chat.completions.create(model="deepseek-chat", messages=ctx, tools=TOOLS)
    reply = resp.choices[0].message

    if reply.tool_calls:
        for tc in reply.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            yield f"data: [工具] {name}\n\n"
            result = rag.answer(args["query"]) if name == "search_knowledge" else TOOL_MAP.get(name, lambda **kw: "?")(**args)
            ctx.append({"role": "assistant", "content": reply.content or "", "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": name, "arguments": tc.function.arguments}}]})
            ctx.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    stream = DS.chat.completions.create(model="deepseek-chat", messages=ctx, stream=True)
    full = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full += text
            yield f"data: {text}\n\n"

    memory.add_message("assistant", full)
    memory.extract_and_store(user_msg, full)
    yield "data: [DONE]\n\n"


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(title="AI 助手 v3.0 Cloud", description="零依赖云端版 | 记忆+RAG+流式+工具", version="3.0-cloud")


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def home():
    return """<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI 助手 v3.0</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:-apple-system,sans-serif;background:#1a1a2e;color:#eee;height:100vh;display:flex;flex-direction:column}
.header{background:#16213e;padding:16px 24px;text-align:center;border-bottom:2px solid #6C47FF}.header h1{font-size:1.3em;color:#6C47FF}.header p{font-size:.8em;color:#999;margin-top:4px}
.chat{flex:1;overflow-y:auto;padding:20px;max-width:800px;margin:0 auto;width:100%}
.msg{margin-bottom:16px;display:flex;gap:10px}.msg.user{flex-direction:row-reverse}
.msg .avatar{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.2em;flex-shrink:0}
.msg.assistant .avatar{background:#6C47FF}.msg.user .avatar{background:#e94560}
.msg .bubble{max-width:75%;padding:12px 16px;border-radius:16px;line-height:1.6;font-size:.95em}
.msg.assistant .bubble{background:#16213e;border-bottom-left-radius:4px}.msg.user .bubble{background:#6C47FF;border-bottom-right-radius:4px}
.msg .bubble .tool{color:#ffd700;font-size:.8em;margin-bottom:4px}
.input-area{background:#16213e;padding:16px 24px;display:flex;gap:10px;max-width:800px;margin:0 auto;width:100%;border-top:1px solid #333}
.input-area input{flex:1;padding:12px 16px;border-radius:24px;border:none;background:#1a1a2e;color:#eee;font-size:.95em;outline:none}
.input-area input:focus{box-shadow:0 0 0 2px #6C47FF}
.input-area button{padding:12px 24px;border-radius:24px;border:none;background:#6C47FF;color:#fff;font-size:.95em;cursor:pointer}
.input-area button:hover{background:#8059ff}
.info-bar{text-align:center;padding:8px;font-size:.75em;color:#666;background:#111}.info-bar a{color:#6C47FF}
</style></head><body>
<div class="header"><h1>AI 助手 v3.0 Cloud</h1><p>记忆 + RAG + 流式 + 工具 | Railway 云端部署</p></div>
<div class="chat" id="chat"></div>
<div class="input-area"><input id="input" placeholder="问我任何问题..." onkeydown="if(event.key==='Enter')send()"><button onclick="send()">发送</button></div>
<div class="info-bar"><a href="/docs">API 文档</a> | <a href="/health">健康检查</a> | <a href="https://github.com/ButOmit/AI-Agent-Learning">GitHub</a></div>
<script>
function addMsg(role,text){let d=document.getElementById('chat'),m=document.createElement('div');m.className='msg '+role;m.innerHTML=`<div class="avatar">${role==='user'?'👤':'🤖'}</div><div class="bubble">${text}</div>`;d.appendChild(m);d.scrollTop=d.scrollHeight;return m}
async function send(){let i=document.getElementById('input'),msg=i.value.trim();if(!msg)return;i.value='';addMsg('user',msg);let b=addMsg('assistant','');let resp=await fetch('/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});let r=await resp.body.getReader(),d=new TextDecoder(),txt='';while(true){let{value,done}=await r.read();if(done)break;let lines=d.decode(value).split('\\n');for(let l of lines){if(l.startsWith('data: ')){let c=l.slice(6);if(c==='[DONE]')break;if(c.startsWith('[工具]')){b.querySelector('.bubble').innerHTML+='<div class="tool">🔧 '+c+'</div>'}else{txt+=c;b.querySelector('.bubble').innerHTML=txt.replace(/\\n/g,'<br>')}}}}}
</script></body></html>"""


@app.get("/health")
async def health():
    return {"status": "ok", "rag_docs": len(KNOWLEDGE_BASE), "memories": len(memory.long_term)}


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        memory.add_message("user", req.message)
        ctx = memory.build_context(req.message)
        ctx.append({"role": "user", "content": req.message})
        resp = DS.chat.completions.create(model="deepseek-chat", messages=ctx, tools=TOOLS)
        reply = resp.choices[0].message
        if reply.tool_calls:
            for tc in reply.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                result = rag.answer(args["query"]) if name == "search_knowledge" else TOOL_MAP.get(name, lambda **kw: "?")(**args)
                ctx.append({"role": "assistant", "content": reply.content or "", "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": name, "arguments": tc.function.arguments}}]})
                ctx.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
            resp = DS.chat.completions.create(model="deepseek-chat", messages=ctx)
            reply = resp.choices[0].message
        memory.add_message("assistant", reply.content)
        memory.extract_and_store(req.message, reply.content)
        return {"reply": reply.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(agent_chat_stream(req.message), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


if __name__ == "__main__":
    import uvicorn
    print(" AI 助手 v3.0 Cloud 启动")
    print(" http://0.0.0.0:8000")
    print(" 零 Ollama 依赖，纯 DeepSeek API + BM25")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
