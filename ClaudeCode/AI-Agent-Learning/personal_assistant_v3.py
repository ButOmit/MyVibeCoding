"""
🌟 个人 AI 助手 v3.0 —— 终极集成版
======================================
把 22 个项目中最强的技术整合进一个可演示的完整系统：

  集成清单：
    项目13 流式输出      → SSE 打字机效果
    项目17 混合检索      → BM25 + Embedding + RRF + LLM 重排序
    项目19 FastAPI       → REST API + /docs 自动文档
    项目20 三层记忆      → 短期窗口 + 长期 Chroma + 自动摘要
    项目18 MCP 协议      → 工具动态发现（可选）

  架构：
    Browser → FastAPI → MemoryManager → RAGEngine → DeepSeek
                    ↘  MCP Tools（时间/代码/文件）

  启动后访问：
    http://127.0.0.1:8000        → 聊天界面
    http://127.0.0.1:8000/docs   → API 文档（面试官可以在这试）
    http://127.0.0.1:8000/health → 健康检查
"""

import sys
import os
import io
import json
import requests
import numpy as np
import jieba
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
import tiktoken
import chromadb

load_dotenv()

# ============================================================
# 配置
# ============================================================

DS = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
OLLAMA_EMBED = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "agent_workspace", "memory_v3")
os.makedirs(MEMORY_DIR, exist_ok=True)

# ============================================================
# 知识库文档
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
    "Embedding模型把文本映射成固定长度的浮点数向量，语义相近的文字向量方向也相近。",
    "FastAPI是Python高性能Web框架，支持异步处理和自动生成API文档。",
    "MCP协议（Model Context Protocol）是AI工具集成的开放标准。",
]


# ============================================================
# 第一部分: 三层记忆系统（项目20）
# ============================================================

class MemoryManager:
    """管理短期记忆（token窗口）+ 长期记忆（Chroma）+ 自动摘要"""

    def __init__(self, max_tokens: int = 4000):
        try:
            self.encoder = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self.summary = ""
        self.total_summarized = 0

        # Chroma 长期记忆
        self.chroma = chromadb.PersistentClient(path=MEMORY_DIR)
        try:
            self.chroma.delete_collection("memories")
        except Exception:
            pass
        self.collection = self.chroma.create_collection(
            name="memories", metadata={"hnsw:space": "cosine"},
        )
        self._mem_counter = 0

    def _embed(self, text: str) -> list[float]:
        resp = requests.post(OLLAMA_EMBED, json={
            "model": EMBED_MODEL, "prompt": text,
        }, timeout=60)
        return resp.json()["embedding"]

    def _count_tokens(self, messages: list[dict]) -> int:
        total = 0
        for m in messages:
            total += len(self.encoder.encode(str(m.get("content", "")))) + 4
        return total + 2

    def store_memory(self, fact: str):
        self._mem_counter += 1
        self.collection.add(
            ids=[f"m{self._mem_counter}"],
            embeddings=[self._embed(fact)],
            documents=[fact],
            metadatas=[{"source": "conversation"}],
        )

    def recall_memories(self, query: str, n: int = 3) -> list[str]:
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_embeddings=[self._embed(query)],
            n_results=min(n, self.collection.count()),
        )
        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]
        return [d for d, dist in zip(docs, dists) if dist < 0.7]

    def summarize_messages(self, messages: list[dict]) -> str:
        dialogue = ""
        for m in messages:
            role = "用户" if m["role"] == "user" else "助手"
            dialogue += f"{role}: {str(m.get('content', ''))[:300]}\n"

        response = DS.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "system", "content": "把对话压缩成简洁摘要，保留关键事实。100字以内。",
            }, {"role": "user", "content": f"摘要：\n{dialogue}"}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        # Token 超限 → 压缩旧消息
        if self._count_tokens(self.messages) > self.max_tokens - 1500:
            old = self.messages[1:4]
            if old:
                self.summary += " | " + self.summarize_messages(old)
                self.messages = [self.messages[0]] + self.messages[4:]
                self.total_summarized += len(old)

    def build_context(self, user_msg: str) -> list[dict]:
        system = "你是智能助手，用中文回答，简洁有帮助。"
        if self.summary:
            system += f"\n\n[历史摘要]\n{self.summary}"
        memories = self.recall_memories(user_msg)
        if memories:
            system += "\n\n[相关记忆]\n" + "\n".join(f"- {m}" for m in memories)
        return [{"role": "system", "content": system}] + self.messages

    def extract_and_store(self, user_msg: str, reply: str):
        prompt = f"提取值得记住的事实，一行一个，- 开头。没有就答'无'。\n用户: {user_msg[:300]}\n助手: {reply[:300]}"
        resp = DS.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        for line in resp.choices[0].message.content.strip().split("\n"):
            fact = line.strip()
            if fact.startswith("- ") and fact != "- 无":
                self.store_memory(fact[2:])


# ============================================================
# 第二部分: 混合 RAG 引擎（项目17）
# ============================================================

class RAGEngine:
    """BM25 + Embedding 混合检索 + LLM 重排序"""

    def __init__(self, documents: list[str]):
        self.docs = documents
        self.vectors = []
        self._ready = False

    def _ensure_ready(self):
        if self._ready:
            return
        # BM25 索引
        tokenized = [list(jieba.cut(d)) for d in self.docs]
        self.bm25 = BM25Okapi(tokenized)
        # Embedding 向量（只算一次）
        print(f"  [RAG] 正在为 {len(self.docs)} 篇文档生成 Embedding...")
        for i, d in enumerate(self.docs):
            resp = requests.post(OLLAMA_EMBED, json={
                "model": EMBED_MODEL, "prompt": d,
            }, timeout=60)
            self.vectors.append(np.array(resp.json()["embedding"]))
            if (i + 1) % 8 == 0:
                print(f"    {i+1}/{len(self.docs)} 完成...")
        print(f"  [RAG] Embedding 就绪！")
        self._ready = True

    def _embed(self, text: str) -> np.ndarray:
        resp = requests.post(OLLAMA_EMBED, json={
            "model": EMBED_MODEL, "prompt": text,
        }, timeout=60)
        return np.array(resp.json()["embedding"])

    def search(self, query: str, top_k: int = 3) -> list[str]:
        self._ensure_ready()
        # BM25
        bm25_scores = self.bm25.get_scores(list(jieba.cut(query)))
        bm25_ranked = sorted(
            [(i, s) for i, s in enumerate(bm25_scores) if s > 0],
            key=lambda x: x[1], reverse=True,
        )[:10]
        # Embedding
        qv = self._embed(query)
        embed_ranked = sorted(
            [(i, float(np.dot(qv, v) / (np.linalg.norm(qv) * np.linalg.norm(v) + 1e-10)))
             for i, v in enumerate(self.vectors)],
            key=lambda x: x[1], reverse=True,
        )[:10]
        # RRF 融合
        scores = {}
        for rank, (idx, _) in enumerate(bm25_ranked):
            scores[idx] = scores.get(idx, 0) + 1.0 / (60 + rank + 1)
        for rank, (idx, _) in enumerate(embed_ranked):
            scores[idx] = scores.get(idx, 0) + 1.0 / (60 + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.docs[i] for i, _ in ranked[:top_k]]

    def rag_answer(self, question: str) -> str:
        chunks = self.search(question, top_k=3)
        context = "\n\n".join(chunks)
        resp = DS.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "system", "content": "根据资料回答问题。资料没有就说不知道。100字以内。",
            }, {
                "role": "user", "content": f"资料：\n{context}\n\n问题：{question}",
            }],
            temperature=0.3,
        )
        return resp.choices[0].message.content


# ============================================================
# 第三部分: 工具函数
# ============================================================

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_python_code(code: str) -> str:
    old = sys.stdout
    cap = io.StringIO()
    sys.stdout = cap
    try:
        local_vars = {}
        exec(code, {}, local_vars)
        sys.stdout = old
        out = cap.getvalue().strip()
        if out:
            return out
        if local_vars:
            return str(list(local_vars.values())[-1])
        return "执行完成"
    except Exception as e:
        sys.stdout = old
        return f"出错：{e}"


TOOLS = [
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前日期和时间", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "run_python_code", "description": "执行Python代码，用于计算和数据处理", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python代码"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "search_knowledge", "description": "搜索AI知识库", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索关键词"}}, "required": ["query"]}}},
]

TOOL_MAP = {
    "get_current_time": get_current_time,
    "run_python_code": run_python_code,
}


# ============================================================
# 第四部分: 核心 Agent
# ============================================================

# 全局初始化
memory = MemoryManager()
rag = RAGEngine(KNOWLEDGE_BASE)


def agent_chat_stream(user_msg: str):
    """流式对话：工具检测 + 流式输出"""
    memory.add_message("user", user_msg)

    context = memory.build_context(user_msg)
    context.append({"role": "user", "content": user_msg})

    # 阶段1：工具检测
    resp = DS.chat.completions.create(
        model="deepseek-chat", messages=context, tools=TOOLS,
    )
    reply = resp.choices[0].message

    if reply.tool_calls:
        for tc in reply.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            yield f"data: [工具] {name}\n\n"

            if name == "search_knowledge":
                result = rag.rag_answer(args.get("query", ""))
            else:
                result = TOOL_MAP.get(name, lambda **kw: "未知工具")(**args)

            context.append({"role": "assistant", "content": reply.content or "", "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": name, "arguments": tc.function.arguments}}]})
            context.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    # 阶段2：流式输出
    stream = DS.chat.completions.create(
        model="deepseek-chat", messages=context, stream=True,
    )
    full_reply = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full_reply += text
            yield f"data: {text}\n\n"

    memory.add_message("assistant", full_reply)
    memory.extract_and_store(user_msg, full_reply)
    yield "data: [DONE]\n\n"


def agent_chat(user_msg: str) -> str:
    """非流式对话"""
    memory.add_message("user", user_msg)
    context = memory.build_context(user_msg)
    context.append({"role": "user", "content": user_msg})

    resp = DS.chat.completions.create(
        model="deepseek-chat", messages=context, tools=TOOLS,
    )
    reply = resp.choices[0].message

    if reply.tool_calls:
        for tc in reply.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            if name == "search_knowledge":
                result = rag.rag_answer(args.get("query", ""))
            else:
                result = TOOL_MAP.get(name, lambda **kw: "未知")(**args)
            context.append({"role": "assistant", "content": reply.content or "", "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": name, "arguments": tc.function.arguments}}]})
            context.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

        resp = DS.chat.completions.create(model="deepseek-chat", messages=context)
        reply = resp.choices[0].message

    memory.add_message("assistant", reply.content)
    memory.extract_and_store(user_msg, reply.content)
    return reply.content


# ============================================================
# 第五部分: FastAPI 应用
# ============================================================

app = FastAPI(title="🌟 AI 助手 v3.0", description="集成：三层记忆 + 混合RAG + 流式 + 工具调用", version="3.0.0")


class ChatRequest(BaseModel):
    message: str = Field(description="用户消息")


class ChatResponse(BaseModel):
    reply: str


class MemoryStats(BaseModel):
    short_term_messages: int
    long_term_memories: int
    has_summary: bool
    total_summarized: int


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI 助手 v3.0</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,sans-serif;background:#1a1a2e;color:#eee;height:100vh;display:flex;flex-direction:column}
.header{background:#16213e;padding:16px 24px;text-align:center;border-bottom:2px solid #6C47FF}
.header h1{font-size:1.3em;color:#6C47FF}
.header p{font-size:0.8em;color:#999;margin-top:4px}
.chat{flex:1;overflow-y:auto;padding:20px;max-width:800px;margin:0 auto;width:100%}
.msg{margin-bottom:16px;display:flex;gap:10px}
.msg.user{flex-direction:row-reverse}
.msg .avatar{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.2em;flex-shrink:0}
.msg.assistant .avatar{background:#6C47FF}
.msg.user .avatar{background:#e94560}
.msg .bubble{max-width:75%;padding:12px 16px;border-radius:16px;line-height:1.6;font-size:0.95em}
.msg.assistant .bubble{background:#16213e;border-bottom-left-radius:4px}
.msg.user .bubble{background:#6C47FF;border-bottom-right-radius:4px}
.msg .bubble .tool{color:#ffd700;font-size:0.8em;margin-bottom:4px}
.input-area{background:#16213e;padding:16px 24px;display:flex;gap:10px;max-width:800px;margin:0 auto;width:100%;border-top:1px solid #333}
.input-area input{flex:1;padding:12px 16px;border-radius:24px;border:none;background:#1a1a2e;color:#eee;font-size:0.95em;outline:none}
.input-area input:focus{box-shadow:0 0 0 2px #6C47FF}
.input-area button{padding:12px 24px;border-radius:24px;border:none;background:#6C47FF;color:#fff;font-size:0.95em;cursor:pointer}
.input-area button:hover{background:#8059ff}
.info-bar{text-align:center;padding:8px;font-size:0.75em;color:#666;background:#111}
.info-bar a{color:#6C47FF}
</style>
</head>
<body>
<div class="header"><h1>🤖 AI 助手 v3.0</h1><p>三层记忆 + 混合RAG + 工具调用</p></div>
<div class="chat" id="chat"></div>
<div class="input-area">
<input id="input" placeholder="问我任何问题..." onkeydown="if(event.key==='Enter')send()">
<button onclick="send()">发送</button>
</div>
<div class="info-bar"><a href="/docs">📖 API 文档</a> | <a href="/memory/stats">🧠 记忆状态</a> | <a href="https://github.com/ButOmit/AI-Agent-Learning">💻 GitHub</a></div>
<script>
function addMsg(role,text){let d=document.getElementById('chat'),m=document.createElement('div');m.className='msg '+role;m.innerHTML=`<div class="avatar">${role==='user'?'👤':'🤖'}</div><div class="bubble">${text}</div>`;d.appendChild(m);d.scrollTop=d.scrollHeight;return m}
async function send(){let i=document.getElementById('input'),msg=i.value.trim();if(!msg)return;i.value='';addMsg('user',msg);let b=addMsg('assistant','');let resp=await fetch('/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});let r=await resp.body.getReader(),d=new TextDecoder(),txt='';while(true){let{value,done}=await r.read();if(done)break;let lines=d.decode(value).split('\\n');for(let l of lines){if(l.startsWith('data: ')){let c=l.slice(6);if(c==='[DONE]')break;if(c.startsWith('[工具]')){b.querySelector('.bubble').innerHTML+='<div class="tool">🔧 '+c+'</div>'}else{txt+=c;b.querySelector('.bubble').innerHTML=txt.replace(/\\n/g,'<br>')}}}}}
</script>
</body>
</html>"""


@app.get("/health")
async def health():
    return {"status": "ok", "rag_docs": len(KNOWLEDGE_BASE), "memories": memory.collection.count()}


@app.get("/memory/stats", response_model=MemoryStats)
async def memory_stats():
    return MemoryStats(
        short_term_messages=len(memory.messages),
        long_term_memories=memory.collection.count(),
        has_summary=bool(memory.summary),
        total_summarized=memory.total_summarized,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        reply = agent_chat(req.message)
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        agent_chat_stream(req.message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🌟 AI 助手 v3.0 启动中...")
    print("=" * 60)
    print(f"  📖 知识库: {len(KNOWLEDGE_BASE)} 篇文档")
    print(f"  🧠 记忆系统: 短期(token窗口) + 长期(Chroma) + 自动摘要")
    print(f"  🔍 RAG: BM25 + Embedding + RRF 混合检索")
    print(f"  🔧 工具: 时间查询 / Python执行 / 知识库搜索")
    print(f"  ⚡ 流式: SSE 打字机效果")
    print()
    print(f"  🌐 http://127.0.0.1:8000       → 聊天界面")
    print(f"  📖 http://127.0.0.1:8000/docs  → API 文档")
    print(f"  ❤️ http://127.0.0.1:8000/health → 健康检查")
    print()
    print("面试时把这个页面打开，直接演示！")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
