"""
个人 AI 助手 v2.0 —— Miya 的整合作品
==========================================
之前 15 个项目的技术精华全在这里：

  v1.0 (项目07): Gradio 网页 + 8 工具 + 关键词 RAG
  v2.0 (本文件): + 流式打字效果 + Embedding 语义 RAG + 多轮工具协作

技术栈：
  对话引擎: DeepSeek API (deepseek-chat)
  语义搜索: Ollama nomic-embed-text (免费本地模型)
  网页界面: Gradio ChatInterface
  输出模式: 流式 (stream=True)，打字机效果

架构亮点：
  1. 多轮工具循环 → "加载文档然后搜索" 一气呵成
  2. 流式输出 → AI 边想边打字，不用等
  3. 语义 RAG → "Apple手机" 能搜到 "苹果公司发布智能手机"
"""
import sys
import os
import json
import io
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

import requests
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 第1部分：路径和配置
# ============================================================

WORK_DIR = os.path.join(os.path.dirname(__file__), "agent_workspace")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "document_library")
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

OLLAMA_EMBED = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

# 示例文档 —— 没有自己的文档时用它测试 RAG
SAMPLE_TEXT = """# Python 与人工智能导论

## 第一章：什么是人工智能

人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支，旨在创建能够模拟人类智能的系统。AI 系统可以执行学习、推理、感知、语言理解等任务。

人工智能主要分为三类：狭义人工智能（ANI）、通用人工智能（AGI）和超级人工智能（ASI）。目前我们使用的所有 AI 都属于狭义人工智能，它们只能在特定领域内执行任务。

## 第二章：机器学习基础

机器学习（Machine Learning）是人工智能的核心方法之一。它让计算机通过数据来学习，而不是通过显式编程。机器学习的三种主要范式是：监督学习、无监督学习和强化学习。

监督学习使用带标签的数据来训练模型。例如，给模型看很多猫和狗的图片，并告诉它每张图是什么，模型就能学会区分猫和狗。

无监督学习则使用没有标签的数据。它试图在数据中发现隐藏的模式或结构，比如将相似的客户分成不同群体。

强化学习让智能体在环境中通过试错来学习。智能体执行动作，环境给予奖励或惩罚，智能体逐渐学会最优策略。

## 第三章：深度学习

深度学习（Deep Learning）是机器学习的一个子领域，使用多层神经网络来学习数据的表示。深度学习在图像识别、自然语言处理、语音识别等领域取得了突破性进展。

著名的深度学习架构包括：卷积神经网络（CNN）用于图像处理，循环神经网络（RNN）用于序列数据，Transformer 架构用于自然语言处理。

## 第四章：Python 在 AI 中的应用

Python 是目前 AI 开发中最流行的编程语言。它的语法简洁，拥有丰富的科学计算和机器学习库，如 NumPy、Pandas、Scikit-learn、PyTorch 和 TensorFlow。

Python 的优势在于其庞大的社区和生态系统。无论你想做什么 AI 任务，几乎都能找到对应的 Python 库来帮助你。

## 第五章：未来展望

AI 技术正在快速发展。大语言模型（LLM）如 GPT、Claude 和 DeepSeek 正在改变人类与计算机交互的方式。未来的 AI 系统将更加智能、更加可靠，但也带来了伦理和安全性方面的挑战。
"""

SAMPLE_PATH = os.path.join(DOCS_DIR, "AI入门简介.txt")
if not os.path.exists(SAMPLE_PATH):
    with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
        f.write(SAMPLE_TEXT)


# ============================================================
# 第2部分：Embedding 向量工具 + RAG 引擎
# ============================================================

def embed(text: str) -> np.ndarray:
    """把文字转成神经网络语义向量（768维）"""
    resp = requests.post(OLLAMA_EMBED, json={
        "model": EMBED_MODEL, "prompt": text,
    }, timeout=60)
    return np.array(resp.json()["embedding"])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度 —— 衡量两个向量有多'接近'"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


class EmbeddingRAGEngine:
    """用神经网络 Embedding 做语义搜索的检索引擎。

    和关键词搜索的区别：
      关键词: "Apple手机" 找不到 "苹果公司..."（字面不匹配）
      Embedding: "Apple手机" ≈ "苹果公司发布智能手机"（语义相近）
    """

    def __init__(self):
        self.chunks = []
        self.vectors = []
        self.chunk_size = 200

    def load_file(self, filepath: str) -> str:
        filepath = filepath.replace("\\", "/")
        if not os.path.exists(filepath):
            return f"文件不存在：{filepath}"

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            # 切分段落
            paragraphs = text.split("\n")
            self.chunks = []
            current = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                current += para + "\n"
                if len(current) >= self.chunk_size:
                    self.chunks.append(current.strip())
                    current = ""
            if current.strip():
                self.chunks.append(current.strip())

            # 每个段落生成 Embedding 向量
            print(f"[RAG] 正在为 {len(self.chunks)} 个段落生成语义向量...")
            self.vectors = []
            for i, chunk in enumerate(self.chunks):
                self.vectors.append(embed(chunk))
                if (i + 1) % 5 == 0:
                    print(f"  {i+1}/{len(self.chunks)} 完成...")

            print(f"[RAG] 完成！{len(self.vectors)} 个向量，每个 {len(self.vectors[0])} 维")
            return f"已加载文件：{filepath}，共 {len(self.chunks)} 个段落，已生成语义向量"

        except Exception as e:
            return f"加载失败：{e}（Ollama 在运行吗？试试 ollama serve）"

    def search(self, query: str, top_k: int = 3) -> str:
        if not self.chunks:
            return "没有已加载的文档。请先用 load_document 加载一个文件。"

        query_vec = embed(query)
        scores = [cosine_similarity(query_vec, v) for v in self.vectors]

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = ranked[:top_k]

        result = []
        for idx, score in top:
            if score > 0.2:
                result.append(
                    f"[段落 {idx+1}，语义相似度: {score:.3f}]\n{self.chunks[idx]}"
                )

        if not result:
            return "没有找到相关内容。试试换个说法？"
        return "\n\n---\n\n".join(result)


# 全局 RAG 引擎
rag = EmbeddingRAGEngine()


# ============================================================
# 第3部分：8 个工具函数
# ============================================================

def get_weather(city: str) -> str:
    """查询天气（wttr.in API）"""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        current = data["current_condition"][0]
        weather_desc = current["weatherDesc"][0]["value"]
        temp = current["temp_C"]
        feels = current["FeelsLikeC"]
        humidity = current["humidity"]

        # 也获取明天的预报
        tomorrow = data["weather"][1]
        tmr_high = tomorrow["maxtempC"]
        tmr_low = tomorrow["mintempC"]
        tmr_desc = tomorrow["hourly"][4]["weatherDesc"][0]["value"]

        return (
            f"城市：{city}\n"
            f"当前：{weather_desc}，{temp}°C（体感 {feels}°C），湿度 {humidity}%\n"
            f"明天：{tmr_desc}，{tmr_low}°C ~ {tmr_high}°C"
        )
    except Exception as e:
        return f"天气查询失败：{e}"


def web_search(query: str, max_results: int = 5) -> str:
    """DuckDuckGo 网页搜索"""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "未找到相关结果。"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}\n   {r['href']}\n   {r['body'][:150]}")
        return "\n\n".join(lines)
    except ImportError:
        return "请先安装：pip install ddgs"
    except Exception as e:
        return f"搜索失败：{e}"


def run_python_code(code: str) -> str:
    """执行 Python 代码"""
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
        return "代码执行完成（无输出）"
    except Exception as e:
        sys.stdout = old_stdout
        return f"执行出错：{e}"


def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_file(path: str) -> str:
    """读文件"""
    try:
        full = os.path.join(WORK_DIR, path) if not os.path.isabs(path) else path
        if not os.path.exists(full):
            return f"文件不存在：{full}"
        with open(full, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 3000:
            content = content[:3000] + "\n...(内容太长了，后面省略)..."
        return content
    except Exception as e:
        return f"读取失败：{e}"


def write_file(path: str, content: str) -> str:
    """写文件"""
    try:
        full = os.path.join(WORK_DIR, path) if not os.path.isabs(path) else path
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已保存到：{full}"
    except Exception as e:
        return f"写入失败：{e}"


def load_document(filepath: str) -> str:
    """加载文档到 RAG 引擎"""
    return rag.load_file(filepath)


def search_document(query: str) -> str:
    """语义搜索已加载的文档"""
    return rag.search(query)


# ============================================================
# 第4部分：工具注册表
# ============================================================

TOOL_REGISTRY = {
    "get_weather": {
        "func": get_weather,
        "description": "查询城市实时天气和明天预报。城市名用英文（如 Shanghai, Beijing, Tokyo）。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市英文名"},
            },
            "required": ["city"],
        },
    },
    "web_search": {
        "func": web_search,
        "description": "在互联网上搜索最新信息。需要查实时资讯时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "max_results": {"type": "integer", "description": "最多返回几条（默认5）"},
            },
            "required": ["query"],
        },
    },
    "run_python_code": {
        "func": run_python_code,
        "description": "执行 Python 代码。需要数学计算、数据处理时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的 Python 代码"},
            },
            "required": ["code"],
        },
    },
    "get_current_time": {
        "func": get_current_time,
        "description": "获取当前的日期和时间。",
        "parameters": {"type": "object", "properties": {}},
    },
    "read_file": {
        "func": read_file,
        "description": f"读取文件内容（文件在 {WORK_DIR} 目录下）。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "func": write_file,
        "description": f"把内容保存到文件（保存到 {WORK_DIR} 目录下）。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件名"},
                "content": {"type": "string", "description": "要写入的内容"},
            },
            "required": ["path", "content"],
        },
    },
    "load_document": {
        "func": load_document,
        "description": f"加载一个 txt 文档到语义搜索引擎。加载后可以用 search_document 搜索文档内容。文档在 {DOCS_DIR} 目录下。",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "文档的完整路径"},
            },
            "required": ["filepath"],
        },
    },
    "search_document": {
        "func": search_document,
        "description": "用神经网络语义搜索在已加载的文档中查找相关内容。能理解同义词和跨语言（如 'Apple手机' 能匹配 '苹果公司发布智能手机'）。回答问题前必须先搜索。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索问题（用自然语言即可，Embedding 模型理解语义）"},
            },
            "required": ["query"],
        },
    },
}


# ============================================================
# 第5部分：流式 Agent 类
# ============================================================

class StreamingAgent:
    """
    流式 Agent —— 支持多轮工具调用 + 打字机效果输出。

    和之前 Agent 的区别：
      1. 工具调用阶段用 while 循环（支持连续调用多个工具）
      2. 最终回复用 stream=True（逐字返回，打字机效果）
      3. 工具执行时 yield 状态消息（用户知道 AI 在干什么）
    """

    def __init__(self, system_prompt: str = ""):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.messages = [
            {"role": "system", "content": system_prompt or "你是友好的AI助手，用中文回答。"},
        ]
        self.tools = []
        self.tool_map = {}

    def add_tool(self, name: str, func, description: str, parameters: dict):
        self.tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        })
        self.tool_map[name] = func

    def chat_stream(self, user_input: str):
        """
        流式对话 —— 生成器函数。

        阶段A：非流式 while 循环 → 检测并执行工具（可能多轮）
        阶段B：流式 stream=True → 逐字 yield 最终回复
        """
        self.messages.append({"role": "user", "content": user_input})

        tools_for_api = self.tools if self.tools else None
        max_turns = 5
        turns = 0

        # ============================================================
        # 阶段A：工具调用循环（非流式）
        # 用 while 循环是因为模型可能连续调用多个工具
        # 比如：load_document → search_document → 然后才是文字回复
        # ============================================================
        while turns < max_turns:
            turns += 1

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages,
                tools=tools_for_api,
            )
            reply = response.choices[0].message

            # 没有工具调用 → 跳出循环，进入流式回复阶段
            if not reply.tool_calls:
                # 如果模型直接给了文字回复（不需要工具）
                if reply.content:
                    # 用流式重新生成这个回复
                    break
                # content 为空且无 tool_calls → 不应该出现，但也跳出
                break

            # 有工具调用 → 逐个执行
            for tool_call in reply.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                # 通知用户 AI 正在用什么工具
                status_msg = f"\n🔧 正在使用 **{func_name}** 工具..."
                yield status_msg + "\n\n"

                try:
                    tool_result = str(self.tool_map[func_name](**func_args))
                except Exception as e:
                    tool_result = f"工具执行出错：{e}"

                self.messages.append({
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": tool_call.function.arguments,
                        },
                    }],
                })
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

        # ============================================================
        # 阶段B：流式输出最终回复
        # ============================================================
        stream = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            stream=True,
        )

        full_content = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_content += delta.content
                yield delta.content

        self.messages.append({"role": "assistant", "content": full_content})

    def clear_memory(self):
        """清空对话记忆，只保留系统提示"""
        self.messages = [self.messages[0]]


# ============================================================
# 第6部分：Gradio 网页界面
# ============================================================

SYSTEM_PROMPT = f"""你是一个全能的个人 AI 助手，名字叫 "Miya's AI Assistant v2.0"。

你有以下能力：
  - 查天气 (get_weather) —— 实时天气和明天预报
  - 搜索网页 (web_search) —— 获取最新信息
  - 执行 Python 代码 (run_python_code) —— 计算、处理数据
  - 获取当前时间 (get_current_time)
  - 读写文件 (read_file / write_file) —— 文件在 {WORK_DIR} 目录
  - 语义搜索文档 (search_document) —— 用神经网络 Embedding 理解文档内容
  - 加载文档 (load_document) —— 把文档加载到搜索引擎，文档在 {DOCS_DIR} 目录

重要规则：
  1. 如果用户问文档相关问题，先 load_document 加载文件，再用 search_document 搜索
  2. 如果用户问天气或需要实时信息，用对应的工具获取
  3. 用中文回答，语气友好、热情
  4. 搜索文档时，用自然语言提问即可，Embedding 模型理解语义

示例文档路径：{SAMPLE_PATH}"""


def chat_with_ai(user_message: str, history: list):
    """
    Gradio ChatInterface 的回调函数。
    这是个生成器 —— 每个 yield 都会实时显示在聊天框里。
    """
    # 懒加载：第一次调用时初始化 Agent（避免启动时就要 API key）
    if not hasattr(chat_with_ai, "agent"):
        chat_with_ai.agent = StreamingAgent(system_prompt=SYSTEM_PROMPT)
        for name, info in TOOL_REGISTRY.items():
            chat_with_ai.agent.add_tool(
                name, info["func"], info["description"], info["parameters"]
            )

    try:
        full_response = ""
        for chunk in chat_with_ai.agent.chat_stream(user_message):
            full_response += chunk
            yield full_response
    except Exception as e:
        yield f"出错了：{e}\n\n请检查网络连接和 API 配置。"


# ============================================================
# 第7部分：主程序
# ============================================================

if __name__ == "__main__":
    import gradio as gr
    import socket

    # 获取本机 IP（手机可以通过局域网访问）
    local_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    print("=" * 60)
    print("个人 AI 助手 v2.0")
    print(f"电脑访问: http://127.0.0.1:7860")
    print(f"手机访问: http://{local_ip}:7860")
    print("升级：流式打字效果 + Embedding 语义 RAG + 多轮工具协作")
    print("=" * 60)

    ui = gr.ChatInterface(
        fn=chat_with_ai,
        title="个人 AI 助手 v2.0 — Miya 的作品",
        description=f"""
        **v2.0 全新升级：**
        - ✨ **流式打字效果**：AI 边想边显示，像真人聊天
        - 🧠 **Embedding 语义搜索**：用神经网络理解文档，支持同义词和跨语言
        - 🔄 **多轮工具协作**：自动链式调用（加载→搜索→回答）

        **我能做什么：**
        - 查天气、搜网页、执行 Python 代码
        - 读写文件（目录：`{WORK_DIR}`）
        - 语义搜索你的文档（目录：`{DOCS_DIR}`）
        """,
        examples=[
            "帮我算一下 1 加到 100 是多少",
            "上海今天天气怎么样？",
            "搜索一下 Python 3.13 有什么新特性",
            f"加载文档 {SAMPLE_PATH} 然后告诉我机器学习有哪些范式",
            "现在几点了？今年还剩百分之多少的时间？计算一下",
            "帮我写一首关于编程的诗，保存到 poem.txt",
        ],
        theme="soft",
    )

    ui.launch()
