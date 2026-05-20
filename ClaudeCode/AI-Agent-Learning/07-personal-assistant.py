"""
个人 AI 助手 —— 带网页界面的全能 Agent
=========================================
整合了全部 8 个工具，用 Gradio 做了网页界面。
面试时用手机打开就能演示 —— 比简历上的任何话都有说服力。
"""
import sys
import os
import json
import io
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

WORK_DIR = os.path.join(os.path.dirname(__file__), "agent_workspace")
os.makedirs(WORK_DIR, exist_ok=True)

# ============================================================
# 8 个工具函数
# ============================================================

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
        return "代码执行完成（无输出）"
    except Exception as e:
        sys.stdout = old_stdout
        return f"执行出错：{e}"


def get_current_time() -> str:
    """获取当前日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_file(path: str) -> str:
    """读取文件内容"""
    if not os.path.exists(path):
        return f"文件不存在：{path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 3000:
            content = content[:3000] + "\n...(内容已截断)"
        return content
    except Exception as e:
        return f"读取失败：{e}"


def write_file(path: str, content: str) -> str:
    """写入内容到文件"""
    try:
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"文件已写入：{path}（{len(content)} 个字符）"
    except Exception as e:
        return f"写入失败：{e}"


def web_search(query: str, max_results: int = 5) -> str:
    """用 DuckDuckGo 搜索网页"""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"- {r['title']}\n  {r['href']}")
        if not results:
            return "没有搜索到相关结果。"
        return "\n\n".join(results)
    except Exception as e:
        return f"搜索失败：{e}"


def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    try:
        import requests
        url = f"https://wttr.in/{city}?format=j1"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        current = data["current_condition"][0]
        weather_desc = current["weatherDesc"][0]["value"]
        temp_c = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        humidity = current["humidity"]
        wind_speed = current["windspeedKmph"]

        forecast = data["weather"][0]
        today_max = forecast["maxtempC"]
        today_min = forecast["mintempC"]
        tomorrow = data["weather"][1]
        tomorrow_max = tomorrow["maxtempC"]
        tomorrow_min = tomorrow["mintempC"]
        tomorrow_desc = tomorrow["hourly"][4]["weatherDesc"][0]["value"]

        return f"""城市：{city}
当前天气：{weather_desc}
当前温度：{temp_c}°C（体感 {feels_like}°C）
湿度：{humidity}%
风速：{wind_speed} km/h
今日温度范围：{today_min}°C ~ {today_max}°C
明日：{tomorrow_desc}，{tomorrow_min}°C ~ {tomorrow_max}°C"""
    except Exception as e:
        return f"查询天气失败：{e}。请检查城市名是否正确（如：Shanghai、Beijing）"


# ============================================================
# RAG 引擎：文档检索
# ============================================================

DOCS_DIR = os.path.join(os.path.dirname(__file__), "document_library")
os.makedirs(DOCS_DIR, exist_ok=True)


class RAGEngine:
    """文档检索引擎"""

    def __init__(self):
        self.chunks = []
        self.chunk_size = 200

    def load_file(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return f"文件不存在：{filepath}"
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
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
            return f"已加载文件：{filepath}，共切成 {len(self.chunks)} 个段落"
        except Exception as e:
            return f"加载失败：{e}"

    def search(self, query: str, top_k: int = 3) -> str:
        if not self.chunks:
            return "没有已加载的文档。请先用 load_document 加载一个文件。"
        q = query.lower()
        scored = []
        for i, chunk in enumerate(self.chunks):
            chunk_lower = chunk.lower()
            score = 0
            score += chunk_lower.count(q) * 5
            for j in range(len(q) - 1):
                bigram = q[j:j+2]
                if bigram in chunk_lower:
                    score += 1
            for word in q.split():
                if word in chunk_lower:
                    score += 1
            scored.append((score, i, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = scored[:top_k]
        result = []
        for score, idx, chunk in top_chunks:
            if score > 0:
                result.append(f"[段落 {idx+1}，相关度: {score}]\n{chunk}")
        if not result:
            return "没有找到相关内容。试试换个关键词。"
        return "\n\n---\n\n".join(result)


rag = RAGEngine()

# 放一篇示例文章
sample_text = """# Python 与人工智能导论

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

sample_path = os.path.join(DOCS_DIR, "AI入门简介.txt")
if not os.path.exists(sample_path):
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(sample_text)


def load_document(filepath: str) -> str:
    return rag.load_file(filepath)


def search_document(query: str) -> str:
    return rag.search(query)


# ============================================================
# 工具注册表
# ============================================================

ALL_TOOLS = {
    "get_weather": {
        "func": get_weather,
        "description": "查询指定城市的实时天气和未来预报。当用户问天气、温度、是否下雨时必须用此工具。城市名用英文（如 Shanghai、Beijing）。",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市英文名，如 Shanghai"}},
            "required": ["city"],
        },
    },
    "load_document": {
        "func": load_document,
        "description": "加载一个文档到 RAG 引擎。用户想阅读某篇文档时使用。",
        "parameters": {
            "type": "object",
            "properties": {"filepath": {"type": "string", "description": "文档的完整路径"}},
            "required": ["filepath"],
        },
    },
    "search_document": {
        "func": search_document,
        "description": "在已加载的文档中搜索相关内容。回答关于文档的问题前必须先搜索。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索关键词或问题"}},
            "required": ["query"],
        },
    },
    "run_python_code": {
        "func": run_python_code,
        "description": "执行 Python 代码并返回结果。用于计算、处理数据。",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "要执行的 Python 代码"}},
            "required": ["code"],
        },
    },
    "get_current_time": {
        "func": get_current_time,
        "description": "获取当前日期和时间。",
        "parameters": {"type": "object", "properties": {}},
    },
    "read_file": {
        "func": read_file,
        "description": "读取电脑上的文件内容。",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "文件路径"}},
            "required": ["path"],
        },
    },
    "write_file": {
        "func": write_file,
        "description": "将内容写入文件。用户要保存、记录时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "要写入的内容"},
            },
            "required": ["path", "content"],
        },
    },
    "web_search": {
        "func": web_search,
        "description": "搜索网页获取最新信息。需要查新闻、资料时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "max_results": {"type": "integer", "description": "最多返回几条结果，默认5"},
            },
            "required": ["query"],
        },
    },
}


# ============================================================
# Agent 类（内联版本，不依赖外部文件）
# ============================================================

class Agent:
    def __init__(self, system_prompt: str = ""):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.messages = [{"role": "system", "content": system_prompt or "你是有用的助手。"}]
        self.tools = []
        self.tool_map = {}

    def add_tool(self, name: str, func, description: str, parameters: dict):
        self.tools.append({
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        })
        self.tool_map[name] = func

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            tools=self.tools if self.tools else None,
        )
        reply = response.choices[0].message

        if reply.tool_calls:
            for tool_call in reply.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                tool_result = self.tool_map[func_name](**func_args)

                print(f"  [工具: {func_name}({func_args})]")

                self.messages.append({
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [{
                        "id": tool_call.id, "type": "function",
                        "function": {"name": func_name, "arguments": tool_call.function.arguments},
                    }],
                })
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result),
                })

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages,
            )
            reply = response.choices[0].message

        self.messages.append({"role": "assistant", "content": reply.content})
        return reply.content

    def clear_memory(self):
        self.messages = [self.messages[0]]


# ============================================================
# Gradio 网页界面
# ============================================================

SYSTEM_PROMPT = f"""你是一个全能的 AI 助手。你可以：
- 运行 Python 代码做计算
- 查看当前日期时间
- 读写文件（工作目录：{WORK_DIR}）
- 搜索网页获取最新信息
- 查询任意城市的天气（get_weather）
- 加载文档并用 RAG 检索回答（load_document + search_document）
用中文回答，语气友好。

文档库目录：{DOCS_DIR}，里面有一篇 AI 入门简介可以加载测试。"""


def chat_with_ai(user_message, history):
    """Gradio 回调：每次用户发消息，返回 AI 回复"""
    if not hasattr(chat_with_ai, "agent"):
        chat_with_ai.agent = Agent(system_prompt=SYSTEM_PROMPT)
        for name, info in ALL_TOOLS.items():
            chat_with_ai.agent.add_tool(
                name, info["func"], info["description"], info["parameters"]
            )
    try:
        return chat_with_ai.agent.chat(user_message)
    except Exception as e:
        return f"出错了：{e}"


if __name__ == "__main__":
    import gradio as gr
    import socket

    # 获取本机局域网 IP
    local_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    print("=" * 50)
    print("个人 AI 助手 正在启动...")
    print(f"电脑访问: http://127.0.0.1:7860")
    print(f"手机访问: http://{local_ip}:7860")
    print("=" * 50)

    ui = gr.ChatInterface(
        fn=chat_with_ai,
        title="个人 AI 助手 - Miya 的作品",
        description="""
        **我能做什么？**
        - 🧮 计算数学题、运行 Python 代码
        - 🕐 查看当前时间
        - 🌤 查询任意城市天气
        - 📁 读写文件
        - 🔍 搜索网页获取最新信息
        - 📚 加载文档，基于原文回答问题（RAG）
        - 💬 记住对话上下文

        **试试问我：** "上海今天天气怎么样" | "加载 AI 入门简介然后问我机器学习有哪些类型" | "最近有什么科技新闻"
        """,
        examples=[
            "帮我算 1 加到 100 是多少",
            "上海今天天气怎么样",
            "搜索一下 Python 最新版本有什么新特性",
            f"加载文档 {sample_path}",
            "帮我写一首关于编程的诗，保存到 poem.txt",
        ],
    )

    ui.launch()
