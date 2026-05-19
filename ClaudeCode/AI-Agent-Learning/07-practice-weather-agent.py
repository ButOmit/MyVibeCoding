"""
个人 AI 助手 —— 带网页界面的全能 Agent
=========================================
整合了全部 5 个工具，用 Gradio 做了网页界面。
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
# 5 个工具函数
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
        return f"出错了：{e}"

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


# ============================================================
# 工具注册表
# ============================================================

ALL_TOOLS = {
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
    "get_weather":{
        "func":get_weather,
        "description": "查询指定城市的实时天气和未来预报。当用户问天气、温度、是否下雨时必须用此工具。城市名用英文（如 Shanghai、Beijing）。",
        "parameters":{
            "type":"object",
            "properties": {"city": {"type": "string", "description": "城市英文名，如 Shanghai"}},
            "required":["city"]
        }
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
用中文回答，语气友好。"""


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

    print("=" * 50)
    print("个人 AI 助手 正在启动...")
    print("浏览器打开显示的地址就能用，手机也能访问")
    print("=" * 50)

    ui = gr.ChatInterface(
        fn=chat_with_ai,
        title="个人 AI 助手 - Miya 的作品",
        description="""
        **我能做什么？**
        - 🧮 计算数学题、运行 Python 代码
        - 🕐 查看当前时间
        - 📁 读写文件
        - 🔍 搜索网页获取最新信息
        - 💬 记住对话上下文

        **试试问我：** "帮我算 1 到 100 的和" | "现在几点了" | "最近有什么科技新闻"
        """,
        examples=[
            "帮我算 1 加到 100 是多少",
            "现在几点了",
            "搜索一下 Python 最新版本有什么新特性",
            "帮我写一首关于编程的诗，保存到 poem.txt",
        ],
    )

    ui.launch(share=False)
