"""
天气 Agent —— 查真正的天气数据！
===================================
新工具：get_weather —— 调用 wttr.in 免费天气 API
不需要 API Key，直接 HTTP 请求就能拿到任何城市的天气。

今天学的新东西：
  1. requests 发 HTTP 请求（跟浏览器的工作原理一样）
  2. JSON 解析（API 返回的数据格式）
  3. 异常处理（网络不好时优雅降级）
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

# ============================================================
# 新工具：查天气
# ============================================================

def get_weather(city: str) -> str:
    """
    查询指定城市的天气。
    使用 wttr.in 免费 API，不需要 API Key！
    """
    try:
        import requests

        # wttr.in 的 API：加 ?format=j1 返回 JSON 格式数据
        url = f"https://wttr.in/{city}?format=j1"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # 从返回的 JSON 里提取关键信息
        current = data["current_condition"][0]
        weather_desc = current["weatherDesc"][0]["value"]
        temp_c = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        humidity = current["humidity"]
        wind_speed = current["windspeedKmph"]

        # 获取今明两天的天气预报
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
        return f"查询天气失败：{e}。请检查城市名是否正确（如：Shanghai、Beijing、Tokyo）"


# ============================================================
# 保留之前的全部 5 个工具
# ============================================================

def run_python_code(code: str) -> str:
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
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_file(path: str) -> str:
    if not os.path.exists(path):
        return f"文件不存在：{path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return content[:3000] + "\n...(内容已截断)" if len(content) > 3000 else content
    except Exception as e:
        return f"读取失败：{e}"


def write_file(path: str, content: str) -> str:
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
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"- {r['title']}\n  {r['href']}")
        return "\n\n".join(results) if results else "没有搜索到相关结果。"
    except Exception as e:
        return f"搜索失败：{e}"


# ============================================================
# 工具注册表（新增了 get_weather）
# ============================================================

ALL_TOOLS = {
    "get_weather": {
        "func": get_weather,
        "description": "查询指定城市的实时天气和未来预报。当用户问天气、温度、是否下雨时使用。城市名用英文（如 Shanghai、Beijing）。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市英文名，如 Shanghai、Beijing、Tokyo"}
            },
            "required": ["city"],
        },
    },
    "run_python_code": {
        "func": run_python_code,
        "description": "执行 Python 代码计算。",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python 代码"}},
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
        "description": "读取文件内容。",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "文件路径"}},
            "required": ["path"],
        },
    },
    "write_file": {
        "func": write_file,
        "description": "写入内容到文件。",
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
        "description": "搜索网页获取信息。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "max_results": {"type": "integer", "description": "最多返回几条，默认5"},
            },
            "required": ["query"],
        },
    },
}


# ============================================================
# Agent 类
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

    def add_tool(self, name, func, description, parameters):
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
                    "role": "tool", "tool_call_id": tool_call.id,
                    "content": str(tool_result),
                })

            response = self.client.chat.completions.create(
                model="deepseek-chat", messages=self.messages,
            )
            reply = response.choices[0].message

        self.messages.append({"role": "assistant", "content": reply.content})
        return reply.content


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    agent = Agent(system_prompt="你是一个 AI 助手，能查天气、做计算、搜网页、读写文件、看时间。用中文回答，语气友好。当用户问天气时必须用 get_weather 工具。")

    for name, info in ALL_TOOLS.items():
        agent.add_tool(name, info["func"], info["description"], info["parameters"])

    print("=" * 50)
    print("天气 Agent —— 问我天气！")
    print("试试：'上海今天天气怎么样'、'北京明天会下雨吗'")
    print("输入 'quit' 退出")
    print("=" * 50)

    while True:
        user_input = input("\n你: ")
        if user_input.lower() == "quit":
            print("再见！")
            break
        reply = agent.chat(user_input)
        print(f"AI: {reply}")
