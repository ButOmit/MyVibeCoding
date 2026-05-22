"""
本地模型 Agent —— 不花一分钱的 AI 助手
========================================
Ollama 自带 OpenAI 兼容接口 (http://localhost:11434/v1)
用同一套 openai 库，换个 base_url，就能免费无限调用本地模型！

对比：
  DeepSeek API: api.deepseek.com      → 花钱，需要网络
  Ollama 本地:   localhost:11434/v1    → 免费，断网也能跑
"""
import sys
import os
import json
import io
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI

# ============================================================
# 两个模型客户端
# ============================================================

# 云端 DeepSeek（花钱，需要网络）
from dotenv import load_dotenv
load_dotenv()

deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 本地 Ollama（免费，断网也能跑）
ollama_client = OpenAI(
    api_key="ollama",  # Ollama 不需要真的 key，填啥都行
    base_url="http://localhost:11434/v1",
)

# 当前使用的客户端（可以切换）
client = ollama_client  # ← 改成 deepseek_client 就切回云端
MODEL_NAME = "qwen2.5:3b"  # ← Ollama 模型名
# MODEL_NAME = "deepseek-chat"  # ← 切回 DeepSeek 时用这个

print(f"当前后端: {'Ollama 本地模型' if client == ollama_client else 'DeepSeek 云端'}")
print(f"模型: {MODEL_NAME}")

# ============================================================
# 工具函数
# ============================================================

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


def get_weather(city: str) -> str:
    """查询天气"""
    try:
        import requests
        url = f"https://wttr.in/{city}?format=j1"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        current = data["current_condition"][0]
        wd = current["weatherDesc"][0]["value"]
        return f"{city}: {wd}, {current['temp_C']}°C (体感 {current['FeelsLikeC']}°C)"
    except Exception as e:
        return f"查询失败：{e}"


# ============================================================
# 工具注册表
# ============================================================

def build_tools():
    """构建当前客户端用的工具列表"""
    return [
        {
            "type": "function",
            "function": {
                "name": "run_python_code",
                "description": "执行 Python 代码计算。",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Python 代码"}},
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "获取当前日期和时间。",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询城市天气。",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "城市英文名"}},
                    "required": ["city"],
                },
            },
        },
    ]


tool_map = {
    "run_python_code": run_python_code,
    "get_current_time": get_current_time,
    "get_weather": get_weather,
}


# ============================================================
# Agent 类
# ============================================================

class Agent:
    def __init__(self, system_prompt: str = ""):
        self.messages = [{"role": "system", "content": system_prompt or "你是友好的AI助手。"}]

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        tools = build_tools()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=self.messages,
            tools=tools,
        )
        reply = response.choices[0].message

        if reply.tool_calls:
            for tool_call in reply.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                tool_result = tool_map[func_name](**func_args)

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

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=self.messages,
            )
            reply = response.choices[0].message

        self.messages.append({"role": "assistant", "content": reply.content})
        return reply.content


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    agent = Agent(system_prompt="你是一个 AI 助手。能执行 Python 代码、看时间、查天气。用中文回答，语气友好。")

    print("=" * 50)
    print("本地模型 Agent（Ollama + qwen2.5:3b）")
    print("0 API 费用 | 数据不离开电脑 | 断网也能用")
    print("试试：'现在几点了' | '上海天气怎么样' | '算一下 1+2+...+100'")
    print("输入 'quit' 退出")
    print("=" * 50)

    while True:
        user_input = input("\n你: ")
        if user_input.lower() == "quit":
            print("再见！")
            break
        reply = agent.chat(user_input)
        print(f"AI: {reply}")
