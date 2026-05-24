"""
流式输出 Agent —— 让 AI 像真人一样"打字"
===========================================
核心区别只有一行：

  非流式: stream=False（默认） → 等 AI 全部写完，一次性返回
  流式:   stream=True          → AI 写一个字就发一个字，逐字打印

就像看直播 vs 看录播：
  - 非流式＝录播，视频全传完才能看
  - 流式＝直播，实时看到每一帧

架构：
  工具调用阶段 → 非流式（内部处理，用户不用看过程）
  最终回复阶段 → 流式输出（用户看到打字效果）
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


# ============================================================
# 工具注册表
# ============================================================

TOOL_REGISTRY = {
    "run_python_code": {
        "func": run_python_code,
        "description": "执行一段 Python 代码并返回结果。需要计算、处理数据时使用。",
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
}


# ============================================================
# 流式 Agent 类
# ============================================================

class StreamingAgent:
    """
    流式 Agent：
      - 工具调用阶段：非流式（后台处理，用户看不见）
      - 最终回复阶段：流式输出（逐字打印，打字机效果）
    """

    def __init__(self, system_prompt: str = ""):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.messages = [
            {"role": "system", "content": system_prompt or "你是有用的AI助手，用中文回答，语气友好。"},
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
        流式对话：AI 边想边打字，一个字一个字蹦出来。

        生成器函数（generator）—— 用 yield 代替 return，
        每生成一个字就立刻抛给调用方，不用等全部写完。
        """
        self.messages.append({"role": "user", "content": user_input})

        # ============================================================
        # 第1步：非流式调用，检测 AI 是否想用工具
        # （工具调用在后台完成，不需要流式——用户也看不懂 JSON）
        # ============================================================
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

                print(f"  [工具调用: {func_name}({func_args})]")

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

        # ============================================================
        # 第2步：流式输出最终回复
        # 把 stream 打开 → 遍历 chunks → 逐字打印
        # ============================================================
        stream = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            stream=True,  # ← 就这一个参数！True = 流式，False = 非流式
        )

        full_content = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_content += delta.content
                # yield 出去，调用方可以逐字处理（打印、存起来、发网页等）
                yield delta.content

        self.messages.append({"role": "assistant", "content": full_content})


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    agent = StreamingAgent()

    for name, info in TOOL_REGISTRY.items():
        agent.add_tool(name, info["func"], info["description"], info["parameters"])

    print("=" * 50)
    print("流式输出 Agent（打字机效果）")
    print("对比：stream=False → 一股脑全出来")
    print("      stream=True  → 逐字蹦出来")
    print("试试：'帮我算 1 加到 100'  '现在几点了'  'quit' 退出")
    print("=" * 50)

    while True:
        user_input = input("\n你: ")
        if user_input.lower() == "quit":
            print("再见！")
            break

        print("AI: ", end="", flush=True)
        for text_chunk in agent.chat_stream(user_input):
            print(text_chunk, end="", flush=True)
        print()  # 最后换个行
