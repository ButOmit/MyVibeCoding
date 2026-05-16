"""
Agent 框架 —— 整洁、可扩展、简历能拿得出手
=============================================
用类和字典把 Agent 的核心逻辑封装好。
以后再写新工具，只需 3 步：
  1. 写一个函数（做什么）
  2. 写工具描述（告诉 AI 这个工具怎么用）
  3. agent.add_tool(函数, 描述)
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
# 第1层：工具定义
# ============================================================

def run_python_code(code: str) -> str:
    """执行 Python 代码并返回结果"""
    try:
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured
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


# ============================================================
# 第2层：工具注册表（所有工具在这里登记）
# ============================================================

TOOL_REGISTRY = {
    "run_python_code": {
        "func": run_python_code,
        "description": "执行一段 Python 代码并返回结果。需要计算、处理数据时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码",
                }
            },
            "required": ["code"],
        },
    },
    "get_current_time": {
        "func": get_current_time,
        "description": "获取当前的日期和时间。用户问时间时使用。",
        "parameters": {"type": "object", "properties": {}},
    },
}


# ============================================================
# 第3层：Agent 类（核心引擎）
# ============================================================

class Agent:
    """AI Agent —— 能用工具、有记忆、可持续对话"""

    def __init__(self, system_prompt: str = ""):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.messages = [
            {"role": "system", "content": system_prompt or "你是一个有用的助手，能用工具帮助用户。用中文回答。"},
        ]
        self.tools = []       # 给 API 看的工具列表
        self.tool_map = {}    # 工具名 → 真实函数

    def add_tool(self, name: str, func, description: str, parameters: dict):
        """注册一个新工具。只需调用这个函数，Agent 就能用新工具了！"""
        self.tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        })
        self.tool_map[name] = func

    def chat(self, user_input: str) -> str:
        """跟 Agent 说一句话，返回回复"""
        self.messages.append({"role": "user", "content": user_input})

        # 第1次调用：AI 可能直接回答，也可能调用工具
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            tools=self.tools if self.tools else None,
        )
        reply = response.choices[0].message

        # 如果 AI 想用工具
        if reply.tool_calls:
            for tool_call in reply.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                tool_result = self.tool_map[func_name](**func_args)

                print(f"  [Agent 调用: {func_name}({func_args})]")
                print(f"  [返回: {str(tool_result)[:60]}]")

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
                    "content": str(tool_result),
                })

            # 第2次调用：把工具结果发给 AI 做总结
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages,
            )
            reply = response.choices[0].message

        self.messages.append({"role": "assistant", "content": reply.content})
        return reply.content

    def clear_memory(self):
        """清空对话记忆，只保留 system 消息"""
        self.messages = [self.messages[0]]


# ============================================================
# 第4层：主程序 —— 只要几行就能启动一个 Agent
# ============================================================

if __name__ == "__main__":
    agent = Agent()

    # 从注册表加载所有工具
    for name, info in TOOL_REGISTRY.items():
        agent.add_tool(name, info["func"], info["description"], info["parameters"])

    print("=" * 50)
    print("Agent 框架 v1.0 —— 输入 'quit' 退出，'clear' 清空记忆")
    print("试试：'帮我算 2的10次方' 或 '现在几点了'")
    print("=" * 50)

    while True:
        user_input = input("\n你: ")
        if user_input.lower() == "quit":
            print("再见！")
            break
        if user_input.lower() == "clear":
            agent.clear_memory()
            print("(记忆已清空)")
            continue
        reply = agent.chat(user_input)
        print(f"AI: {reply}")
