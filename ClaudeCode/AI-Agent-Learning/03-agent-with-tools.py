"""
Agent 初体验 —— 让 AI 能运行 Python 代码
===========================================
核心概念 Function Calling：
  1. 告诉 AI "你有一个工具可以用"
  2. AI 判断需要用工具时，会"请求调用"它
  3. 我们帮 AI 执行工具，把结果返回
  4. AI 看到结果后，组织语言回答用户

这整个循环就叫 Agent Loop（智能体循环）
"""
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv
import json
import io

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ============ 第1步：定义一个工具（函数） ============
# 这个函数真的会执行 Python 代码，并把结果返回


def run_python_code(code: str) -> str:
    """执行 Python 代码并返回结果"""
    try:
        # 用 StringIO 拦截 print 的输出
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured

        local_vars = {}
        exec(code, {}, local_vars)
        sys.stdout = old_stdout  # 恢复标准输出

        output = captured.getvalue().strip()
        if output:
            return output
        if local_vars:
            return str(list(local_vars.values())[-1])
        return "代码执行完成（无输出）"
    except Exception as e:
        sys.stdout = old_stdout  # 出错也要恢复！
        return f"执行出错：{e}"


# ============ 第2步：把工具"注册"给 DeepSeek ============
# 用 JSON 格式描述这个工具是干嘛的、接受什么参数
tools = [
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "执行一段 Python 代码并返回结果。当你需要计算、处理数据时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要执行的 Python 代码，比如 'sum(range(1, 101))'",
                    }
                },
                "required": ["code"],
            },
        },
    }
]

# 工具名 → 真实函数的映射表
tool_map = {"run_python_code": run_python_code}

# ============ 第3步：Agent 主循环 ============
messages = [
    {"role": "system", "content": "你是一个能执行Python代码的助手。遇到需要计算的问题时，调用 run_python_code 工具。最后用中文回答用户。"},
]

print("=" * 50)
print("AI Agent 聊天室 —— 我能帮你算数学、处理数据！")
print("试试说：'1加到100是多少' 或 '帮我算 3的100次方'")
print("输入 'quit' 退出")
print("=" * 50)

while True:
    user_input = input("\n你: ")
    if user_input.lower() == "quit":
        print("再见！")
        break

    messages.append({"role": "user", "content": user_input})

    # 第1次调用：AI 可能选择直接回答，也可能选择调用工具
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
    )

    reply = response.choices[0].message

    # ---------- 关键：检查 AI 是否想调用工具 ----------
    if reply.tool_calls:
        # AI 说"我想用工具！"
        for tool_call in reply.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"  [Agent 调用工具: {func_name}({func_args})]")

            # 真正执行这个工具函数
            tool_result = tool_map[func_name](**func_args)
            print(f"  [工具返回: {tool_result[:60]}...]")

            # 把工具调用和结果都加入记忆
            messages.append(
                {
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

        # 第2次调用：把工具结果发给 AI，让它总结
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
        )
        reply = response.choices[0].message

    # ---------- 打印最终回复 ----------
    print(f"AI: {reply.content}")

    # 把 AI 的最终回复记入历史
    messages.append({"role": "assistant", "content": reply.content})
