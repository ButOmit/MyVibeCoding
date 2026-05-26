"""
多 Agent 协作 —— AI 团队作战
================================
之前所有项目都是单 Agent：一个 AI 包揽所有事。
现在引入团队分工：不同 Agent 有不同的角色和工具，协作完成任务。

团队架构（3人小组）：
  ┌──────────────┐
  │   协调员      │ ← 分析用户需求，分配任务给合适的专家
  │ Coordinator  │
  └──┬────────┬──┘
     │        │
     ▼        ▼
  ┌──────┐ ┌──────┐
  │ 研究员 │ │ 程序员 │  ← 各有各的工具，各司其职
  │Researcher│ │ Coder  │
  └──────┘ └──────┘

和单 Agent 的区别：
  单 Agent: 一个模型 → 全部工具 → 一次性回答
  多 Agent: 协调员分任务 → 研究员查资料 → 程序员写代码 → 协调员汇总

真实世界的应用：
  - AutoGPT/BabyAGI: Planner + Executor
  - ChatGPT Code Interpreter: 对话 Agent + 代码 Agent
  - 客服系统: 路由 Agent + 专业领域 Agent
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
# 工具函数（有的工具只有特定 Agent 能用）
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
        return "代码执行完成"
    except Exception as e:
        sys.stdout = old_stdout
        return f"执行出错：{e}"


def get_current_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# 工具注册表
# ============================================================

TOOLS = {
    "run_python_code": {
        "func": run_python_code,
        "def": {
            "type": "function",
            "function": {
                "name": "run_python_code",
                "description": "执行 Python 代码。需要计算、处理数据时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python 代码"},
                    },
                    "required": ["code"],
                },
            },
        },
    },
    "get_current_time": {
        "func": get_current_time,
        "def": {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "获取当前的日期和时间。",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    },
}


# ============================================================
# SpecialistAgent —— 有专属角色和工具
# ============================================================

class SpecialistAgent:
    """
    专业 Agent：有自己的角色定位和专属工具。

    和之前 Agent 类的区别：
      - 每个 Agent 有不同的 system prompt（定义了它的角色）
      - 每个 Agent 有不同的工具（程序员工具有代码执行权限，研究员没有）
    """

    def __init__(self, name: str, role: str, tool_names: list[str] = None):
        self.name = name
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.messages = [{"role": "system", "content": role}]

        # 只给这个 Agent 分配指定的工具
        self.tool_defs = []
        self.tool_map = {}
        for tname in (tool_names or []):
            if tname in TOOLS:
                self.tool_defs.append(TOOLS[tname]["def"])
                self.tool_map[tname] = TOOLS[tname]["func"]

    def ask(self, task: str) -> str:
        """
        给这个 Agent 一个任务，让它独立完成（包括用工具）。
        返回完成结果。
        """
        self.messages.append({"role": "user", "content": task})

        tools = self.tool_defs if self.tool_defs else None

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            tools=tools,
        )
        reply = response.choices[0].message

        if reply.tool_calls:
            for tool_call in reply.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                tool_result = self.tool_map[func_name](**func_args)

                print(f"    [{self.name} 调用工具: {func_name}({func_args})]")

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


# ============================================================
# MultiAgentTeam —— 协调多个 Agent 协作
# ============================================================

class MultiAgentTeam:
    """
    多 Agent 协作引擎。

    工作流程：
      1. 协调员分析用户需求 → 拆解成子任务
      2. 把每个子任务派给最合适的 Agent
      3. 收集所有 Agent 的结果，汇总回答
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

        # 研究员：能查时间，负责信息收集
        self.researcher = SpecialistAgent(
            name="研究员",
            role="你是信息研究员，负责查找事实、收集数据、分析信息。用中文回答，简洁准确。",
            tool_names=["get_current_time"],
        )

        # 程序员：能写代码，负责计算任务
        self.coder = SpecialistAgent(
            name="程序员",
            role="你是程序员，负责写 Python 代码解决计算问题。用中文回答，代码要正确可运行。",
            tool_names=["run_python_code", "get_current_time"],
        )

        self.coordinator_messages = [
            {"role": "system", "content": """你是团队协调员。管理一个AI团队：

团队成员：
- 研究员：负责查找信息、收集数据
- 程序员：负责编写Python代码、执行计算

你的工作：
1. 分析用户的问题，拆解成子任务
2. 决定哪个子任务给哪个成员
3. 收集所有成员的成果，汇总成完整回答

输出格式：
【任务拆解】简单说明需要几个步骤
然后调用 ask_researcher 和 ask_coder 来派发任务。
用中文回答，语气像一个项目经理。"""},
        ]

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "ask_researcher",
                    "description": "让研究员去查信息/数据。传入一个具体的问题，研究员会返回答案。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string", "description": "给研究员的具体任务"},
                        },
                        "required": ["task"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_coder",
                    "description": "让程序员去写代码/做计算。传入一个具体的编程任务，程序员会返回代码执行结果。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string", "description": "给程序员的具体任务"},
                        },
                        "required": ["task"],
                    },
                },
            },
        ]

    def handle(self, user_input: str):
        """
        处理用户请求：
        1. 协调员分析需求
        2. 派发任务给专家
        3. 汇总结果
        """
        print(f"\n{'=' * 55}")
        print(f"用户请求: {user_input}")
        print(f"{'=' * 55}")

        self.coordinator_messages.append({"role": "user", "content": user_input})

        # 协调员决定派谁干活
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=self.coordinator_messages,
            tools=self.tools,
        )
        reply = response.choices[0].message

        if reply.tool_calls:
            self.coordinator_messages.append({
                "role": "assistant",
                "content": reply.content or "",
                "tool_calls": [
                    {
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in reply.tool_calls
                ],
            })

            for tool_call in reply.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                task = func_args["task"]

                # 协调员把任务派给对应的专家
                if func_name == "ask_researcher":
                    print(f"\n  >>> 协调员 → 研究员: \"{task}\"")
                    result = self.researcher.ask(task)
                    print(f"  <<< 研究员回复: {result[:100]}...")

                elif func_name == "ask_coder":
                    print(f"\n  >>> 协调员 → 程序员: \"{task}\"")
                    result = self.coder.ask(task)
                    print(f"  <<< 程序员回复: {result[:100]}...")

                else:
                    result = "未知任务"

                self.coordinator_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # 协调员汇总所有专家的结果
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=self.coordinator_messages,
            )
            reply = response.choices[0].message

        final_answer = reply.content
        self.coordinator_messages.append({"role": "assistant", "content": final_answer})

        print(f"\n{'=' * 55}")
        print("最终回答:")
        print(f"{'=' * 55}")

        return final_answer


# ============================================================
# 对比：单 Agent vs 多 Agent
# ============================================================

def compare_single_vs_multi():
    """
    演示同一个问题，单 Agent 和多 Agent 的区别。

    单 Agent: 一个模型，所有工具，一次性处理
    多 Agent: 协调员分配 → 专家各司其职 → 汇总
    """
    print("=" * 60)
    print("单 Agent vs 多 Agent 对比")
    print("=" * 60)

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    question = "现在几点了？计算 2026年还剩下百分之多少的时间"

    # --- 单 Agent 模式 ---
    print("\n【单 Agent 模式】")
    print("-" * 40)
    print("一个 Agent 拿着所有工具，一次性收到全部指令...")

    single_messages = [
        {"role": "system", "content": "你是全能助手。用中文回答。用工具获取数据和计算。"},
        {"role": "user", "content": question},
    ]

    single_tools = [TOOLS["get_current_time"]["def"], TOOLS["run_python_code"]["def"]]
    single_map = {
        "get_current_time": TOOLS["get_current_time"]["func"],
        "run_python_code": TOOLS["run_python_code"]["func"],
    }

    response = client.chat.completions.create(
        model="deepseek-chat", messages=single_messages, tools=single_tools,
    )
    reply = response.choices[0].message

    if reply.tool_calls:
        for tc in reply.tool_calls:
            args = json.loads(tc.function.arguments)
            result = single_map[tc.function.name](**args)
            print(f"  [单Agent调用: {tc.function.name}({args}) → {str(result)[:60]}]")

            single_messages.append({
                "role": "assistant", "content": reply.content or "",
                "tool_calls": [{
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }],
            })
            single_messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": str(result),
            })

        response = client.chat.completions.create(
            model="deepseek-chat", messages=single_messages,
        )
        reply = response.choices[0].message

    print(f"  单Agent结果: {reply.content}")

    # --- 多 Agent 模式 ---
    print("\n【多 Agent 模式】")
    print("-" * 40)
    print("协调员分析需求 → 分派给研究员和程序员 → 汇总")

    team = MultiAgentTeam()
    final = team.handle(question)

    print(f"\n{'─' * 60}")
    print("区别总结:")
    print("  单Agent: 一个大脑 → 所有工具 → 做出所有决定")
    print("  多Agent: 协调员分派 → 专家专注 → 结果更可靠")
    print("  多Agent 优势: 每个 Agent 只关注自己的领域，不会分心")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 先演示对比
    compare_single_vs_multi()

    # 进入交互模式
    print(f"\n{'=' * 60}")
    print("多 Agent 协作团队")
    print("试试复杂问题：")
    print("  '帮我算一下，如果从1900年到现在一共过了多少天'")
    print("  '分析一下当前时间点，距离今年结束还有多少个工作日'")
    print("  '计算圆周率的前20位，并告诉我现在几点了'")
    print("输入 'quit' 退出")
    print(f"{'=' * 60}")

    team = MultiAgentTeam()

    while True:
        user_input = input("\n你: ")
        if user_input.lower() == "quit":
            print("团队散会！再见！")
            break
        answer = team.handle(user_input)
        print(f"协调员: {answer}")
