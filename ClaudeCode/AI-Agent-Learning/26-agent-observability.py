"""
Agent 可观测性 —— 生产环境的"监控摄像头"
============================================
之前你的 Agent 只有 print()，上线后完全不知道：
  - 每个请求花了多长时间？
  - 哪个工具调用最多？
  - Token 消耗了多少？还剩多少预算？
  - 为什么某次回答特别慢？

可观测性三大支柱（面试必备概念）：
  1. 日志（Logging）    — 发生了什么
  2. 指标（Metrics）    — 量化数据（耗时、Token数）
  3. 追踪（Tracing）    — 一次请求的完整调用链

架构：
  Agent 请求
    ├── [可观测层] 记录开始时间、用户输入
    ├── LLM 调用   → 记录 Token 消耗、耗时
    ├── 工具调用    → 记录工具名、参数、结果
    └── [可观测层] 汇总统计数据
"""

import sys
import os
import io
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional
from contextlib import contextmanager
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 第一层：结构化日志 —— 替代 print()
# ============================================================

# 生产环境不用 print()，用结构化日志：
#   每条日志 = 时间 + 级别 + 模块 + 消息 + 上下文数据
#   可以输出到文件、发送到监控平台、按级别过滤

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent")


# ============================================================
# 第二层：指标收集器
# ============================================================

class AgentMetrics:
    """收集 Agent 运行的所有量化数据"""

    def __init__(self):
        self.total_requests = 0
        self.total_tokens = 0
        self.total_tool_calls = 0
        self.total_errors = 0

        # 每次请求的耗时（用于计算平均值）
        self.response_times: list[float] = []

        # 工具调用统计
        self.tool_usage: dict[str, int] = defaultdict(int)  # 工具名 → 调用次数

        # 按小时的请求分布
        self.hourly_requests: dict[int, int] = defaultdict(int)

        # Token 消耗历史（用于画趋势图）
        self.token_history: list[tuple[str, int]] = []

    def record_request(self, duration: float, tokens: int, tool_calls: int):
        self.total_requests += 1
        self.response_times.append(duration)
        self.total_tokens += tokens
        self.total_tool_calls += tool_calls

        hour = datetime.now().hour
        self.hourly_requests[hour] += 1
        self.token_history.append((
            datetime.now().strftime("%H:%M"),
            tokens,
        ))

    def record_tool(self, tool_name: str):
        self.tool_usage[tool_name] += 1

    def record_error(self):
        self.total_errors += 1

    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0
        return sum(self.response_times) / len(self.response_times)

    @property
    def p95_response_time(self) -> float:
        """95分位延迟：95%的请求在此时间内完成"""
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]


# ============================================================
# 第三层：调用追踪器
# ============================================================

class RequestTracer:
    """追踪一次请求的完整调用链"""

    def __init__(self, request_id: str, user_input: str):
        self.request_id = request_id
        self.user_input = user_input[:100]
        self.start_time = time.time()
        self.spans: list[dict] = []  # 每个步骤的记录

        logger.info(f"[{request_id}] 请求开始: {self.user_input}")

    @contextmanager
    def span(self, name: str):
        """追踪一个步骤：记录开始、结束、耗时。

        用法:
          with tracer.span("LLM调用"):
              response = client.chat(...)
        """
        start = time.time()
        span_data = {"name": name, "start": start}
        try:
            yield span_data
        except Exception as e:
            span_data["error"] = str(e)
            raise
        finally:
            span_data["duration"] = time.time() - start
            self.spans.append(span_data)
            logger.info(
                f"[{self.request_id}] {name} "
                f"耗时 {span_data['duration']:.2f}s"
            )

    def finish(self, tokens_used: int):
        total_time = time.time() - self.start_time
        logger.info(
            f"[{self.request_id}] 请求完成 "
            f"总耗时 {total_time:.2f}s | Token {tokens_used} | 步骤 {len(self.spans)}"
        )

    def summary(self) -> str:
        """生成调用链摘要"""
        total = time.time() - self.start_time
        lines = [
            f"请求 {self.request_id} 调用链:",
            f"  用户: {self.user_input}",
            f"  总耗时: {total:.2f}s",
        ]
        for s in self.spans:
            err = f" [错误: {s.get('error')}]" if 'error' in s else ""
            lines.append(f"  └─ {s['name']}: {s['duration']:.2f}s{err}")
        return "\n".join(lines)


# ============================================================
# 第四层：可观测 Agent
# ============================================================

class ObservableAgent:
    """带完整可观测性的 Agent"""

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.metrics = AgentMetrics()
        self._req_counter = 0

        # 工具定义
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "获取当前日期和时间",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_python_code",
                    "description": "执行 Python 代码",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python 代码"},
                        },
                        "required": ["code"],
                    },
                },
            },
        ]

    def _get_time(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _run_code(self, code: str) -> str:
        old = sys.stdout
        cap = io.StringIO()
        sys.stdout = cap
        try:
            local_vars = {}
            exec(code, {}, local_vars)
            sys.stdout = old
            out = cap.getvalue().strip()
            return out or str(list(local_vars.values())[-1]) if local_vars else "执行完成"
        except Exception as e:
            sys.stdout = old
            return f"出错：{e}"

    def chat(self, user_message: str) -> str:
        """带可观测性的对话"""
        self._req_counter += 1
        req_id = f"req-{self._req_counter:04d}"

        # 创建追踪器
        tracer = RequestTracer(req_id, user_message)

        messages = [
            {"role": "system", "content": "你是助手，用中文回答，简洁清晰。"},
            {"role": "user", "content": user_message},
        ]

        token_used = 0
        tool_calls_count = 0

        # ---- LLM 调用 ----
        with tracer.span("LLM 调用（首次）"):
            response = self.client.chat.completions.create(
                model="deepseek-chat", messages=messages, tools=self.tools,
            )
            reply = response.choices[0].message
            token_used += response.usage.total_tokens if response.usage else 0

        # ---- 工具调用（如果有） ----
        if reply.tool_calls:
            with tracer.span(f"工具调用（{len(reply.tool_calls)}个）"):
                for tc in reply.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)

                    self.metrics.record_tool(name)
                    tool_calls_count += 1

                    tool_start = time.time()
                    if name == "get_current_time":
                        result = self._get_time()
                    elif name == "run_python_code":
                        result = self._run_code(args["code"])
                    else:
                        result = "未知工具"
                    tool_time = time.time() - tool_start

                    logger.info(
                        f"[{req_id}] 工具 {name}({args}) "
                        f"→ {str(result)[:50]} ({tool_time:.2f}s)"
                    )

                    messages.append({
                        "role": "assistant",
                        "content": reply.content or "",
                        "tool_calls": [{
                            "id": tc.id, "type": "function",
                            "function": {"name": name, "arguments": tc.function.arguments},
                        }],
                    })
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id, "content": str(result),
                    })

            # ---- 第二次 LLM 调用（汇总工具结果） ----
            with tracer.span("LLM 调用（汇总）"):
                response = self.client.chat.completions.create(
                    model="deepseek-chat", messages=messages,
                )
                reply = response.choices[0].message
                token_used += response.usage.total_tokens if response.usage else 0

        # ---- 记录指标 ----
        total_time = time.time() - tracer.start_time
        self.metrics.record_request(total_time, token_used, tool_calls_count)
        tracer.finish(token_used)

        return reply.content


# ============================================================
# 第五层：监控面板
# ============================================================

def print_dashboard(agent: ObservableAgent):
    """打印可观测性面板"""
    m = agent.metrics

    print("\n" + "=" * 60)
    print("Agent 可观测性面板")
    print("=" * 60)

    print(f"""
┌─────────────────────────────────────────────┐
│  📊 请求统计                                  │
├─────────────────────────────────────────────┤
│  总请求:     {m.total_requests:<5}                            │
│  总 Token:   {m.total_tokens:<5}                            │
│  工具调用:   {m.total_tool_calls:<5}                            │
│  错误数:     {m.total_errors:<5}                            │
├─────────────────────────────────────────────┤
│  ⏱️  延迟                                    │
├─────────────────────────────────────────────┤
│  平均响应:   {m.avg_response_time:.2f}s                          │
│  P95 延迟:   {m.p95_response_time:.2f}s                          │
├─────────────────────────────────────────────┤
│  🔧 工具使用 TOP5                             │""")

    top_tools = sorted(m.tool_usage.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (tool, count) in enumerate(top_tools, 1):
        bar = "█" * min(count, 30)
        print(f"│  {i}. {tool:<20} {count:>4}  {bar}")

    print("""├─────────────────────────────────────────────┤
│  🕐 按小时请求分布                             │""")

    for hour in sorted(m.hourly_requests.keys()):
        count = m.hourly_requests[hour]
        bar = "█" * min(count, 40)
        print(f"│  {hour:02d}:00  {count:>4}  {bar}")

    print("""├─────────────────────────────────────────────┤
│  💰 Token 消耗趋势（最近10条）                  │""")

    for ts, tokens in m.token_history[-10:]:
        bar = "█" * min(tokens // 50, 30)
        print(f"│  {ts}  {tokens:>5}  {bar}")

    print("└─────────────────────────────────────────────┘")


# ============================================================
# 对比：有无可观测性
# ============================================================

def demo():
    print("=" * 60)
    print("Agent 可观测性 —— 生产环境的监控摄像头")
    print("=" * 60)

    print("""
之前你的 Agent：
  用户输入 → ??? → 输出结果
  不知道耗时、Token、调用链 ← 完全不可观测

现在你的 Agent：
  用户输入 → [日志] → [LLM调用 1.2s] → [工具调用 0.3s] → [LLM汇总 0.8s] → 输出结果
  每条都有：时间戳、耗时、Token数、调用链 ← 完全可观测
""")

    agent = ObservableAgent()

    # 模拟一些请求
    queries = [
        "现在几点了？",
        "帮我算一下 1234 * 5678",
        "计算 2 的 20 次方",
        "现在几点？今天星期几？",
        "1加到100等于多少",
    ]

    print("执行测试请求...\n")
    for i, q in enumerate(queries, 1):
        print(f"--- 请求 {i}: {q}")
        try:
            answer = agent.chat(q)
            print(f"    回答: {answer[:80]}...")
        except Exception as e:
            print(f"    出错: {e}")
            agent.metrics.record_error()

    # 打印监控面板
    print_dashboard(agent)

    print("\n" + "=" * 60)
    print("面试话术：")
    print('  "我在 Agent 项目中实施了可观测性，')
    print('   包含结构化日志、调用链追踪、Token 统计和性能指标。')
    print('   用 P95 延迟监控服务质量，按工具统计调用频率，')
    print('   按小时分析请求分布做容量规划。')
    print('   这些是生产环境 Agent 必备的基础设施。\"')


if __name__ == "__main__":
    demo()
