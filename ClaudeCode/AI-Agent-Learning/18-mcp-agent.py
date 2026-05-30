"""
MCP Agent —— 通过 MCP 协议发现和调用工具
============================================
核心架构变化：

  之前的 Agent（项目 03/04/07）:
    ┌──────────────────────┐
    │  Agent 类             │
    │  ├── TOOL_REGISTRY    │ ← 工具定义硬编码在 Agent 里
    │  ├── tool_map         │ ← 工具函数也在 Agent 里
    │  └── chat()           │
    └──────────────────────┘

  MCP Agent（本项目）:
    ┌──────────┐   MCP 协议    ┌────────────────┐
    │  Agent   │ ←──────────→ │  MCP Server     │
    │ (客户端)  │   JSON-RPC   │ (FastMCP)       │
    │          │   over stdio  │  ├── get_time   │
    │ 不包含   │               │  ├── run_code   │
    │ 任何工具  │               │  └── read_file  │
    └──────────┘               └────────────────┘
            工具在外部，Agent 动态发现！

本文件使用原始 JSON-RPC over stdio 通信（不依赖 MCP SDK 的 stdio_client），
这样你能理解 MCP 协议的本质：就是标准的 JSON-RPC 2.0 而已！

运行方式：
  python 18-mcp-agent.py
"""

import sys
import os
import json
import asyncio
import subprocess

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 轻量 MCP 客户端 —— 直接 JSON-RPC over stdio
# ============================================================
# MCP 协议本质：JSON-RPC 2.0 请求/响应，通过 stdin/stdout 传输。
#
# 三个核心方法：
#   1. initialize  → 握手，交换协议版本和能力
#   2. tools/list  → 发现服务器提供了哪些工具
#   3. tools/call  → 调用具体的工具
#
# 之前你用 TOOL_REGISTRY 字典，现在用 JSON-RPC 动态获取！

class MCPClient:
    """轻量 MCP 客户端 —— 通过子进程 + JSON-RPC 和 Server 通信。

    MCP SDK 的 stdio_client 在 Windows 上有兼容问题，
    这里用 subprocess + asyncio 直接实现，也让你看到协议本质。
    """

    def __init__(self, server_script: str):
        self.server_script = server_script
        self.proc: subprocess.Popen = None
        self._request_id = 0

    async def connect(self):
        """启动 MCP Server 子进程，完成初始化握手"""
        self.proc = subprocess.Popen(
            [sys.executable, "-u", self.server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # 初始化握手
        result = await self._rpc_call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mcp-agent", "version": "1.0"},
        })
        server_info = result.get("serverInfo", {})
        print(f"已连接 MCP Server: {server_info.get('name', '?')} v{server_info.get('version', '?')}")

    async def _rpc_call(self, method: str, params: dict = None) -> dict:
        """发送 JSON-RPC 请求，返回响应 result"""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }
        # 发送 → stdin
        req_bytes = (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
        self.proc.stdin.write(req_bytes)
        self.proc.stdin.flush()

        # 接收 → stdout（每次读一行 JSON）
        loop = asyncio.get_event_loop()
        line = await loop.run_in_executor(None, self.proc.stdout.readline)
        response = json.loads(line.decode("utf-8"))

        if "error" in response:
            raise Exception(f"MCP 错误: {response['error']}")
        return response.get("result", {})

    async def list_tools(self) -> list[dict]:
        """🎯 核心：动态发现工具！

        这就是 MCP 和 TOOL_REGISTRY 的本质区别——
        你不需要在 Agent 代码里维护工具列表，Server 会告诉你。
        """
        result = await self._rpc_call("tools/list")
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: dict) -> str:
        """调用工具，返回文本结果"""
        result = await self._rpc_call("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        # MCP 返回 content 数组，提取文本
        content = result.get("content", [])
        if content:
            return content[0].get("text", str(content))
        return str(result)

    async def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.terminate()
            self.proc.wait(timeout=5)


# ============================================================
# MCP Agent —— 通过 MCP 动态发现工具
# ============================================================

class MCPAgent:
    def __init__(self, mcp_client: MCPClient):
        self.ds = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.mcp = mcp_client
        self.tools_for_openai: list[dict] = []

    async def connect(self):
        await self.mcp.connect()

        # 🎯 核心：动态发现工具！
        mcp_tools = await self.mcp.list_tools()
        print(f"MCP Server 提供了 {len(mcp_tools)} 个工具:")

        # 把 MCP 工具格式 → OpenAI Function Calling 格式
        self.tools_for_openai = []
        for tool in mcp_tools:
            self.tools_for_openai.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
                },
            })
            print(f"  - {tool['name']}: {tool.get('description', '')[:60]}...")

    async def chat(self, user_message: str) -> str:
        """一轮对话：让 AI 决定是否调用 MCP 工具"""
        messages = [
            {"role": "system", "content": "你是有用的助手，用中文回答，简洁清晰。"},
            {"role": "user", "content": user_message},
        ]

        response = self.ds.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=self.tools_for_openai or None,
        )
        reply = response.choices[0].message

        if reply.tool_calls:
            for tc in reply.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                print(f"  [MCP 工具调用: {func_name}({func_args})]")

                tool_result = await self.mcp.call_tool(func_name, func_args)

                messages.append({
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [{
                        "id": tc.id, "type": "function",
                        "function": {"name": func_name, "arguments": tc.function.arguments},
                    }],
                })
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": tool_result,
                })

            response = self.ds.chat.completions.create(
                model="deepseek-chat", messages=messages,
            )
            reply = response.choices[0].message

        return reply.content


# ============================================================
# 架构对比
# ============================================================

def show_comparison():
    print("=" * 60)
    print("架构对比：TOOL_REGISTRY vs MCP")
    print("=" * 60)

    print("""
【旧模式：TOOL_REGISTRY（项目 03-07）】
  Agent 代码内部:
    TOOLS = {
      "get_time": {"func": get_time, "def": {"name": ..., "parameters": ...}},
      "run_code": {"func": run_code, "def": {"name": ..., "parameters": ...}},
      "read_file": {"func": read_file, "def": {"name": ..., "parameters": ...}},
    }
  新增工具 → 改 Agent 代码 → 重启 → 生效
  代码行数随工具数量线性增长

【新模式：MCP（本项目）】
  Agent 代码:
    mcp_tools = await mcp.list_tools()  # JSON-RPC: tools/list
    tools_for_openai = convert(mcp_tools)

  MCP Server（独立进程，FastMCP）:
    @mcp.tool()
    def get_current_time() -> str: ...
    @mcp.tool()
    def run_python_code(code: str) -> str: ...
    @mcp.tool()
    def read_file(filepath: str) -> str: ...

  新增工具 → 只在 Server 里加 @mcp.tool() → Agent 自动感知
  工具代码行数不随数量增长，Agent 代码完全不变！
""")


# ============================================================
# 主程序
# ============================================================

async def main():
    print("=" * 60)
    print("MCP Agent —— 通过 MCP 协议动态发现和调用工具")
    print("=" * 60)

    show_comparison()

    server_path = os.path.join(os.path.dirname(__file__), "18-mcp-server.py")
    mcp_client = MCPClient(server_path)
    agent = MCPAgent(mcp_client)

    print("连接 MCP Server...")
    await agent.connect()

    # 测试
    test_queries = [
        "现在几点了？",
        "帮我算一下 123 * 456 + 789",
        "读一下 18-mcp-server.py 这个文件，看看里面定义了哪些工具",
    ]

    for q in test_queries:
        print(f"\n用户: {q}")
        try:
            answer = await agent.chat(q)
            print(f"AI: {answer}")
        except Exception as e:
            print(f"出错: {e}")

    await mcp_client.close()

    print("\n" + "=" * 60)
    print("核心理解：")
    print('  MCP 协议的本质 = JSON-RPC 2.0 over stdio')
    print('')
    print('  三个核心操作：')
    print('    1. initialize   → 握手，交换能力')
    print('    2. tools/list   → 发现工具有哪些（替代 TOOL_REGISTRY）')
    print('    3. tools/call   → 调用工具（替代 tool_map[name](**args)）')
    print('')
    print('  MCP SDK 帮你封装了这些 JSON-RPC 通信，')
    print('  但底层就是 stdin/stdout 传 JSON，极其简单。')
    print("=" * 60)

    print("\n面试话术：")
    print('  "我实践过 MCP（Model Context Protocol），')
    print('   理解它的本质是 JSON-RPC 2.0 over stdio。')
    print('   用 FastMCP 搭建工具服务器，Agent 通过')
    print('   标准化协议动态发现和调用外部工具。')
    print('   新增工具零代码改动，实现了工具和Agent的解耦。"')


if __name__ == "__main__":
    asyncio.run(main())
