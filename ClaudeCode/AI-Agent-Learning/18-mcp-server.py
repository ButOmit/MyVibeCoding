"""
MCP 工具服务器 (FastMCP 版) —— 把工具"外包"给独立服务
========================================================
MCP（Model Context Protocol）的核心思想：
  之前：工具函数 + JSON Schema 都写在 Agent 代码里（TOOL_REGISTRY）
  MCP：  工具跑在独立的服务器进程里，Agent 通过协议动态发现和调用

打个比方：
  之前 = 餐厅后厨和菜单都在同一家店（耦合）
  MCP  = 外卖平台 —— 餐厅只负责做菜（Server），平台只负责点单（Agent）
        餐厅换菜单不影响平台，平台换 UI 不影响餐厅（解耦）

FastMCP = MCP SDK 1.x 的高级 API，用 @mcp.tool() 装饰器自动注册工具，
类型注解自动生成 JSON Schema，比低层 Server API 简洁很多。

真实场景：
  - 公司有 100 个微服务 → 每个服务提供 FastMCP Server
  - Agent 通过 MCP 动态发现所有服务 → 不用手动写 TOOL_REGISTRY
  - 新增服务 → Agent 自动感知，零代码改动
"""

import sys
import io
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from mcp.server.fastmcp import FastMCP

# ---- 创建 FastMCP 服务器 ----
# 一个名字搞定，不需要手写 list_tools/call_tool
mcp = FastMCP("my-toolbox")


# ============================================================
# 工具 1: 获取当前时间
# ============================================================
# @mcp.tool() 自动：
#   1. 从函数名生成工具名
#   2. 从 docstring 提取 description
#   3. 从类型注解生成 inputSchema
# 这就是之前 TOOL_REGISTRY 里的全部信息，现在一个装饰器搞定！

@mcp.tool()
def get_current_time() -> str:
    """获取当前的日期和时间。当用户问'现在几点'、'今天几号'时使用。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# 工具 2: 执行 Python 代码
# ============================================================

@mcp.tool()
def run_python_code(code: str) -> str:
    """执行 Python 代码并返回结果。需要数学计算、数据处理时使用。

    Args:
        code: 要执行的 Python 代码字符串
    """
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
        return "执行完成"
    except Exception as e:
        sys.stdout = old_stdout
        return f"出错：{e}"


# ============================================================
# 工具 3: 读取文件
# ============================================================

@mcp.tool()
def read_file(filepath: str) -> str:
    """读取文件内容。用户要求查看或读取文件时使用。

    Args:
        filepath: 文件的完整路径
    """
    try:
        content = Path(filepath).read_text(encoding="utf-8")
        return content[:2000]
    except FileNotFoundError:
        return f"文件不存在：{filepath}"
    except Exception as e:
        return f"读取失败：{e}"


# ============================================================
# 对比：FastMCP vs 低层 Server API
# ============================================================
# 低层 API（之前的写法）:
#   server = Server("my-toolbox")
#   @server.list_tools()
#   async def list_tools():
#       return [Tool(name="get_current_time", description="...", inputSchema={...})]
#   @server.call_tool()
#   async def call_tool(name, arguments):
#       if name == "get_current_time": ...
#
# FastMCP（本文件）:
#   mcp = FastMCP("my-toolbox")
#   @mcp.tool()
#   def get_current_time() -> str:
#       """获取当前的日期和时间。"""
#       return datetime.now().strftime(...)
#
# 代码量减少 60%+，类型注解自动生成 JSON Schema！


# ---- 启动服务器 ----
if __name__ == "__main__":
    mcp.run(transport="stdio")
