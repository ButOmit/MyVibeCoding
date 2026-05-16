"""
文件工具 Agent —— 让 AI 能读写你的文件
========================================
基于 04-agent-framework.py 的框架，新增两个文件工具。
展示框架的扩展能力：想加工具？写函数 → add_tool → 搞定。
"""
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

# 直接复用框架！
from importlib.util import spec_from_file_location, module_from_spec

spec = spec_from_file_location("framework", os.path.join(os.path.dirname(__file__), "04-agent-framework.py"))
framework = module_from_spec(spec)
spec.loader.exec_module(framework)

Agent = framework.Agent
TOOL_REGISTRY = framework.TOOL_REGISTRY.copy()  # 继承基础工具


# ============================================================
# 新增工具1：读文件
# ============================================================

def read_file(path: str) -> str:
    """读取指定路径的文件内容"""
    if not os.path.exists(path):
        return f"文件不存在：{path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # 限制返回长度，避免 token 爆炸
        if len(content) > 3000:
            content = content[:3000] + "\n...(内容已截断)"
        return content
    except Exception as e:
        return f"读取失败：{e}"


# ============================================================
# 新增工具2：写文件
# ============================================================

def write_file(path: str, content: str) -> str:
    """写入内容到文件。会自动创建不存在的目录。"""
    try:
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"文件已写入：{path}（{len(content)} 个字符）"
    except Exception as e:
        return f"写入失败：{e}"


# ============================================================
# 注册新工具
# ============================================================

TOOL_REGISTRY.update({
    "read_file": {
        "func": read_file,
        "description": "读取电脑上的文件内容。当需要查看某个文件时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件的完整路径"},
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "func": write_file,
        "description": "将内容写入文件。用户要求保存、记录、导出时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要写入的文件路径"},
                "content": {"type": "string", "description": "要写入的内容"},
            },
            "required": ["path", "content"],
        },
    },
})


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    # 指定一个工作目录给 Agent，避免它乱翻
    work_dir = os.path.join(os.path.dirname(__file__), "agent_workspace")
    os.makedirs(work_dir, exist_ok=True)

    agent = Agent(system_prompt=f"""你是一个能用工具的 AI 助手，可以用中文回答问题。
你可以：
- 运行 Python 代码做计算
- 查看当前时间
- 读写电脑上的文件

文件操作的默认目录是：{work_dir}
用户说"记下来"时，你就用 write_file 保存到那个目录。
用户说"看一下"时，你就用 read_file 读取。""")

    for name, info in TOOL_REGISTRY.items():
        agent.add_tool(name, info["func"], info["description"], info["parameters"])

    print("=" * 50)
    print("文件 Agent —— 我能帮你读写文件了！")
    print(f"工作目录：{work_dir}")
    print("试试：")
    print("  '帮我写一首诗保存到 poem.txt'")
    print("  '帮我记一下：明天下午3点开会'")
    print("  '看看 poem.txt 里写了什么'")
    print("输入 'quit' 退出，'clear' 清空记忆")
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
