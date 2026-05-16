"""
联网搜索 Agent —— 让 AI 能查网页
===================================
基于框架，新增两个联网工具：
  1. web_search —— 搜索网页
  2. fetch_url —— 抓取网页内容
"""
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

from importlib.util import spec_from_file_location, module_from_spec

spec = spec_from_file_location("framework", os.path.join(os.path.dirname(__file__), "04-agent-framework.py"))
framework = module_from_spec(spec)
spec.loader.exec_module(framework)

Agent = framework.Agent
TOOL_REGISTRY = framework.TOOL_REGISTRY.copy()


# ============================================================
# 新工具1：网页搜索
# ============================================================

def web_search(query: str, max_results: int = 5) -> str:
    """用 DuckDuckGo 搜索网页，返回标题和链接"""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"- {r['title']}\n  {r['href']}\n  {r['body'][:100]}...")
        if not results:
            return "没有搜索到相关结果。"
        return "\n\n".join(results)
    except Exception as e:
        return f"搜索失败：{e}"


# ============================================================
# 新工具2：抓取网页
# ============================================================

def fetch_url(url: str) -> str:
    """抓取一个网页的内容（纯文本）"""
    try:
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AI-Agent/1.0)"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        # 简单提取纯文本（去掉 HTML 标签）
        import re
        text = re.sub(r"<[^>]+>", "", resp.text)
        text = re.sub(r"\s+", " ", text)
        if len(text) > 2000:
            text = text[:2000] + "...(内容已截断)"
        return text.strip()
    except Exception as e:
        return f"抓取失败：{e}"


# ============================================================
# 注册
# ============================================================

TOOL_REGISTRY.update({
    "web_search": {
        "func": web_search,
        "description": "搜索网页。当需要查找最新信息、新闻、或你不知道的内容时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "max_results": {"type": "integer", "description": "最多返回几条结果，默认5"},
            },
            "required": ["query"],
        },
    },
    "fetch_url": {
        "func": fetch_url,
        "description": "抓取一个网页的详细内容。搜索到感兴趣的链接后，用这个工具打开查看。",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要抓取的网页完整 URL"},
            },
            "required": ["url"],
        },
    },
})


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    agent = Agent(system_prompt="你是一个能上网搜索的 AI 助手。当用户问需要最新信息的问题时，用 web_search 搜索。找到感兴趣的结果后，用 fetch_url 打开看详情。用中文回答。")

    for name, info in TOOL_REGISTRY.items():
        agent.add_tool(name, info["func"], info["description"], info["parameters"])

    print("=" * 50)
    print("联网 Agent —— 我能搜索网页了！")
    print("试试：")
    print("  '最近有什么科技新闻？'")
    print("  '帮我查一下今天的天气'")
    print("  'Python 3.13 有什么新特性'")
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
