"""
RAG 论文阅读助手 —— 让 AI 读懂你的文档
===========================================
核心概念 RAG（检索增强生成）：
  1. 加载文档，切成小段（Chunking）
  2. 用户提问时，找到最相关的小段（Retrieval）
  3. 把相关段落喂给 AI，让它基于原文回答（Augmented Generation）

不需要任何外部 API，纯本地实现。
"""
import sys
import os
import re
import json

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

DOCS_DIR = os.path.join(os.path.dirname(__file__), "document_library")
os.makedirs(DOCS_DIR, exist_ok=True)

# ============================================================
# RAG 引擎：切分 + 检索
# ============================================================

class RAGEngine:
    """文档检索引擎"""

    def __init__(self):
        self.chunks = []      # 切好的文本段
        self.chunk_size = 200  # 每段 200 字左右

    def load_file(self, filepath: str) -> str:
        """加载文件，切成小段，存入 chunks"""
        if not os.path.exists(filepath):
            return f"文件不存在：{filepath}"

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            # 按段落切分，太短的合并，太长的再切
            paragraphs = text.split("\n")
            self.chunks = []
            current = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                current += para + "\n"
                if len(current) >= self.chunk_size:
                    self.chunks.append(current.strip())
                    current = ""
            if current.strip():
                self.chunks.append(current.strip())

            return f"已加载文件：{filepath}，共切成 {len(self.chunks)} 个段落"

        except Exception as e:
            return f"加载失败：{e}"

    def search(self, query: str, top_k: int = 3) -> str:
        """根据问题检索最相关的段落（支持中英文）"""
        if not self.chunks:
            return "没有已加载的文档。请先用 load_file 加载一个文件。"

        q = query.lower()
        scored = []
        for i, chunk in enumerate(self.chunks):
            chunk_lower = chunk.lower()
            score = 0

            # 1) 完整问题匹配（权重最高）
            score += chunk_lower.count(q) * 5

            # 2) 中文：字符 bigram 匹配（2字滑动窗口）
            for j in range(len(q) - 1):
                bigram = q[j:j+2]
                if bigram in chunk_lower:
                    score += 1

            # 3) 英文：空格分词匹配
            for word in q.split():
                if word in chunk_lower:
                    score += 1

            scored.append((score, i, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = scored[:top_k]

        result = []
        for score, idx, chunk in top_chunks:
            if score > 0:
                result.append(f"[段落 {idx+1}，相关度: {score}]\n{chunk}")

        if not result:
            return "没有找到相关内容。试试换个关键词。"
        return "\n\n---\n\n".join(result)


# ============================================================
# 创建全局 RAG 引擎
# ============================================================

rag = RAGEngine()

# 放一篇示例文章到 document_library，方便测试
sample_text = """# Python 与人工智能导论

## 第一章：什么是人工智能

人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支，旨在创建能够模拟人类智能的系统。AI 系统可以执行学习、推理、感知、语言理解等任务。

人工智能主要分为三类：狭义人工智能（ANI）、通用人工智能（AGI）和超级人工智能（ASI）。目前我们使用的所有 AI 都属于狭义人工智能，它们只能在特定领域内执行任务。

## 第二章：机器学习基础

机器学习（Machine Learning）是人工智能的核心方法之一。它让计算机通过数据来学习，而不是通过显式编程。机器学习的三种主要范式是：监督学习、无监督学习和强化学习。

监督学习使用带标签的数据来训练模型。例如，给模型看很多猫和狗的图片，并告诉它每张图是什么，模型就能学会区分猫和狗。

无监督学习则使用没有标签的数据。它试图在数据中发现隐藏的模式或结构，比如将相似的客户分成不同群体。

强化学习让智能体在环境中通过试错来学习。智能体执行动作，环境给予奖励或惩罚，智能体逐渐学会最优策略。

## 第三章：深度学习

深度学习（Deep Learning）是机器学习的一个子领域，使用多层神经网络来学习数据的表示。深度学习在图像识别、自然语言处理、语音识别等领域取得了突破性进展。

著名的深度学习架构包括：卷积神经网络（CNN）用于图像处理，循环神经网络（RNN）用于序列数据，Transformer 架构用于自然语言处理。

## 第四章：Python 在 AI 中的应用

Python 是目前 AI 开发中最流行的编程语言。它的语法简洁，拥有丰富的科学计算和机器学习库，如 NumPy、Pandas、Scikit-learn、PyTorch 和 TensorFlow。

Python 的优势在于其庞大的社区和生态系统。无论你想做什么 AI 任务，几乎都能找到对应的 Python 库来帮助你。

## 第五章：未来展望

AI 技术正在快速发展。大语言模型（LLM）如 GPT、Claude 和 DeepSeek 正在改变人类与计算机交互的方式。未来的 AI 系统将更加智能、更加可靠，但也带来了伦理和安全性方面的挑战。
"""

sample_path = os.path.join(DOCS_DIR, "AI入门简介.txt")
if not os.path.exists(sample_path):
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(sample_text)


# ============================================================
# 工具函数
# ============================================================

def load_document(filepath: str) -> str:
    """加载一个文档到 RAG 引擎，准备被提问"""
    return rag.load_file(filepath)


def search_document(query: str) -> str:
    """在已加载的文档中搜索与问题相关的内容"""
    return rag.search(query)


# ============================================================
# 主程序（终端交互版）
# ============================================================

SYSTEM_PROMPT = f"""你是一个论文阅读助手。用户会加载一篇文档然后提问。
你需要：
1. 用户提问时，先用 search_document 搜索文档里的相关内容
2. 根据搜索结果回答用户问题
3. 如果文档中没有相关信息，诚实告诉用户
4. 回答时引用是第几个段落

文档库目录：{DOCS_DIR}
里面有一篇示例文章 {sample_path}，用户可以加载它来测试。"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "load_document",
            "description": "加载一个 txt 文档。用户想阅读某篇文章时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "文档的完整路径"},
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": "在已加载的文档中搜索与问题相关的内容。回答问题前必须先用此工具搜索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词或问题"},
                },
                "required": ["query"],
            },
        },
    },
]

tool_map = {
    "load_document": load_document,
    "search_document": search_document,
}


if __name__ == "__main__":
    print("=" * 50)
    print("RAG 论文阅读助手")
    print(f"示例文章在：{sample_path}")
    print("试试：")
    print("  1. 先加载文档")
    print(f"  2. 问'机器学习有哪些类型'")
    print("输入 'quit' 退出")
    print("=" * 50)

    while True:
        user_input = input("\n你: ")
        if user_input.lower() == "quit":
            print("再见！")
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
        )
        reply = response.choices[0].message

        if reply.tool_calls:
            for tool_call in reply.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                tool_result = tool_map[func_name](**func_args)

                print(f"  [RAG 工具: {func_name}({func_args})]")

                messages.append({
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [{
                        "id": tool_call.id, "type": "function",
                        "function": {"name": func_name, "arguments": tool_call.function.arguments},
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result),
                })

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
            )
            reply = response.choices[0].message

        print(f"AI: {reply.content}")
        messages.append({"role": "assistant", "content": reply.content})
