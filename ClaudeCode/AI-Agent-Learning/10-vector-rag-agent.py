"""
向量 RAG —— 用向量做语义搜索
================================
从关键词匹配升级到向量检索：
  1. 用 TF-IDF 把文字转成向量（每个词的重要性权重）
  2. 段落也都变成向量存起来
  3. 搜索 = 把问题向量和所有段落向量比余弦相似度

核心概念和神经网络嵌入一模一样，但没有网络依赖。
以后把 TfidfVectorizer 换成 SentenceTransformer 就是真正的语义搜索了。
"""
import sys
import os
import json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

DOCS_DIR = os.path.join(os.path.dirname(__file__), "document_library")
os.makedirs(DOCS_DIR, exist_ok=True)

# ============================================================
# 向量 RAG 引擎
# ============================================================

class VectorRAGEngine:
    """用 TF-IDF 向量做搜索的检索引擎"""

    def __init__(self):
        self.chunks = []          # 文本段落
        self.vectors = None       # 段落的向量矩阵 (N x 特征数)
        self.vectorizer = None    # TF-IDF 向量化器（记住训练时的词表）
        self.chunk_size = 200

    def load_file(self, filepath: str) -> str:
        """加载文件，切成段，转成向量"""
        if not os.path.exists(filepath):
            return f"文件不存在：{filepath}"

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            # 切分段落
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

            # 🔑 核心：用 TF-IDF 把所有段落转成向量
            # TF-IDF = 词频 × 逆文档频率（一个词越少见，权重越高）
            print(f"正在为 {len(self.chunks)} 个段落生成向量...")
            # analyzer='char_wb' + ngram: 用字符级 n-gram，中英文都适用
            # (1,2) 表示同时用单字和两字组合，覆盖"机器学习"→"机""器""学""习""机器""器学""学习"
            self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2))
            self.vectors = self.vectorizer.fit_transform(self.chunks)
            print(f"向量生成完毕！向量维度：{self.vectors.shape}")

            return f"已加载文件：{filepath}，共切成 {len(self.chunks)} 个段落，已生成向量"

        except Exception as e:
            return f"加载失败：{e}"

    def search(self, query: str, top_k: int = 3) -> str:
        """用余弦相似度找最相关的段落"""
        if not self.chunks:
            return "没有已加载的文档。请先用 load_document 加载一个文件。"

        # 🔑 1) 把问题转成向量（用同一个 vectorizer，保证词表一致）
        query_vec = self.vectorizer.transform([query])

        # 🔑 2) 计算问题和每个段落的余弦相似度
        scores = cosine_similarity(query_vec, self.vectors)[0]

        # 3) 排序取 top_k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = ranked[:top_k]

        result = []
        for idx, score in top:
            if score > 0.01:  # 相似度太低就跳过
                result.append(
                    f"[段落 {idx+1}，相似度: {score:.3f}]\n{self.chunks[idx]}"
                )

        if not result:
            return "没有找到相关内容。试试换个说法。"
        return "\n\n---\n\n".join(result)


# ============================================================
# 创建全局引擎
# ============================================================

rag = VectorRAGEngine()

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
    return rag.load_file(filepath)


def search_document(query: str) -> str:
    return rag.search(query)


# ============================================================
# 主程序
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
            "description": "用向量搜索在已加载的文档中查找相关内容。回答问题前必须先用此工具搜索。",
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
    print("向量 RAG 论文阅读助手（TF-IDF 向量搜索版）")
    print(f"示例文章在：{sample_path}")
    print("核心：文字 → 向量 → 余弦相似度 → 最相关段落")
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
