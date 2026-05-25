"""
Embedding RAG —— 真正的语义搜索
===================================
从 TF-IDF 升级到神经网络 Embedding：

  TF-IDF (项目10):  统计词频 → "苹果手机" 和 "iPhone" 完全不沾边
  Embedding (本项目): 神经网络 → "苹果手机" 和 "iPhone" 向量很近！

原理：
  1. Embedding 模型是专门训练来"理解语义"的神经网络
  2. 它把任意文字映射成一个固定长度的向量（比如 768 个浮点数）
  3. 语义相近的文字 → 向量方向相近 → 余弦相似度高
  4. 这和 ChatGPT 理解文字的原理一样，只不过它只输出向量，不输出文字

流程和 TF-IDF 完全一样，唯一区别：
  之前: TfidfVectorizer → 统计词频向量
  现在: Ollama Embedding API → 神经网络语义向量
"""
import sys
import os
import json
import requests
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# DeepSeek 对话客户端 + Ollama Embedding API
# ============================================================

ds_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

OLLAMA_EMBED = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

DOCS_DIR = os.path.join(os.path.dirname(__file__), "document_library")
os.makedirs(DOCS_DIR, exist_ok=True)


# ============================================================
# 向量工具函数
# ============================================================

def embed(text: str) -> np.ndarray:
    """
    把一段文字变成向量。

    调用 Ollama 的 Embedding API → 返回 768 个浮点数组成的数组。
    这就是神经网络对这段文字的"理解"。
    """
    resp = requests.post(OLLAMA_EMBED, json={
        "model": EMBED_MODEL,
        "prompt": text,
    }, timeout=60)
    vec = resp.json()["embedding"]
    return np.array(vec)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度 —— 衡量两个向量"方向有多接近"。

    范围 [-1, 1]，越大越相似。
    数学上就是两个向量夹角的余弦值。
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


# ============================================================
# Embedding RAG 引擎
# ============================================================

class EmbeddingRAGEngine:
    """
    用神经网络 Embedding 做语义搜索。

    和项目10的 VectorRAGEngine 接口一样，但内部完全不同：
      - 10: TfidfVectorizer（统计模型）
      - 14: Embedding 模型（神经网络模型）
    """

    def __init__(self):
        self.chunks = []
        self.vectors = []       # 每个段落的 Embedding 向量
        self.chunk_size = 200

    def load_file(self, filepath: str) -> str:
        filepath = filepath.replace("\\", "/")
        if not os.path.exists(filepath):
            return f"文件不存在：{filepath}"

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            # 切分段落（和之前一样）
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

            # 每个段落都调用 Embedding API 生成向量
            print(f"正在为 {len(self.chunks)} 个段落生成 Embedding 向量...")
            self.vectors = []
            for i, chunk in enumerate(self.chunks):
                vec = embed(chunk)
                self.vectors.append(vec)
                if (i + 1) % 5 == 0:
                    print(f"  {i+1}/{len(self.chunks)} 完成...")

            print(f"Embedding 完毕！共 {len(self.vectors)} 个向量，每个 {len(self.vectors[0])} 维")
            return f"已加载文件：{filepath}，共 {len(self.chunks)} 个段落，已生成语义向量"

        except Exception as e:
            return f"加载失败：{e}"

    def search(self, query: str, top_k: int = 3) -> str:
        """用语义搜索找最相关的段落"""
        if not self.chunks:
            return "没有已加载的文档。"

        # 把问题也转成 Embedding 向量
        query_vec = embed(query)

        # 计算和每个段落的余弦相似度
        scores = [cosine_similarity(query_vec, v) for v in self.vectors]

        # 排序取 top_k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = ranked[:top_k]

        result = []
        for idx, score in top:
            if score > 0.2:  # Embedding 的相似度比 TF-IDF 更有意义
                result.append(
                    f"[段落 {idx+1}，语义相似度: {score:.3f}]\n{self.chunks[idx]}"
                )

        if not result:
            return "没有找到相关内容。"
        return "\n\n---\n\n".join(result)


# ============================================================
# TF-IDF vs Embedding 对比演示
# ============================================================

def compare_search():
    """
    用同一个查询，对比 TF-IDF 和 Embedding 的搜索结果。
    Embedding 能理解语义，TF-IDF 只能匹配字面。
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as tfidf_cosine

    # 模拟文档：包含同义词和近义表达
    docs = [
        "今天天气真好，阳光明媚。",
        "苹果公司发布了新款智能手机。",
        "iPhone 16 的销量创下新纪录。",
        "我喜欢吃苹果这种水果。",
        "机器学习是人工智能的重要分支。",
        "深度学习使用多层神经网络来学习数据表示。",
        "今天下雨了，出门记得带伞。",
    ]

    # === TF-IDF 搜索 ===
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2))
    tfidf_vectors = tfidf.fit_transform(docs)
    queries_tfidf = {
        "今天气候不错": "同义表达（天气≈气候，不错≈好）",
        "Apple手机": "跨语言同义（Apple=苹果，手机=智能手机）",
        "计算机怎么学习": "语义相近（计算机学习≈机器学习）",
    }

    print("=" * 60)
    print("TF-IDF vs Embedding 语义搜索对比")
    print("=" * 60)

    for query, description in queries_tfidf.items():
        print(f"\n{'─' * 60}")
        print(f"查询: \"{query}\" ← {description}")
        print(f"{'─' * 60}")

        # TF-IDF
        q_tfidf = tfidf.transform([query])
        tfidf_scores = tfidf_cosine(q_tfidf, tfidf_vectors)[0]
        tfidf_ranked = sorted(enumerate(tfidf_scores), key=lambda x: x[1], reverse=True)

        # Embedding
        q_embed = embed(query)
        embed_scores = [cosine_similarity(q_embed, embed(d)) for d in docs]
        embed_ranked = sorted(enumerate(embed_scores), key=lambda x: x[1], reverse=True)

        print()
        print(f"{'排名':<5} {'TF-IDF (字面匹配)':<45} {'Embedding (语义理解)':<45}")
        print(f"{'─' * 5} {'─' * 45} {'─' * 45}")
        for rank in range(3):
            tidx, tscore = tfidf_ranked[rank]
            eidx, escore = embed_ranked[rank]
            ttext = docs[tidx][:40]
            etext = docs[eidx][:40]
            print(f"{rank+1:<5} [{tscore:.2f}] {ttext:<40} | [{escore:.2f}] {etext:<40}")


# ============================================================
# 创建全局引擎
# ============================================================

rag = EmbeddingRAGEngine()

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


if __name__ == "__main__":
    print("=" * 60)
    print("Embedding RAG 论文阅读助手（语义搜索版）")
    print(f"向量模型: {EMBED_MODEL} (神经网络)")
    print(f"对比 TF-IDF: 字面匹配 → 语义理解")
    print("=" * 60)

    # 先跑对比演示
    print("\n先看一个 TF-IDF vs Embedding 的对比：\n")
    try:
        compare_search()
    except Exception as e:
        print(f"(对比演示需要 nomic-embed-text 模型已安装: {e})")

    print(f"\n{'=' * 60}")
    print("开始文档问答")
    print(f"示例文章: {sample_path}")
    print("输入 'quit' 退出")
    print(f"{'=' * 60}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "load_document",
                "description": "加载一个 txt 文档到 RAG 引擎。",
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
                "description": "用语义搜索在已加载的文档中查找相关内容。回答问题前必须先搜索。",
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

    while True:
        user_input = input("\n你: ")
        if user_input.lower() == "quit":
            print("再见！")
            break

        messages.append({"role": "user", "content": user_input})

        response = ds_client.chat.completions.create(
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

            response = ds_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
            )
            reply = response.choices[0].message

        print(f"AI: {reply.content}")
        messages.append({"role": "assistant", "content": reply.content})
