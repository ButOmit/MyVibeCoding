"""
混合检索 + 重排序 —— RAG 进阶
===================================
从单一搜索升级到组合拳：

  项目10: TF-IDF 关键词搜索        ← 字面匹配，"苹果手机"找不到"iPhone"
  项目14: Embedding 语义搜索        ← 语义理解，但"时间复杂度"可能返回"空间复杂度"
  本项目: BM25 + Embedding 混合检索  ← 互补优势 + RRF融合 + LLM重排序

三种技术的协同：
  BM25       → 精准匹配关键词（稀有词、专有名词、代码）
  Embedding  → 理解语义和同义词（跨语言、近义表达）
  RRF        → 融合两个排名（不依赖分数绝对值，对异构系统友好）
  LLM Rerank → 用 LLM 做最终把关（让AI自己判断哪个段落最相关）

真实世界的 RAG 系统都这么做：
  - Cohere Rerank / BGE Reranker → 专门的排序模型
  - WeChat/知乎的搜索 → BM25 + Embedding + 深度学习排序
  - LangChain 的 EnsembleRetriever → 多路召回 + RRF

安装依赖：
  pip install rank-bm25 sentence-transformers
"""
import sys
import os
import json
import requests
import numpy as np
import jieba

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

DS = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

OLLAMA_EMBED = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


# ============================================================
# 准备测试文档 —— 刻意设计成能体现不同搜索优缺点的内容
# ============================================================

DOCUMENTS = [
    "人工智能（Artificial Intelligence）是计算机科学的分支，旨在模拟人类智能。",
    "机器学习是人工智能的核心方法，让计算机从数据中学习规律。",
    "机器学习的三种主要范式：监督学习、无监督学习和强化学习。",
    "深度学习使用多层神经网络来提取数据的层次化特征表示。",
    "卷积神经网络（CNN）特别擅长处理图像数据，用卷积核扫描图片提取特征。",
    "循环神经网络（RNN）用于处理序列数据，比如时间序列、自然语言。",
    "Transformer架构通过自注意力机制（Self-Attention）并行处理整个序列，是当前大模型的基础。",
    "大语言模型（LLM）如DeepSeek、GPT、Claude等，基于Transformer架构，参数量达千亿级别。",
    "Python是很流行的编程语言，尤其在AI和数据分析领域广泛使用。",
    "强化学习中，智能体（Agent）通过与环境交互获得奖励信号来学习最优策略。",
    "GPU比CPU更适合矩阵运算，能大幅加速深度学习模型的训练过程。",
    "过拟合是机器学习中的常见问题，模型在训练集表现好但测试集表现差。",
]

print("=" * 60)
print("混合检索 + 重排序 —— RAG 进阶")
print(f"测试文档: {len(DOCUMENTS)} 篇，向量模型: {EMBED_MODEL}")
print("=" * 60)


# ============================================================
# 第一步: BM25 关键词检索
# ============================================================
# BM25 (Best Match 25) 是 TF-IDF 的进化版，工业界标配。
#
# 和 TF-IDF 的关键区别：
#   TF-IDF: score = TF * IDF（简单乘法，词频不受限制）
#   BM25:   score = IDF * (TF * (k1+1)) / (TF + k1*(1-b+b*len/avg_len))
#           - k1 控制词频饱和度（出现10次 ≠ 10倍重要性）
#           - b  控制文档长度惩罚（长文档不会天然占优）
#
# 为什么不用 BM25 替代 Embedding？
#   "苹果手机" 和 "iPhone" —— BM25 完全不匹配，Embedding 能对上。
#   所以两者互补，不是替代关系。

print("\n" + "=" * 60)
print("第一步: BM25 关键词检索")
print("=" * 60)

# 用 jieba 做中文分词（BM25 需要分词后的 token 列表）
tokenized_docs = [list(jieba.cut(doc)) for doc in DOCUMENTS]
bm25 = BM25Okapi(tokenized_docs)

print("\n用 jieba 分词预览:")
for i, tokens in enumerate(tokenized_docs[:3]):
    print(f"  [{i}] {' / '.join(tokens[:10])}...")


def bm25_search(query: str, top_k: int = 3) -> list[tuple[int, float, str]]:
    """BM25 关键词搜索，返回 [(doc_index, score, content), ...]"""
    tokens = list(jieba.cut(query))
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(idx, scores[idx], DOCUMENTS[idx]) for idx, _ in ranked[:top_k] if scores[idx] > 0]


# ============================================================
# 第二步: Embedding 语义检索（复用项目14的思路）
# ============================================================

print("\n" + "=" * 60)
print("第二步: Embedding 语义检索")
print("=" * 60)

# 预计算所有文档的 Embedding 向量（避免重复调用 API）
print(f"为 {len(DOCUMENTS)} 篇文档生成 Embedding 向量...")
doc_vectors = []
for i, doc in enumerate(DOCUMENTS):
    resp = requests.post(OLLAMA_EMBED, json={
        "model": EMBED_MODEL, "prompt": doc,
    }, timeout=60)
    vec = np.array(resp.json()["embedding"])
    doc_vectors.append(vec)
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(DOCUMENTS)} 完成...")
print(f"完成！每个向量 {len(doc_vectors[0])} 维")


def embed(text: str) -> np.ndarray:
    resp = requests.post(OLLAMA_EMBED, json={
        "model": EMBED_MODEL, "prompt": text,
    }, timeout=60)
    return np.array(resp.json()["embedding"])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def embedding_search(query: str, top_k: int = 3) -> list[tuple[int, float, str]]:
    """Embedding 语义搜索"""
    q_vec = embed(query)
    scores = [cosine_sim(q_vec, v) for v in doc_vectors]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(idx, scores[idx], DOCUMENTS[idx]) for idx, _ in ranked[:top_k] if scores[idx] > 0.2]


# ============================================================
# 第三步: RRF（Reciprocal Rank Fusion）混合检索
# ============================================================
# RRF 是融合多个排序列表的经典算法。
#
# 为什么用 RRF 而不是直接加权分数？
#   - BM25 的分数范围是 [0, ~50]，Embedding 是 [0, 1]，量纲完全不同
#   - 加权之前需要归一化，但归一化依赖于分数分布（最大/最小值）
#   - RRF 只关心排名，不关心绝对分数，天然规避了量纲问题
#
# RRF 公式: score(d) = Σ 1 / (k + rank_i(d))
#   k = 60（经典取值，让第一名贡献 ~1/61，第二名贡献 ~1/62）
#   rank_i(d) 是文档 d 在第 i 个排序列表中的排名（从1开始）

print("\n" + "=" * 60)
print("第三步: RRF 混合检索（BM25 + Embedding）")
print("=" * 60)


def rrf_fusion(
    bm25_results: list[tuple[int, float, str]],
    embed_results: list[tuple[int, float, str]],
    k: int = 60,
) -> list[tuple[int, float, str]]:
    """
    Reciprocal Rank Fusion —— 融合 BM25 和 Embedding 的排序结果。

    k=60 的含义：rank=1 的文档贡献 1/(60+1)=0.0164，
    这个值在工业界被广泛验证效果最好。
    """
    doc_scores = {}
    doc_contents = {}

    # BM25 排名贡献
    for rank, (idx, _, content) in enumerate(bm25_results):
        doc_scores[idx] = doc_scores.get(idx, 0) + 1.0 / (k + rank + 1)
        doc_contents[idx] = content

    # Embedding 排名贡献
    for rank, (idx, _, content) in enumerate(embed_results):
        doc_scores[idx] = doc_scores.get(idx, 0) + 1.0 / (k + rank + 1)
        doc_contents[idx] = content

    # 按融合分数排序
    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [(idx, score, doc_contents[idx]) for idx, score in ranked]


# ============================================================
# 第四步: LLM 重排序（Reranker）
# ============================================================
# 混合检索返回 top-N（比如10篇），但最终只喂给 LLM 2-3 篇。
# 用 LLM 自己再做一次筛选 —— 让它判断每个段落和问题的相关度。
#
# 两种主流重排序方案：
#   1. Cross-Encoder（如 BGE-Reranker）：专用小模型，快且便宜
#   2. LLM Reranker（本项目）：用 DeepSeek 打分，更准但更慢
#
# 本项目演示 LLM Reranker，因为：
#   1. 不需要额外下载模型（BGE-Reranker 要 1GB+）
#   2. 让你理解重排序的原理
#   3. 面试时能说 LLM-as-Reranker vs Cross-Encoder 的取舍

print("\n" + "=" * 60)
print("第四步: LLM 重排序（用 DeepSeek 做 Reranker）")
print("=" * 60)


def llm_rerank(query: str, candidates: list[tuple[int, float, str]], top_k: int = 3) -> list[str]:
    """
    让 LLM 判断每个候选段落和问题的相关度，重新排序。

    思路：给 LLM 列出候选段落，让它挑出最相关的 top_k 个。
    比单纯依赖向量距离更准，因为 LLM 真正"理解"了问题和段落。
    """
    if len(candidates) <= top_k:
        return [c[2] for c in candidates]

    # 构造候选列表
    items_text = ""
    for i, (_, _, content) in enumerate(candidates, 1):
        items_text += f"[{i}] {content}\n"

    prompt = f"""你是搜索质量评估专家。根据用户问题，从以下候选段落中选出最相关的{top_k}个。

不要凭关键词匹配，要判断段落内容是否真正能回答用户的问题。

用户问题：{query}

候选段落：
{items_text}
请选出最相关的{top_k}个段落的编号，按相关度从高到低排列。
只输出编号，用逗号分隔，如: 3,1,5"""

    response = DS.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是搜索质量评估专家。只输出编号，不要解释。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content.strip()
    # 解析编号
    try:
        indices = [int(x.strip()) - 1 for x in answer.replace("，", ",").split(",")]
        return [candidates[i][2] for i in indices if 0 <= i < len(candidates)]
    except (ValueError, IndexError):
        # LLM 输出格式不对时，回退到原始排序
        return [c[2] for c in candidates[:top_k]]


# ============================================================
# 完整对比 —— 四种搜索方案
# ============================================================

print("\n" + "=" * 60)
print("完整对比：四种搜索方案")
print("=" * 60)

test_queries = [
    # 场景A: 精确术语匹配 —— BM25 占优
    ("Transformer架构的自注意力机制", "精确术语匹配"),
    # 场景B: 语义同义表达 —— Embedding 占优
    ("计算机怎么从经验里学习", "语义表达（机器学习≈从经验里学习）"),
    # 场景C: 混合概念 —— 两者互补
    ("大型语言模型用了什么网络结构", "混合（大模型+网络结构）"),
    # 场景D: 同义词 —— Embedding 明显占优
    ("深度学习训练的时候用什么硬件加速", "同义表达（硬件加速=GPU/计算资源）"),
]

for query, scenario in test_queries:
    print(f"\n{'─' * 60}")
    print(f"查询: \"{query}\" ({scenario})")
    print(f"{'─' * 60}")

    # 1. BM25
    bm25_results = bm25_search(query)
    # 2. Embedding
    embed_results = embedding_search(query)
    # 3. RRF 混合
    hybrid_results = rrf_fusion(bm25_results, embed_results)
    # 4. LLM 重排序（在混合结果上再做重排）
    reranked = llm_rerank(query, hybrid_results, top_k=3)

    print(f"\n{'方案':<20} {'排名':<6} {'分数':<10} {'内容预览'}")
    print(f"{'─' * 20} {'─' * 6} {'─' * 10} {'─' * 45}")

    print(f"{'【BM25 关键词】':<20}")
    for rank, (idx, score, content) in enumerate(bm25_results, 1):
        print(f"  {rank:<18} #{idx:<5} {score:<10.4f} {content[:50]}...")

    print(f"{'【Embedding 语义】':<20}")
    for rank, (idx, score, content) in enumerate(embed_results, 1):
        print(f"  {rank:<18} #{idx:<5} {score:<10.4f} {content[:50]}...")

    print(f"{'【RRF 混合】':<20}")
    for rank, (idx, score, content) in enumerate(hybrid_results[:3], 1):
        print(f"  {rank:<18} #{idx:<5} {score:<10.4f} {content[:50]}...")

    print(f"{'【LLM 重排序】':<20}")
    for rank, content in enumerate(reranked, 1):
        print(f"  {rank:<18} {'(AI精选)':<5} {'':<10} {content[:50]}...")


# ============================================================
# 实战：用混合 RAG 回答问题
# ============================================================

print("\n" + "=" * 60)
print("实战：用混合检索 + LLM 重排序做问答")
print("=" * 60)


def advanced_rag_ask(question: str) -> str:
    """
    完整的进阶 RAG 流程：
    1. BM25 关键词召回（10篇）
    2. Embedding 语义召回（10篇）
    3. RRF 融合去重 → top-10
    4. LLM 重排序 → top-3
    5. 把 top-3 作为上下文喂给 LLM 生成答案
    """
    # Step 1-2: 双路召回
    bm25_results = bm25_search(question, top_k=10)
    embed_results = embedding_search(question, top_k=10)

    # Step 3: RRF 融合
    hybrid = rrf_fusion(bm25_results, embed_results)

    # Step 4: LLM 重排序
    best_chunks = llm_rerank(question, hybrid, top_k=3)

    # Step 5: 生成答案
    context = "\n\n".join(best_chunks)

    response = DS.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "你是智能助手。根据以下参考资料回答问题。如果资料中没有相关信息，诚实说不知道。",
            },
            {
                "role": "user",
                "content": f"参考资料：\n{context}\n\n问题：{question}\n\n请根据参考资料回答，控制在100字以内。",
            },
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


questions = [
    "什么是Transformer的自注意力机制？",
    "机器学习有哪些学习范式？",
    "大语言模型和Transformer有什么关系？",
    "训练深度学习模型用什么硬件比较好？",
]

for q in questions:
    print(f"\n问: {q}")
    answer = advanced_rag_ask(q)
    print(f"答: {answer}")


# ============================================================
# 总结对照表
# ============================================================

print("\n" + "=" * 60)
print("四种搜索方案对照")
print("=" * 60)

table = """
┌──────────────────┬────────────────────┬────────────────────┬────────────────┐
│ 方案              │ 强项                │ 弱项                │ 适用场景        │
├──────────────────┼────────────────────┼────────────────────┼────────────────┤
│ BM25 关键词       │ 专有名词/术语/代码  │ 同义词/跨语言       │ 精确查找        │
│ (项目10的朋友)     │ 不会漏掉生僻词      │ "苹果"找不到"iPhone" │ 法律/医疗文档   │
├──────────────────┼────────────────────┼────────────────────┼────────────────┤
│ Embedding 语义    │ 同义词/近义/跨语言  │ 可能把不同主题的    │ 模糊问题/闲聊   │
│ (项目14的方式)     │ "气候好"→"天气不错" │ 近义文档排在前面    │ 多语言场景      │
├──────────────────┼────────────────────┼────────────────────┼────────────────┤
│ RRF 混合          │ 互补不足            │ 两个都漏了就没办法  │ 通用RAG系统     │
│ (关键词+语义)      │ 不依赖绝对分数      │                    │ 企业搜索        │
├──────────────────┼────────────────────┼────────────────────┼────────────────┤
│ LLM 重排序        │ 真正的"理解"排序    │ 慢（多一次LLM调用） │ 高精度需求       │
│ (混合+AI把关)      │ 可以理解微妙的关联  │ 成本高              │ 最终结果把关    │
└──────────────────┴────────────────────┴────────────────────┴────────────────┘
"""
print(table)

print("面试话术：")
print('  "RAG 系统不能只依赖单一检索方式。')
print('   我实践过 BM25 关键词 + Embedding 语义的混合检索，')
print('   用 RRF（Reciprocal Rank Fusion）融合两路结果，')
print('   再用 LLM 做最终重排序提升精度。')
print('   这套组合拳能明显提升搜索命中率，尤其在中英文混合场景。\"')
