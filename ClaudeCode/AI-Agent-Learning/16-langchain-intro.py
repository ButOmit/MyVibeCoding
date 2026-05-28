"""
LangChain 入门 —— 用框架做你之前手写的事
=============================================
LangChain 是目前最主流的 AI 开发框架。学了它，面试时能说
"我既理解底层原理（手写过Agent），也会用工业界框架（LangChain）"。

核心思想：把你手写的代码封装成标准化的"链"和"组件"。

本文件对照你之前的项目，展示 LangChain 的三大核心模块：

  模块1: Chat Model    ← 替代你的 client.chat.completions.create()
  模块2: Tools + Agent  ← 替代你的 Agent 类 + TOOL_REGISTRY
  模块3: RAG           ← 替代你的 EmbeddingRAGEngine + Chroma 向量库

安装依赖：
  pip install langchain langchain-openai langchain-community chromadb
"""
import sys
import os
import io
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# 配置：告诉 LangChain 用哪个模型
# ============================================================

# LangChain 统一了不同 AI 的接口。不管用 DeepSeek/OpenAI/本地模型，
# 都用同一套代码，只改配置就行。
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.7,
)

print("=" * 60)
print("LangChain 三大核心模块对比学习")
print(f"模型: deepseek-chat (通过 {type(llm).__name__})")
print("=" * 60)


# ============================================================
# 模块1: Chat Model —— 替代你的 client.chat.completions.create()
# ============================================================
# 你之前做的:
#   response = client.chat.completions.create(
#       model="deepseek-chat", messages=[{"role": "user", "content": "你好"}]
#   )
#   print(response.choices[0].message.content)
#
# LangChain 的做法:
#   msg = HumanMessage(content="你好")
#   response = llm.invoke([msg])
#   print(response.content)
# ============================================================

print("\n" + "=" * 60)
print("模块1: Chat Model —— 对话模型")
print("=" * 60)

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# LangChain 的 messages 是强类型的，不是裸字典
messages = [
    SystemMessage(content="你是有用的助手，用中文回答问题，20字以内。"),
    HumanMessage(content="什么是 LangChain？"),
]

# 方法1: invoke() —— 非流式，和你之前的用法一样
response = llm.invoke(messages)
print(f"\n[invoke] {response.content}")

# 方法2: stream() —— 流式，对应你的 stream=True
print("\n[stream 流式输出] ", end="", flush=True)
for chunk in llm.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)
print()

# 对比总结
print("\n你手写的:  client.chat.completions.create() → response.choices[0].message.content")
print("LangChain:  llm.invoke(messages) → response.content")


# ============================================================
# 模块2: Tools + Agent  —— 替代你的 Agent 类 + TOOL_REGISTRY
# ============================================================
# 你之前做的:
#   TOOL_REGISTRY = {"get_weather": {"func": ..., "description": ..., "parameters": ...}}
#   class Agent: ... 手动管理消息、检测 tool_calls、执行工具
#
# LangChain 的做法:
#   @tool 装饰器 → 自动生成工具的 JSON Schema
#   create_openai_tools_agent() → 自动构建 Agent
#   AgentExecutor → 自动管理消息、检测 tool_calls、执行工具、循环
# ============================================================

print("\n" + "=" * 60)
print("模块2: Tools + Agent —— 工具调用")
print("=" * 60)

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

# ---- 步骤1: 用 @tool 装饰器定义工具 ----
# 你之前要写 func + description + parameters JSON Schema，
# LangChain 只需要一个装饰器 + 类型注解 + docstring，自动生成一切

@tool
def get_current_time() -> str:
    """获取当前的日期和时间。用户问时间时使用。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def run_python_code(code: str) -> str:
    """执行 Python 代码。需要计算、处理数据时使用。

    Args:
        code: 要执行的 Python 代码
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


tools = [get_current_time, run_python_code]

# ---- 步骤2: 构建 LangGraph Agent ----
# LangGraph 是 LangChain 的升级版 Agent 框架，用"图"来表示 Agent 的工作流
# 节点 = Agent 思考 / 工具执行
# 边 = 根据是否有 tool_calls 走不同路径

# 把工具绑定到 LLM 上
llm_with_tools = llm.bind_tools(tools)


def should_continue(state: MessagesState):
    """判断下一步：有工具调用就走 tools 节点，否则结束"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    """Agent 思考节点：LLM 决定是回答还是调用工具"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# 构建图
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

# 编译图（带记忆）
agent = workflow.compile(checkpointer=MemorySaver())

# ---- 步骤3: 测试 ----
config = {"configurable": {"thread_id": "test-1"}}
queries = [
    "现在几点了？",
    "帮我算一下 2 的 10 次方等于多少",
]

for q in queries:
    print(f"\n用户: {q}")
    result = agent.invoke(
        {"messages": [SystemMessage(content="你是友好的助手，用中文回答。"), HumanMessage(content=q)]},
        config=config,
    )
    final = result["messages"][-1]
    print(f"AI: {final.content}")

# 对比总结
print("\n你手写的:  Agent 类 → TOOL_REGISTRY → 手动 while 循环 → 手工拼接 messages")
print("LangGraph:  @tool 装饰器 → llm.bind_tools() → 图结构自动化")


# ============================================================
# 模块3: RAG —— 替代你的 EmbeddingRAGEngine
# ============================================================
# 你之前做的:
#   1. 手动读文件 → split("\n") → 合并到 200 字 → chunks
#   2. 每个 chunk 调用 Ollama embed API → 手动存列表
#   3. 搜索时 embed query → 手动算余弦相似度 → 排序
#
# LangChain 的做法:
#   TextLoader → RecursiveCharacterTextSplitter → Chroma 向量库
#   搜索: vector_store.similarity_search(query)
# ============================================================

print("\n" + "=" * 60)
print("模块3: RAG —— 文档检索")
print("=" * 60)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# 用你文档库里的示例文章
DOCS_DIR = os.path.join(os.path.dirname(__file__), "document_library")
sample_path = os.path.join(DOCS_DIR, "AI入门简介.txt")

# 确保示例文档存在
sample_text = """# Python 与人工智能导论

## 第一章：什么是人工智能

人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支。
人工智能主要分为三类：狭义人工智能（ANI）、通用人工智能（AGI）和超级人工智能（ASI）。

## 第二章：机器学习基础

机器学习（Machine Learning）是人工智能的核心方法之一。
机器学习的三种主要范式是：监督学习、无监督学习和强化学习。

## 第三章：深度学习

深度学习（Deep Learning）是机器学习的一个子领域，使用多层神经网络来学习数据的表示。
著名的深度学习架构包括：卷积神经网络（CNN）、循环神经网络（RNN）和 Transformer 架构。

## 第四章：Python 在 AI 中的应用

Python 是目前 AI 开发中最流行的编程语言。
"""
os.makedirs(DOCS_DIR, exist_ok=True)
with open(sample_path, "w", encoding="utf-8") as f:
    f.write(sample_text)

# ---- 步骤1: 加载文档 ----
# 你之前: open(file).read() → split("\n") → 手动合并
# LangChain: TextLoader → RecursiveCharacterTextSplitter（智能切分）
print(f"\n加载文档: {sample_path}")
loader = TextLoader(sample_path, encoding="utf-8")
documents = loader.load()
print(f"  加载了 {len(documents)} 篇文档")

# 智能切分：按段落+句子边界切，尽量保持语义完整
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # 每个 chunk 最多 200 字
    chunk_overlap=50,    # 重叠 50 字（避免关键信息被切断）
    separators=["\n\n", "\n", "。", ".", " "],
)
chunks = text_splitter.split_documents(documents)
print(f"  切成了 {len(chunks)} 个段落（带 overlap）")

# ---- 步骤2: 生成向量并存入数据库 ----
# 你之前: ollama embed API → 手动存 np.array
# LangChain: OllamaEmbeddings → Chroma 向量数据库（自动持久化）
#
# 和你项目14一样用 nomic-embed-text，但 LangChain 封装成一行代码！
print(f"\n生成 Embedding 向量并存入 Chroma...")
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)

# Chroma 是一个轻量级向量数据库，自动做索引、相似度搜索、持久化
vector_store = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=os.path.join(os.path.dirname(__file__), "chroma_db"),
)
print(f"  已存入 {vector_store._collection.count()} 条向量")

# ---- 步骤3: 搜索 ----
# 你之前: embed(query) → 手动余弦相似度 → sorted → top_k
# LangChain: vector_store.similarity_search(query, k=3)
print("\n" + "-" * 40)
queries = [
    "机器学习有哪些学习范式",
    "深度学习用什么架构",
    "Python为什么在AI中流行",
]
for q in queries:
    print(f"\n搜索: '{q}'")
    results = vector_store.similarity_search(q, k=2)
    for i, doc in enumerate(results, 1):
        preview = doc.page_content.replace("\n", " ")[:80]
        print(f"  [{i}] {preview}...")

# ---- 步骤4: RAG 问答 ----
# 你之前: Agent 先 search_document → 把结果喂给 LLM
# LangChain: retriever + prompt 模板 → 链式处理（LCEL）
print("\n" + "-" * 40)
print("RAG 问答:")

# LangChain 的 LCEL (LangChain Expression Language)
# 用 | 管道符把各个步骤串起来，像 Linux 管道一样
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是文档助手。根据以下文档内容回答用户问题。如果文档中没有相关信息，诚实说不知道。\n\n文档内容：\n{context}"),
    ("human", "{question}"),
])

# retriever 负责检索，prompt 负责格式化，llm 负责生成
chain = (
    {"context": vector_store.as_retriever(search_kwargs={"k": 3}), "question": lambda x: x}
    | prompt
    | llm
)

question = "机器学习的三种范式是什么？简单回答"
answer = chain.invoke(question)
print(f"  问题: {question}")
print(f"  回答: {answer.content}")

# 对比总结
print("\n你手写的:  read()→split→embed API→cosine→sorted→LLM")
print("LangChain:  TextLoader→Splitter→Chroma→RetrievalQA 链")


# ============================================================
# 总结对照表
# ============================================================

print("\n" + "=" * 60)
print("对照总结：你手写的代码 vs LangChain")
print("=" * 60)

table = """
┌─────────────────────────┬──────────────────────────────┬────────────────────────────────┐
│ 功能                     │ 你手写的（15个项目）          │ LangChain/Graph               │
├─────────────────────────┼──────────────────────────────┼────────────────────────────────┤
│ 对话模型                 │ client.chat.completions      │ ChatOpenAI(model=...).invoke() │
│ 流式输出                 │ stream=True + for chunk      │ llm.stream(messages)           │
│ 工具定义                 │ TOOL_REGISTRY 字典+JSONSchema│ @tool 装饰器自动生成           │
│ Agent 工具循环           │ Agent 类 + 手动 while 循环   │ LangGraph Agent + 自动图       │
│ 文档加载                 │ open().read() + split("\\n") │ TextLoader + Splitter          │
│ 文本切分                 │ 手动合并段落 (200字)         │ RecursiveCharacterTextSplitter│
│ 向量生成                 │ Ollama embed API             │ OllamaEmbeddings (封装一行)    │
│ 向量存储+搜索             │ 手动 np.array + cosine       │ Chroma 向量数据库              │
│ RAG 问答                 │ 手动 search→LLM              │ RetrievalQA 链                 │
└─────────────────────────┴──────────────────────────────┴────────────────────────────────┘
"""
print(table)

print("面试话术：")
print('  "我既手写过 Agent 和 RAG（理解底层原理），')
print('   也熟练使用 LangChain/LangGraph 框架（工程化开发）。')
print('   比如 Embedding 语义搜索，我能说清楚余弦相似度的数学原理，')
print('   也能用 Chroma + RetrievalQA 链快速搭建。\"')
