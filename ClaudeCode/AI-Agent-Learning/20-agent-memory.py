"""
Agent 记忆系统 —— 从"金鱼记忆"到"长期陪伴"
===============================================
之前你的 Agent（项目 2-4）:
  messages = [{"role": "system", ...}, {"role": "user", ...}, ...]
  问题：对话越长 → tokens 越多 → 超出模型上限 → 必须截断 → 丢失早期信息

工业级 Agent（本项目）需要三层记忆：

  ┌──────────────────────────────────────┐
  │  短期记忆 (Working Memory)            │
  │  ┌─────────────────────────────────┐ │
  │  │ 最近 N 条消息（token 滑动窗口）  │ │
  │  │ 超出上限 → 自动摘要压缩          │ │
  │  └─────────────────────────────────┘ │
  │          ↓ 写入          ↑ 检索       │
  │  ┌─────────────────────────────────┐ │
  │  │  长期记忆 (Long-Term Memory)     │ │
  │  │  Chroma 向量库，持久化存储        │ │
  │  │  自动存重要信息，语义检索         │ │
  │  └─────────────────────────────────┘ │
  │          ↓                           │
  │  ┌─────────────────────────────────┐ │
  │  │  用户画像 (User Profile)         │ │
  │  │  累积用户偏好、习惯、背景信息     │ │
  │  └─────────────────────────────────┘ │
  └──────────────────────────────────────┘

真实世界的应用：
  - ChatGPT: 自动摘要 + 记忆功能（2024年推出）
  - Claude: 长上下文窗口（200K tokens）+ Project Knowledge
  - Character.AI: 角色长期记忆 + 用户画像

安装依赖：
  pip install tiktoken chromadb
"""

import sys
import os
import json
import hashlib
import re
from datetime import datetime
from typing import Optional


def clean_text(text: str) -> str:
    """清理文本中的非法 Unicode 字符（如 surrogate）"""
    return text.encode("utf-8", errors="replace").decode("utf-8")

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import chromadb
from chromadb.config import Settings

load_dotenv()

DS = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

OLLAMA_EMBED = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

# 记忆存储目录
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "agent_memory")


# ============================================================
# 第一层：Token 管理器 —— 比数消息条数更精确
# ============================================================
# 之前你通过 messages 列表长度判断要不要截断，
# 但不同语言 token 数差异很大（中文一个字≈1-2 tokens，英文更少）
# 用 tiktoken 精确计算可以最大化利用模型上下文窗口。

class TokenManager:
    """Token 计数器 + 滑动窗口管理"""

    def __init__(self, model: str = "deepseek-chat", max_tokens: int = 4000):
        # DeepSeek 兼容 OpenAI tokenizer
        try:
            self.encoder = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_tokens

    def count(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def count_messages(self, messages: list[dict]) -> int:
        """计算 messages 列表的总 token 数"""
        total = 0
        for msg in messages:
            total += self.count(msg.get("content", ""))
            total += 4  # 每条消息的格式开销
        return total + 2  # 回复开头开销

    def fit_window(self, messages: list[dict], reserved: int = 1500) -> list[dict]:
        """保留最近的消息，使总 token 不超过限制（留 reserved 给回复）"""
        limit = self.max_tokens - reserved
        kept = []
        total = 0

        # system prompt 必须保留
        if messages and messages[0]["role"] == "system":
            kept.append(messages[0])
            total += self.count(messages[0]["content"]) + 4
            messages = messages[1:]

        # 从最新到最旧，贪心保留
        for msg in reversed(messages):
            msg_tokens = self.count(msg.get("content", "")) + 4
            if total + msg_tokens > limit:
                break
            kept.insert(1 if kept else 0, msg)  # 保持在 system 之后
            total += msg_tokens

        return kept


# ============================================================
# 第二层：对话摘要器 —— 自动压缩旧消息
# ============================================================
# 当对话超出 token 限制时，不直接丢弃旧消息，
# 而是用 LLM 生成摘要保留关键信息。

class ConversationSummarizer:
    """自动压缩旧对话为摘要"""

    def __init__(self):
        self.summary = ""  # 累积摘要

    def summarize(self, messages: list[dict]) -> str:
        """把一组消息压缩成一段摘要"""
        if not messages:
            return ""

        # 拼接成对话文本
        dialogue = ""
        for m in messages:
            role = "用户" if m["role"] == "user" else "助手"
            content = str(m.get("content", ""))[:500]
            dialogue += f"{role}: {content}\n"

        response = DS.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "system",
                "content": "你是对话摘要助手。把以下对话压缩成一段简洁的摘要，保留关键事实和决策。用中文，100字以内。",
            }, {
                "role": "user",
                "content": f"请摘要以下对话：\n{dialogue}",
            }],
            temperature=0.3,
        )
        return clean_text(response.choices[0].message.content.strip())

    def update(self, messages: list[dict]):
        """增量更新摘要：新摘要 = 旧摘要 + 新对话"""
        new_part = self.summarize(messages)
        if self.summary:
            self.summary = self.summary + " | " + new_part
        else:
            self.summary = new_part


# ============================================================
# 第三层：长期记忆 —— Chroma 向量库持久化
# ============================================================
# 用 Chroma 存储重要信息，每次对话时检索相关的历史记忆。
# 这和项目 16（LangChain RAG）用的是同一个向量库，
# 但这里存的是「用户说过的重要信息」，不是文档。

class LongTermMemory:
    """基于 Chroma 的长期记忆存储"""

    def __init__(self, collection_name: str = "agent_memories"):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=MEMORY_DIR)
        # 每次创建新 collection（如果是新用户就新建）
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._counter = 0

    def _embed(self, text: str) -> list[float]:
        """用 Ollama 生成 Embedding"""
        import requests
        resp = requests.post(OLLAMA_EMBED, json={
            "model": EMBED_MODEL, "prompt": text,
        }, timeout=60)
        return resp.json()["embedding"]

    def store(self, fact: str, metadata: dict = None):
        """存储一条记忆"""
        self._counter += 1
        meta = metadata if metadata else {"source": "conversation"}
        self.collection.add(
            ids=[f"mem_{self._counter}"],
            embeddings=[self._embed(fact)],
            documents=[fact],
            metadatas=[meta],
        )

    def recall(self, query: str, n: int = 3) -> list[str]:
        """语义检索最相关的历史记忆"""
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_embeddings=[self._embed(query)],
            n_results=min(n, self.collection.count()),
        )
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # 只返回相似度 > 0.3 的
        filtered = []
        for doc, dist in zip(docs, distances):
            if dist < 0.7:  # cosine distance < 0.7 ≈ similarity > 0.3
                filtered.append(doc)
        return filtered

    def count(self) -> int:
        return self.collection.count()


# ============================================================
# 第四层：用户画像 —— 累积用户偏好
# ============================================================

class UserProfile:
    """跟踪用户偏好和背景信息"""

    def __init__(self):
        self.name: Optional[str] = None
        self.interests: list[str] = []
        self.facts: dict[str, str] = {}  # key → value

    def update_from_message(self, message: str):
        """简单规则提取用户信息（实际系统会用 NER + 意图识别）"""
        lower = message.lower()
        if any(w in lower for w in ["我叫", "我是", "我的名字是"]):
            # 简单提取名字
            import re
            match = re.search(r"[我叫是]+\s*(\w{2,4})", message)
            if match:
                self.name = match.group(1)

    def get_context(self) -> str:
        """生成用户画像文本，喂给 LLM"""
        parts = []
        if self.name:
            parts.append(f"用户名字: {self.name}")
        if self.interests:
            parts.append(f"兴趣: {', '.join(self.interests)}")
        if self.facts:
            parts.append("已知信息: " + "; ".join(
                f"{k}={v}" for k, v in self.facts.items()
            ))
        return "\n".join(parts) if parts else ""


# ============================================================
# 第五层：MemoryAgent —— 整合所有记忆系统
# ============================================================

class MemoryAgent:
    """
    带完整记忆系统的 Agent。

    处理流程：
      用户消息 →
        1. 更新用户画像
        2. 从长期记忆中检索相关内容
        3. 加载短期记忆（token 滑动窗口）
        4. 如果超出 token 限制 → 摘要压缩旧消息
        5. 组装完整上下文 → 发送给 LLM
        6. 提取回复中的重要信息 → 存入长期记忆
    """

    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or "你是贴心的智能助手，用中文回答。记住用户说过的重要信息。"
        self.token_mgr = TokenManager()
        self.summarizer = ConversationSummarizer()
        self.long_term = LongTermMemory()
        self.profile = UserProfile()

        # 短期记忆：system + 最近消息
        self.messages = [{"role": "system", "content": self.system_prompt}]

        # 对话统计
        self.total_turns = 0
        self.total_summarized = 0

    def _build_context(self, user_message: str) -> list[dict]:
        """组装送给 LLM 的完整上下文"""
        context_msg = []

        # 1. System prompt（基础）
        base_system = self.system_prompt

        # 2. 用户画像
        profile = self.profile.get_context()
        if profile:
            base_system += "\n\n[用户画像]\n" + profile

        # 3. 对话摘要（如果有被压缩的历史）
        if self.summarizer.summary:
            base_system += "\n\n[历史对话摘要]\n" + self.summarizer.summary

        # 4. 长期记忆（从向量库检索到的相关内容）
        recalled = self.long_term.recall(user_message)
        if recalled:
            base_system += "\n\n[相关的历史记忆]\n" + "\n".join(
                f"- {m}" for m in recalled
            )

        context_msg.append({"role": "system", "content": base_system})

        # 5. 近期消息（已经过 token 窗口裁剪）
        context_msg.extend(self.messages[1:])  # 跳过原始 system prompt

        return context_msg

    def _extract_memories(self, user_msg: str, ai_reply: str):
        """从对话中提取值得长期记住的信息"""
        prompt = f"""分析以下对话，提取值得长期记住的事实。只提取客观信息，不要主观评价。
如果没有任何值得记住的事实，返回空列表。

用户: {user_msg[:500]}
助手: {ai_reply[:500]}

返回格式：每行一个事实，用 "- " 开头。没有就返回 "无"。"""

        response = DS.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        result = response.choices[0].message.content.strip()

        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("- ") and line != "- 无":
                fact = line[2:].strip()
                if fact:
                    self.long_term.store(fact)

    def chat(self, user_message: str) -> dict:
        """发送消息，返回 AI 回复和记忆状态"""
        self.total_turns += 1
        self.profile.update_from_message(user_message)

        # 构建上下文
        context = self._build_context(user_message)
        context.append({"role": "user", "content": user_message})

        # Token 检查：如果太长，压缩旧消息
        current_tokens = self.token_mgr.count_messages(context)
        if current_tokens > self.token_mgr.max_tokens - 1500:
            # 压缩 system prompt 之后的前半部分消息
            old_messages = self.messages[1:min(5, len(self.messages))]
            if old_messages:
                print(f"  [记忆] Token 超限({current_tokens})，压缩 {len(old_messages)} 条旧消息...")
                self.summarizer.update(old_messages)
                # 移除被压缩的消息
                self.messages = [self.messages[0]] + self.messages[len(old_messages)+1:]
                self.total_summarized += len(old_messages)
                # 重新构建上下文
                context = self._build_context(user_message)
                context.append({"role": "user", "content": user_message})

        # 调用 LLM
        messages_for_api = context
        response = DS.chat.completions.create(
            model="deepseek-chat", messages=messages_for_api,
        )
        reply = clean_text(response.choices[0].message.content)

        # 存入短期记忆
        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": reply})

        # 提取长期记忆
        self._extract_memories(user_message, reply)

        return {
            "reply": reply,
            "tokens_used": self.token_mgr.count_messages(messages_for_api),
            "short_term_msgs": len(self.messages),
            "long_term_memories": self.long_term.count(),
            "has_summary": bool(self.summarizer.summary),
            "total_summarized": self.total_summarized,
        }


# ============================================================
# 对比：无记忆 vs 有记忆
# ============================================================

def compare_memory():
    """演示记忆系统对连续对话的影响"""
    print("=" * 60)
    print("对比：无记忆 Agent vs 有记忆 Agent")
    print("=" * 60)

    # 场景：用户在多轮对话中逐步透露信息
    conversation = [
        "你好！我叫小明，我是计算机专业的大一学生。",
        "我正在学 Python，你能推荐一些学习资源吗？",
        "我之前说的我的专业是什么？我上几年级？",
    ]

    # --- 无记忆模式 ---
    print("\n【无记忆 Agent】每轮只看到当前消息：")
    print("-" * 40)
    for msg in conversation:
        print(f"\n用户: {msg}")
        response = DS.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "简洁回答，20字以内。"},
                {"role": "user", "content": msg},
            ],
        )
        print(f"AI: {response.choices[0].message.content}")

    # --- 有记忆模式 ---
    print("\n\n【有记忆 Agent】多轮对话 + 用户画像：")
    print("-" * 40)
    agent = MemoryAgent()
    for msg in conversation:
        print(f"\n用户: {msg}")
        result = agent.chat(msg)
        print(f"AI: {result['reply']}")
        print(f"  (短期: {result['short_term_msgs']}条, "
              f"长期: {result['long_term_memories']}个记忆, "
              f"Token: {result['tokens_used']})")


# ============================================================
# 交互演示
# ============================================================

def interactive_demo():
    """交互式演示记忆系统"""
    print("\n" + "=" * 60)
    print("MemoryAgent —— 三层记忆系统")
    print("=" * 60)
    print("试试连续多轮对话，看 Agent 记住你的信息：")
    print("  - '我叫XX，我喜欢XX'")
    print("  - '我刚才说我叫什么？'")
    print("  - '你还记得我喜欢什么吗？'")
    print("  - /stats 查看记忆状态")
    print("  - quit 退出")
    print("=" * 60)

    agent = MemoryAgent()

    while True:
        user_input = input("\n你: ")
        if user_input.lower() == "quit":
            print(f"\n会话总结：共 {agent.total_turns} 轮对话，"
                  f"长期记忆 {agent.long_term.count()} 条，"
                  f"压缩 {agent.total_summarized} 条旧消息")
            break

        if user_input == "/stats":
            print(f"  短期记忆: {len(agent.messages)} 条消息")
            print(f"  长期记忆: {agent.long_term.count()} 条")
            print(f"  对话摘要: {'有' if agent.summarizer.summary else '无'}")
            profile = agent.profile.get_context()
            print(f"  用户画像: {profile if profile else '未建立'}")
            continue

        result = agent.chat(user_input)
        print(f"AI: {result['reply']}")
        if result["has_summary"]:
            print(f"  💾 记忆状态: 短期{result['short_term_msgs']}条 | "
                  f"长期{result['long_term_memories']}个 | "
                  f"已压缩{result['total_summarized']}条")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Agent 记忆系统 —— 三层记忆架构")
    print("=" * 60)
    print("""
三层记忆：
  短期记忆 → 最近 N 条消息，token 滑动窗口
  长期记忆 → Chroma 向量库，语义检索历史信息
  对话摘要 → token 超限时自动压缩，不丢上下文

用户画像 → 自动累积用户偏好和背景
""")

    # 先跑对比演示
    compare_memory()

    # 交互式体验
    interactive_demo()
