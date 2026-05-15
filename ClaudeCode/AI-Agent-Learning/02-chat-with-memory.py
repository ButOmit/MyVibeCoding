"""
带记忆的 AI 对话 —— 让 DeepSeek 记住你们聊过什么
==================================================
原理：把每句话都存进一个列表，每次发送时把整个列表一起发给 AI。
"""
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ============ 核心：对话记忆 ============
# 用一个列表存所有消息，每条消息是一个字典 {"role": "...", "content": "..."}
messages = [
    {"role": "system", "content": "你是一个友好的助手，喜欢用简洁的方式回答问题。"},
]

print("=" * 50)
print("AI 聊天室（输入 'quit' 退出，输入 'clear' 清空记忆）")
print("=" * 50)

while True:
    # 第1步：获取用户输入
    user_input = input("\n你: ")

    if user_input.lower() == "quit":
        print("再见！")
        break

    if user_input.lower() == "clear":
        # 清空对话记忆，只保留 system 消息
        messages = [messages[0]]
        print("(对话记忆已清空)")

    # 第2步：把用户说的话加入记忆
    messages.append({"role": "user", "content": user_input})

    # 第3步：把整个记忆发给 AI
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )

    # 第4步：取出 AI 的回复
    ai_reply = response.choices[0].message.content

    # 第5步：把 AI 的回复也加入记忆
    messages.append({"role": "assistant", "content": ai_reply})

    # 第6步：打印
    print(f"AI: {ai_reply}")

    # 显示当前记忆中有多少条消息
    print(f"(已记住 {len(messages) - 1} 条消息)")
