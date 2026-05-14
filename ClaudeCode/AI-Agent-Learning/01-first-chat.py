"""
你的第一个 AI 程序 —— 和 DeepSeek 对话
======================================
这个程序做的事情：发送一句话给 DeepSeek，然后打印它的回复。
"""
import sys
import os

# 修复 Windows 终端中文乱码问题
sys.stdout.reconfigure(encoding="utf-8")

# 第1步：导入库（就像你写作文前先拿笔）
from openai import OpenAI
from dotenv import load_dotenv

# 从 .env 文件中读取 API Key（不写死在代码里，防止泄露）
load_dotenv()

# 第2步：准备"钥匙"和"地址"
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从环境变量读取，安全！
    base_url="https://api.deepseek.com",
)

# 第3步：发送消息，获取回复
# messages 就是你发给 AI 的话
response = client.chat.completions.create(
    model="deepseek-chat",           # 使用 DeepSeek 的聊天模型
    messages=[
        {"role": "system", "content": "你是一个友好的助手，喜欢用简洁的方式回答问题。"},
        {"role": "user", "content": "你好！请用一句话介绍你自己。"},
    ],
)

# 第4步：把回复打印出来
# response.choices[0].message.content 就是 AI 回复的内容
print("AI 回复：")
print(response.choices[0].message.content)
