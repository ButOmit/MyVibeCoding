"""
视觉 Agent —— 让 AI 看懂图片
================================
核心思路：大语言模型不会"看"，但可以调用视觉模型当眼睛。

架构：
  主对话: DeepSeek（负责理解意图、调度工具）→ 有 Function Calling
  视觉工具: llava:7b（负责看图说话）→ 通过 Ollama API 调用

对比：
  纯文本模型: "我是一张图片" → 只能猜
  + 视觉模型:   读像素 → 真正"看到"内容

适用场景：图片问答、图表理解、OCR文字识别、截图分析
"""
import sys
import os
import json
import base64
import requests

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 两个后端：DeepSeek（对话）+ llava（视觉）
# ============================================================

deepseek = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

OLLAMA_API = "http://localhost:11434/api/generate"


def ask_llava(image_path: str, question: str) -> str:
    """调用 llava:7b 分析图片"""
    image_path = image_path.replace("\\", "/")
    if not os.path.exists(image_path):
        return f"图片不存在：{image_path}"

    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        prompt = f"Answer the following question based on the image. Be specific and concise.\n\nQuestion: {question}"

        resp = requests.post(OLLAMA_API, json={
            "model": "llava:7b",
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
        }, timeout=120)
        return resp.json()["response"].strip()
    except Exception as e:
        return f"图片分析失败：{e}"


# ============================================================
# Agent 类
# ============================================================

SYSTEM_PROMPT = """你是一个视觉 AI 助手。你可以分析图片内容。

当用户给你一张图片路径并提问时：
1. 用 analyze_image 工具分析图片
2. 根据分析结果回答用户的问题
3. 如果图片路径不存在，提示用户检查路径

用中文回答，语气友好。"""


class VisionAgent:
    def __init__(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "analyze_image",
                    "description": "分析一张图片的内容。传入图片路径和你想知道的问题，返回图片描述。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_path": {"type": "string", "description": "图片文件路径"},
                            "question": {"type": "string", "description": "关于图片的问题，例如'图片里有什么'"},
                        },
                        "required": ["image_path", "question"],
                    },
                },
            },
        ]

        response = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            tools=tools,
        )
        reply = response.choices[0].message

        if reply.tool_calls:
            for tool_call in reply.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if func_name == "analyze_image":
                    print(f"  [视觉工具: 正在分析 {func_args['image_path']}...]")
                    tool_result = ask_llava(**func_args)
                else:
                    tool_result = "未知工具"

                self.messages.append({
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [{
                        "id": tool_call.id, "type": "function",
                        "function": {"name": func_name, "arguments": tool_call.function.arguments},
                    }],
                })
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

            response = deepseek.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages,
            )
            reply = response.choices[0].message

        self.messages.append({"role": "assistant", "content": reply.content})
        return reply.content


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    agent = VisionAgent()
    test_img = os.path.join(os.path.dirname(__file__), "test_image.png").replace("\\", "/")

    print("=" * 50)
    print("视觉 Agent（DeepSeek 对话 + llava:7b 看图）")
    print(f"测试图片: {test_img}")
    print("试试：")
    print(f"  '帮我看看 {test_img} 这张图里有什么'")
    print("  '这张图里有几个形状，分别是什么颜色'")
    print("你也可以拖入任何图片，问'这张图里有什么'")
    print("输入 'quit' 退出")
    print("=" * 50)

    while True:
        user_input = input("\n你: ")
        if user_input.lower() == "quit":
            print("再见！")
            break
        reply = agent.chat(user_input)
        print(f"AI: {reply}")
