# AI Agent 从零开始学习之路

> 大一暑假目标：AI Agent 实习岗。从第一行 API 调用到多工具智能体，纯动手项目驱动。

## 技能树

```
01 → 02 → 03 → 04       基础：API调用 → 多轮对话 → 工具调用 → 框架重构
              ↓
         05 + 06         扩展：文件读写 + 网页搜索
              ↓
       07 + 08 + 09      整合：Gradio网页助手 + 天气 + 关键词RAG
              ↓
            10           进阶：向量RAG（TF-IDF → 语义搜索）
              ↓
        11 + 12          本地模型 + 视觉理解（Ollama + llava）
              ↓
            13           体验：流式输出（打字机效果）
              ↓
            14           升级：Embedding 语义搜索（神经网络向量）
              ↓
            15           收尾：多 Agent 协作（团队分工）
```

## 项目清单

| # | 文件 | 学到什么 | 核心概念 |
|---|------|---------|---------|
| 01 | `01-first-chat.py` | 第一次调用 AI API | OpenAI SDK, API Key 安全 |
| 02 | `02-chat-with-memory.py` | 多轮对话记忆 | messages 列表, 上下文管理 |
| 03 | `03-agent-with-tools.py` | AI 调用工具 | Function Calling, JSON Schema |
| 03 | `03-practice.py` | 亲手给 AI 加工具 | 独立实现 `get_current_time` |
| 04 | `04-agent-framework.py` | 重构可复用框架 | Agent 类, 工具注册表模式 |
| 05 | `05-file-agent.py` | AI 读写文件 | 跨文件 import, 工作目录 |
| 06 | `06-web-agent.py` | AI 搜索网页 | DuckDuckGo, HTTP 请求 |
| 07 | `07-personal-assistant.py` | Gradio 网页界面 | 8 工具集成, 手机可访问 |
| 07 | `07-practice-weather-agent.py` | 整合天气工具 | API 解析, JSON 处理 |
| 08 | `08-weather-agent.py` | 实时天气查询 | wttr.in API, requests |
| 09 | `09-rag-agent.py` | 文档检索 RAG | 分块→检索→增强生成 |
| 10 | `10-vector-rag-agent.py` | 向量语义搜索 | TF-IDF, 余弦相似度, 中文n-gram |
| 11 | `11-local-agent.py` | 本地模型部署 | Ollama, OpenAI兼容接口, 免费AI |
| 12 | `12-vision-agent.py` | 图片理解 Agent | llava:7b, 视觉模型, 多模态 Agent |
| 13 | `13-streaming-agent.py` | 流式输出 | stream=True, generator, 打字机效果 |
| 14 | `14-embedding-rag-agent.py` | Embedding 语义搜索 | nomic-embed-text, 神经网络向量, 跨语言理解 |
| 15 | `15-multi-agent.py` | 多 Agent 协作 | 团队分工, 协调员+研究员+程序员, 任务派发 |

## 技术栈

- **语言**: Python 3
- **AI API**: DeepSeek (OpenAI 兼容)
- **前端**: Gradio (ChatInterface)
- **搜索**: TF-IDF (scikit-learn), DuckDuckGo
- **数据**: wttr.in 天气 API, requests

## 运行方式

```bash
pip install openai python-dotenv gradio scikit-learn ddgs requests
# 创建 .env 文件，写入 DEEPSEEK_API_KEY=你的key
python 07-personal-assistant.py  # 网页版
python 10-vector-rag-agent.py    # 向量RAG版
```

## 下一步

- [x] 本地模型部署（Ollama + qwen2.5:3b）
- [x] 多模态图片理解（llava:7b 视觉 Agent）
- [x] 流式输出（AI 打字效果）
- [x] 多 Agent 协作（团队分工）
- [x] Embedding 语义搜索（真正的向量模型）
