"""
Docker 部署实战 —— 把 AI 助手打包上线
========================================
之前你的 Agent 只能在本机跑，面试官看不到。
Docker 把它打包成标准化容器，一键部署到任何服务器。

核心概念（30秒入门）：

  传统部署：
    你的电脑 → 装 Python → pip install → 改配置 → 启动
    别人电脑 → 装 Python → pip install → 改配置 → 启动 ← 环境不一样，可能报错

  Docker 部署：
    你写 Dockerfile（配方） → docker build（打包） → docker run（启动）
    任何人拿到镜像 → docker run → 一模一样的环境 → 绝对不报错

  类比：
    Docker 镜像 = 装好系统的 U 盘
    Docker 容器 = 插上 U 盘启动的电脑
    docker-compose = 同时管理多台电脑的遥控器

本项目文件：
  24-docker-deploy.py       ← 本文件（概念讲解 + 演示）
  Dockerfile                ← 构建 AI 助手镜像的配方
  docker-compose.yml        ← 一键启动 AI 助手 + 依赖服务
  .dockerignore             ← 告诉 Docker 哪些文件不要打包
  .env.example              ← 环境变量模板

安装依赖：
  安装 Docker Desktop: https://www.docker.com/products/docker-desktop
"""

# ============================================================
# Docker 只是壳子，业务逻辑还是你写的 v3.0
# 这里演示 Docker 部署时的几个关键点
# ============================================================

print("=" * 60)
print("Docker 部署实战 —— 把 AI 助手打包成容器")
print("=" * 60)

print("""
┌─────────────────────────────────────────────────────────────┐
│                    Docker 架构全景图                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   docker-compose.yml  ← 编排文件，定义所有服务                │
│   ┌─────────────────────────────────────────┐               │
│   │  服务1: ai-assistant (FastAPI)           │               │
│   │  ┌───────────────────────────────────┐  │               │
│   │  │  Dockerfile 构建的镜像              │  │               │
│   │  │  ├── Python 3.11                  │  │               │
│   │  │  ├── requirements.txt → 所有依赖   │  │               │
│   │  │  ├── personal_assistant_v3.py     │  │               │
│   │  │  └── 环境变量 (.env)               │  │               │
│   │  └───────────────────────────────────┘  │               │
│   │  端口: 8000:8000 (宿主机:容器)            │               │
│   └─────────────────────────────────────────┘               │
│                                                             │
│   服务2: ollama (可选，Embedding 模型)                        │
│   ┌─────────────────────────────────────────┐               │
│   │  镜像: ollama/ollama:latest              │               │
│   │  端口: 11434:11434                       │               │
│   │  卷: ./ollama_data:/root/.ollama         │               │
│   └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")

print("""
部署只需三步：

  1. 装 Docker Desktop（一次性）
  2. 配置 .env 文件（DEEPSEEK_API_KEY=sk-xxx）
  3. 终端运行: docker-compose up -d

然后访问 http://localhost:8000 —— 和本地跑一模一样！

面试话术：
  "我把 AI 助手用 Docker 打包了，docker-compose 一键启动。
   环境完全隔离，不依赖本机 Python 版本。
   可以部署到任何云服务器（阿里云/腾讯云/AWS）。"
""")


# ============================================================
# 演示：自动生成 Docker 配置文件
# ============================================================

def generate_docker_files():
    """生成完整的 Docker 部署文件集"""
    import os
    base = os.path.dirname(__file__)

    files = {}

    # ---- Dockerfile ----
    files["Dockerfile"] = '''# ============================================
# AI 助手 v3.0 Docker 镜像
# ============================================
# 基于官方 Python 镜像（轻量版）
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 先复制依赖文件（利用 Docker 缓存层，改代码不用重装依赖）
COPY requirements.txt .

# 安装依赖（用清华镜像加速）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目文件
COPY personal_assistant_v3.py .
COPY document_library/ ./document_library/

# 创建数据目录
RUN mkdir -p agent_workspace

# 声明端口
EXPOSE 8000

# 启动命令
CMD ["python", "-u", "personal_assistant_v3.py"]
'''

    # ---- docker-compose.yml ----
    files["docker-compose.yml"] = '''# ============================================
# AI 助手 + Ollama 一键编排
# ============================================
# 使用: docker-compose up -d
# 停止: docker-compose down

version: "3.8"

services:
  # ---- AI 助手主服务 ----
  ai-assistant:
    build: .
    container_name: ai-assistant-v3
    ports:
      - "8000:8000"
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - OLLAMA_HOST=ollama  # 容器内通过服务名访问
    volumes:
      - ./agent_workspace:/app/agent_workspace  # 持久化记忆数据
    restart: unless-stopped
    depends_on:
      - ollama

  # ---- Ollama 向量模型服务 ----
  ollama:
    image: ollama/ollama:latest
    container_name: ollama-embed
    ports:
      - "11434:11434"
    volumes:
      - ./ollama_data:/root/.ollama  # 持久化模型文件
    restart: unless-stopped
    # 启动后自动拉取 embedding 模型
    entrypoint: ["/bin/sh", "-c"]
    command:
      - |
        ollama serve &
        sleep 5
        ollama pull nomic-embed-text
        tail -f /dev/null
'''

    # ---- .dockerignore ----
    files[".dockerignore"] = '''# Git
.git
.gitignore

# Python
__pycache__
*.pyc
.venv
venv/

# 环境（安全！不要把 .env 打进镜像）
.env
*.env

# 工作数据
chroma_db/
agent_workspace/
ollama_data/

# IDE
.vscode/
.idea/

# 系统
.DS_Store
Thumbs.db
'''

    # ---- requirements.txt ----
    files["requirements.txt"] = '''fastapi>=0.100.0
uvicorn>=0.23.0
openai>=1.0.0
python-dotenv>=1.0.0
numpy>=1.24.0
requests>=2.31.0
jieba>=0.42.0
rank-bm25>=0.2.0
tiktoken>=0.5.0
chromadb>=0.4.0
'''

    # ---- .env.example ----
    files[".env.example"] = '''# ============================================
# AI 助手环境变量配置
# ============================================
# 复制此文件为 .env 并填入你的 Key
# cp .env.example .env

# DeepSeek API Key（必填）
DEEPSEEK_API_KEY=sk-your-key-here

# Ollama 地址（默认本机）
OLLAMA_HOST=localhost
'''

    # ---- 写入文件 ----
    for filename, content in files.items():
        filepath = os.path.join(base, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    print("[OK] Docker 部署文件已生成：")
    for f in files:
        fp = os.path.join(base, f)
        size = os.path.getsize(fp)
        print(f"  {f} ({size}B)")
    print()
    print("使用方法：")
    print("  1. cp .env.example .env    # 填入你的 API Key")
    print("  2. docker-compose up -d    # 一键启动")
    print("  3. open http://localhost:8000")


if __name__ == "__main__":
    generate_docker_files()

    print("\n" + "=" * 60)
    print("Dockerfile 逐行解读（面试时能讲这个就够了）：")
    print("=" * 60)
    print("""
  FROM python:3.11-slim
    → 基于 Python 官方镜像。slim = 精简版，体积小

  COPY requirements.txt .
  RUN pip install -r requirements.txt
    → 先装依赖再拷代码。为什么？Docker 有"层缓存"——
      改代码时前几步不变，不用重装依赖，构建快 10 倍

  COPY personal_assistant_v3.py .
    → 只有一行代码一行配置，不是一堆脚本

  EXPOSE 8000
    → 告诉 Docker 这个容器用 8000 端口

  docker-compose 的 depends_on + restart
    → 服务挂了自动重启，保证高可用

  面试官最在意什么？
    1. 你知道层缓存优化构建速度
    2. 你知道 .dockerignore 排除 .env 保护密钥
    3. 你知道 docker-compose 编排多服务
    4. 你知道 volumes 持久化数据
""")
