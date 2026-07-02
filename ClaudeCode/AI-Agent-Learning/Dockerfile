# ============================================
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

# 复制项目文件（云端版，零 Ollama 依赖）
COPY personal_assistant_v3_cloud.py .
COPY document_library/ ./document_library/

# 创建数据目录
RUN mkdir -p agent_workspace

# 声明端口
EXPOSE 8000

# 启动命令
CMD ["python", "-u", "personal_assistant_v3_cloud.py"]
