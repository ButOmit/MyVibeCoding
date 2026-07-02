FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY personal_assistant_v3_cloud.py .

EXPOSE 8000
CMD ["uvicorn", "personal_assistant_v3_cloud:app", "--host", "0.0.0.0", "--port", "8000"]
