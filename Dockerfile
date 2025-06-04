# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Установим компилятор, иначе не соберётся numpy 1.26.4
RUN apt-get update && apt-get install -y build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Переменные окружения
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/model_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/model_cache \
    HF_HOME=/app/model_cache

COPY requirements.txt .

# Устанавливаем все зависимости — numpy будет 1.26.4
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

COPY . .

# Монтируем модельный кэш
VOLUME ["/app/model_cache"]

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]