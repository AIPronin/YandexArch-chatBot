FROM python:3.10-slim

WORKDIR /app

# Устанавливаем зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем Python пакеты
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY rag_bot.py .
COPY test_setup.py .

ENV GGML_CUDA=0
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=off"

# Создаем папки
RUN mkdir -p /app/models /app/cache

# Запуск
CMD ["python", "rag_bot.py"]