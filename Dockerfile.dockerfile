# Базовый образ с Python и поддержкой PyTorch
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    wget \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Установка Python-зависимостей
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование проекта
COPY . /app

# Установка ультраконфигураций для YOLO
RUN python -c "import ultralytics; ultralytics.checks()"

# Установка mmdetection (если используется)
RUN git clone https://github.com/open-mmlab/mmdetection.git /app/mmdetection && \
    cd /app/mmdetection && pip install -e .

# Путь к данным
ENV DATASET_DIR="/app/data"

# Стандартная команда по умолчанию
CMD ["python", "src/train.py", "--config", "config/config.yaml"]
