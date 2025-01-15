import torch
import cv2
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np
import random
import os
import gdown

# Функция загрузки весов
def download_weights(url, output_path):
    """
    Скачивает веса модели с Google Диска.
    :param url: Ссылка на файл (публичный доступ).
    :param output_path: Локальный путь для сохранения.
    """
    if not os.path.exists(output_path):
        print(f"Скачивание весов с {url}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"Веса уже загружены: {output_path}")

# Настройки
weight_url = "https://drive.google.com/file/d/1aNofUExxia1-ENFwwGgjBr13T3dQWY7I/view?usp=drive_link"  # Ссылка на веса модели
local_weight_path = "checkpoints/distil_checkpoints.pth"  # Локальный путь для сохранения весов
input_video = "cource_project/videos/example.mp4"  # Путь к входному видео
output_video = "cource_project/videos/output_video_distilled.mp4"  # Путь к выходному видео
score_threshold = 0.5  # Порог вероятности

# Загрузка весов
download_weights(weight_url, local_weight_path)

# Инициализация устройства
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используемое устройство: {device}")

# Загрузка модели RetinaNet
model = retinanet_resnet50_fpn(pretrained=False, num_classes=1)
model.load_state_dict(torch.load(local_weight_path, map_location=device))
model.to(device)
model.eval()

# Функция для обработки кадров и добавления предсказаний
def process_frame(frame, model, score_threshold):
    # Преобразование кадра в тензор
    image_tensor = F.to_tensor(frame).unsqueeze(0).to(device)

    # Предсказания модели
    with torch.no_grad():
        outputs = model(image_tensor)

    # Извлечение результатов
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    # Фильтрация боксов по порогу вероятности
    indices = np.where(scores > score_threshold)[0]
    boxes = boxes[indices]
    scores = scores[indices]

    return boxes, scores

# Открытие входного видео
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise RuntimeError(f"Не удалось открыть видео: {input_video}")

# Получение параметров видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Обработка видео
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Обработка кадра
    boxes, scores = process_frame(frame, model, score_threshold)

    # Отрисовка предсказаний
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box.astype(int)
        color = (0, 255, 0)  # Зеленый цвет
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"Conf: {score:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)

cap.release()
out.release()

print(f"Готово! Результат сохранен в {output_video}")
