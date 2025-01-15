# Импорт библиотек
from ultralytics import YOLO
import cv2
import random
import os
import gdown

# Функция для загрузки весов с Google Диска
def download_weights(url, output_path):
    """
    Скачивает веса модели с Google Диска.
    :param url: Ссылка на файл (публичный доступ).
    :param output_path: Локальный путь для сохранения.
    """
    if not os.path.exists(output_path):
        print(f"Скачивание весов модели с {url}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"Веса уже загружены: {output_path}")

# Функция для обработки видео
def track_video(model, source_video_path, output_file="results/output_video.mp4"):
    # Открытие видеофайла для чтения
    video_capture = cv2.VideoCapture(source_video_path)

    if not video_capture.isOpened():
        raise RuntimeError(f"Unable to open the input video file: {source_video_path}")

    # Получение параметров видео
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Настройка видеокодека и выходного видеофайла
    video_codec = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, video_codec, frame_rate, (frame_width, frame_height))

    # Цикл для чтения и обработки кадров видео
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Обработка кадра с использованием модели YOLO
        results = model.track(frame, iou=0.5, conf=0.5, persist=True, imgsz=640, verbose=False, tracker="botsort.yaml")
        if results[0].boxes.id is not None:
            bounding_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            object_ids = results[0].boxes.id.cpu().numpy().astype(int)

            # Отрисовка bounding boxes и индексов объектов на кадре
            for bbox, obj_id in zip(bounding_boxes, object_ids):
                random.seed(int(obj_id))
                box_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                # Отображение прямоугольника и индекса объекта на кадре
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 2)
                cv2.putText(frame, f"ObjID: {obj_id}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        video_writer.write(frame)

    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()

# Ссылки и настройки
weight_url = "https://drive.google.com/file/d/1Jah10FxmKPnDnD6LEWoB-6ZacpcIU1M5/view?usp=drive_link"  # Ссылка на веса YOLO
local_weight_path = "checkpoints/yolo_best.pt"  # Локальный путь для сохранения весов
input_video = "videos/example.mp4"  # Входное видео
output_video = "./output_video.mp4"  # Результирующее видео

# Загрузка весов
download_weights(weight_url, local_weight_path)

# Загрузка и подготовка модели YOLO
model = YOLO(local_weight_path)
model.fuse()

# Вызов функции для обработки видео
track_video(model, source_video_path=input_video, output_file=output_video)
