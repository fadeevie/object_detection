# **Object Detection Project**

## **Описание проекта**
Данный проект посвящен обучению и инференсу моделей для задачи детекции окон. Реализованы следующие модели:
- YOLOv9
- YOLOv10
- Deformable DETR
- RetinaNet
- Модель дистилляции (YOLO → RetinaNet)

Данные хранятся на Google Drive и автоматически загружаются перед обучением и инференсом.

---

## **Структура проекта**
- `checkpoints/` — сохранённые веса моделей.
- `config/` — YAML-файлы конфигураций.
- `scripts/` — скрипты для инференса моделей.
- `src/` — исходный код для обучения.
- `requirements.txt` — зависимости для работы проекта.
- `Dockerfile` — инструкции для создания Docker-образа.

---

## **Установка**
### **Клонирование репозитория**
После создания репозитория выполните команду:
```bash
git clone https://github.com/fadeevie/course_proj.git
cd course_proj

### **Создание Docker-образа**
docker build -t object-detection .

## **Данные**
Для работы проекта требуется два набора данных:

1. YOLO Dataset для моделей YOLOv9 и YOLOv10:
https://drive.google.com/drive/folders/1fzEL3dfwFDcfulWE_QWPlft_edhzXs8e?usp=drive_link

2. COCO Dataset для Deformable DETR, RetinaNet и дистилляции:
https://drive.google.com/drive/folders/1nGUcc7bbz8Y1MHl1Wp-znNlFQByjwzVs?usp=drive_link

## **Запуск**
### **Обучение**
Используйте команды для запуска обучения каждой модели.

### **YOLOv9**
docker run --rm object-detection python src/train_yolo9.py --config config/config_yolo9.yaml

### **YOLOv10**
docker run --rm object-detection python src/train_yolo10.py --config config/config_yolo10.yaml

### **Deformable DETR**
docker run --rm object-detection python src/train_deformable_detr.py --config config/config_deformable_detr.py

### **RetinaNet**
docker run --rm object-detection python src/train_retinanet.py

### **Модель дистилляции **
docker run --rm object-detection python src/train_distil.py

### **Инференс**
Для инференса видео используйте скрипты в scripts/.
Для тестирования инференса моделей вы можете скачать видео по ссылке:
https://drive.google.com/file/d/1MOmRvJntyMArd5zel6xTuxCIHoq0-E8i/view?usp=sharing
После скачивания поместите видео в папку `videos` в проекте.

### **YOLOv9**
docker run --rm object-detection python src/inference/inference_yolo.py \
    --video videos/example_video.mp4 \
    --output yolo9_output.mp4

### **YOLOv10**
docker run --rm object-detection python src/inference/inference_yolo.py \
    --video videos/example_video.mp4 \
    --output yolo10_output.mp4

### **RetinaNet**
docker run --rm object-detection python src/inference/inference_retinanet.py \
    --video videos/example_video.mp4 \
    --output retinanet_output.mp4

### **Модель дистилляции **
docker run --rm object-detection python src/inference/inference_distil.py \
    --video videos/example_video.mp4 \
    --output distil_output.mp4
