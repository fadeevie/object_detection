import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from tqdm import tqdm
from ultralytics import YOLO
import gdown
import zipfile

# Устройство для вычислений
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используемое устройство: {device}")

# Функция для скачивания датасета из Google Drive
def download_and_unzip_dataset(url, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    zip_path = os.path.join(output_dir, "dataset.zip")
    print(f"Скачивание датасета из {url}...")
    gdown.download(url, zip_path, quiet=False)

    print(f"Распаковка датасета в {output_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("Датасет успешно загружен и распакован.")

# Пути к данным
DATASET_URL = "https://drive.google.com/uc?id=1nGUcc7bbz8Y1MHl1Wp-znNlFQByjwzVs"  
datasets_dir = "./data"
train_img_dir = os.path.join(datasets_dir, "images/train")
train_ann_file = os.path.join(datasets_dir, "annotations/instances_train_fixed.json")
val_img_dir = os.path.join(datasets_dir, "images/val")
val_ann_file = os.path.join(datasets_dir, "annotations/instances_val_fixed.json")

# Скачивание датасета (если необходимо)
if not os.path.exists(datasets_dir):
    download_and_unzip_dataset(DATASET_URL, datasets_dir)

# Пути к весам моделей
teacher_weights = "./checkpoints/yolo_teacher_weights.pt"
student_weights = "./checkpoints/retinanet_student_weights.pth"

# Загрузка учителя (YOLO)
teacher_model = YOLO(teacher_weights)
teacher_model.to(device)
teacher_model.eval()

# Функция преобразования предсказаний YOLO в формат RetinaNet
def yolo_to_retinanet_format(yolo_results):
    retinanet_targets = []
    for result in yolo_results:
        boxes = result.boxes.xyxy.cpu()  # Координаты боксов
        labels = torch.zeros(len(boxes), dtype=torch.int64)  # Один класс: 'окно'
        retinanet_targets.append({"boxes": boxes, "labels": labels})
    return retinanet_targets

# Загрузка студента (RetinaNet)
student_model = retinanet_resnet50_fpn(pretrained=False, num_classes=1).to(device)
if os.path.exists(student_weights):
    student_model.load_state_dict(torch.load(student_weights, map_location=device))
    print("Веса студента загружены.")
else:
    print("Студент обучается с нуля.")

# Оптимизатор
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4, weight_decay=1e-4)

# Функция преобразования аннотаций COCO в формат RetinaNet
def coco_to_retinanet_format(targets):
    retinanet_targets = []
    for target in targets:
        if len(target) == 0:  # Пропуск пустых аннотаций
            retinanet_targets.append({"boxes": torch.empty((0, 4), dtype=torch.float32), "labels": torch.empty((0,), dtype=torch.int64)})
            continue

        boxes, labels = [], []
        for obj in target:
            x_min, y_min, w, h = obj["bbox"]
            x_max = x_min + w
            y_max = y_min + h
            if w > 0 and h > 0:  # Убираем некорректные боксы
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(0)  # Один класс: 'окно'

        retinanet_targets.append({"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64)})
    return retinanet_targets

# Датасеты и загрузчики данных
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

train_dataset = CocoDetection(train_img_dir, train_ann_file)
val_dataset = CocoDetection(val_img_dir, val_ann_file)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Функция обучения
def train_distillation(teacher_model, student_model, train_loader, optimizer, device, epochs=10):
    student_model.train()
    teacher_model.eval()

    for epoch in range(epochs):
        total_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{epochs}"):

            images = [F.to_tensor(img).to(device) for img in images]

            targets = coco_to_retinanet_format(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                yolo_results = teacher_model(images)
                teacher_targets = yolo_to_retinanet_format(yolo_results)
                teacher_targets = [{k: v.to(device) for k, v in t.items()} for t in teacher_targets]

            optimizer.zero_grad()
            outputs = student_model(images, teacher_targets)
            loss = sum(loss for loss in outputs.values())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Эпоха {epoch + 1}: Средняя потеря: {avg_loss:.4f}")

        save_path = f"./checkpoints/retinanet_distilled_epoch_{epoch + 1}.pth"
        torch.save(student_model.state_dict(), save_path)
        print(f"Сохранены веса: {save_path}")

# Обучение модели
train_distillation(teacher_model, student_model, train_loader, optimizer, device, epochs=10)

# Сохранение финальной модели
final_weights_path = "./checkpoints/retinanet_distilled_final.pth"
torch.save(student_model.state_dict(), final_weights_path)
print(f"Финальные веса сохранены: {final_weights_path}")
