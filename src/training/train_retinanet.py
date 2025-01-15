import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torchvision.ops.boxes import box_iou
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Устройство для вычислений
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Устройство: {device}")

# Пути к данным
train_img_dir = "/app/data/images/train"
train_ann_file = "/app/data/annotations/instances_train.json"
val_img_dir = "/app/data/images/val"
val_ann_file = "/app/data/annotations/instances_val.json"

# Подготовка аннотаций COCO для RetinaNet
def collate_fn(batch):
    return tuple(zip(*batch))

def coco_to_retinanet_format(targets):
    retinanet_targets = []
    for target in targets:
        if len(target) == 0:  # Пропуск пустых аннотаций
            retinanet_targets.append({"boxes": torch.empty((0, 4), dtype=torch.float32), "labels": torch.empty((0,), dtype=torch.int64)})
            continue

        boxes = []
        for obj in target:
            x_min, y_min, w, h = obj["bbox"]
            x_max = x_min + w
            y_max = y_min + h
            boxes.append([x_min, y_min, x_max, y_max])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.zeros(len(boxes), dtype=torch.int64)  # Все объекты принадлежат классу 0

        retinanet_targets.append({"boxes": boxes, "labels": labels})
    return retinanet_targets

# Загрузка данных COCO
train_dataset = CocoDetection(
    root=train_img_dir,
    annFile=train_ann_file,
    transforms=lambda img, target: (F.to_tensor(img), target),
)
val_dataset = CocoDetection(
    root=val_img_dir,
    annFile=val_ann_file,
    transforms=lambda img, target: (F.to_tensor(img), target),
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Инициализация RetinaNet
model = retinanet_resnet50_fpn(pretrained=False, num_classes=1).to(device)

# Оптимизатор и scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=4, verbose=True)

# Метрики
train_map_metric = MeanAveragePrecision()
val_map_metric = MeanAveragePrecision()

def train_with_metrics(model, train_loader, optimizer, device, scheduler, epochs=60):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, targets in tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{epochs}"):
            images = list(img.to(device) for img in images)
            targets = coco_to_retinanet_format(targets)

            # Проверка targets
            if not targets:
                print("Ошибка: targets пусты!")
                continue
            if len(images) != len(targets):
                print(f"Пропуск батча: количество изображений ({len(images)}) не совпадает с аннотациями ({len(targets)})")
                continue

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Вычисление потерь
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Эпоха {epoch + 1}: Средняя потеря: {avg_loss:.4f}")

        if scheduler:
            scheduler.step(avg_loss)

def validate_with_metrics(model, val_loader, device):
    model.eval()
    val_map_metric.reset()
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Валидация"):
            images = list(image.to(device) for image in images)
            targets = coco_to_retinanet_format(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            val_map_metric.update(outputs, targets)

    val_map = val_map_metric.compute()
    print(f"Результаты метрик на валидации:")
    print(f"mAP@0.50: {val_map['map_50']:.4f}")
    print(f"mAP@0.75: {val_map['map_75']:.4f}")
    print(f"mAP@0.50:0.95: {val_map['map']:.4f}")

    return val_map

# Запуск обучения
train_with_metrics(model, train_loader, optimizer, device, scheduler, epochs=60)
validate_with_metrics(model, val_loader, device)

# Сохранение модели
torch.save(model.state_dict(), "/app/checkpoints/retinanet_model.pth")
print("Модель сохранена: /app/checkpoints/retinanet_model.pth")
