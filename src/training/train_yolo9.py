import os
import gdown
from ultralytics import YOLO
import yaml

def download_dataset(url, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Загрузка zip-архива
    zip_path = os.path.join(output_dir, "dataset.zip")
    print(f"Загрузка датасета из {url}...")
    gdown.download(url, zip_path, quiet=False)

    # Распаковка
    print(f"Распаковка датасета в {output_dir}...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    print("Датасет успешно загружен и распакован.")

def train_yolo(config_path):

    # Чтение конфигурационного файла
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Настройка путей
    datasets_dir = config['datasets_dir']
    model_path = config['pretrained_model']
    yaml_path = config['dataset_yaml']
    dataset_url = config.get('dataset_url', None)

    # Если указан URL, скачиваем датасет
    if dataset_url:
        print(f"Подключен URL: {dataset_url}")
        download_dataset(dataset_url, datasets_dir)

    # Загрузка предобученной модели
    model = YOLO(model_path)

    # Обучение модели
    print("Запуск обучения модели...")
    model.train(data=yaml_path, epochs=config['epochs'], imgsz=config['imgsz'])

    # Сохранение модели
    output_dir = os.path.join(config['output_dir'], 'weights')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(output_dir)
    print(f"Модель сохранена в {output_dir}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Путь к конфигурационному файлу")
    args = parser.parse_args()

    train_yolo(args.config)
