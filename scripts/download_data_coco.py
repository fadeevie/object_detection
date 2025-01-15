from google_drive_downloader import GoogleDriveDownloader as gdd
import os

# ID папки Google Диска
DRIVE_FOLDER_ID = "1nGUcc7bbz8Y1MHl1Wp-znNlFQByjwzVs"

# Локальная папка для данных
LOCAL_DATA_FOLDER = "/app/data"

# Функция для загрузки файла
def download_data_from_drive():
    print("Подключение к Google Диску...")
    gdd.download_file_from_google_drive(
        file_id=DRIVE_FOLDER_ID,
        dest_path=f"{LOCAL_DATA_FOLDER}/data.zip",
        unzip=True
    )
    print("Данные успешно загружены и распакованы!")

if __name__ == "__main__":
    os.makedirs(LOCAL_DATA_FOLDER, exist_ok=True)
    download_data_from_drive()