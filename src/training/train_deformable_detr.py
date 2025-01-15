from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules

# Регистрация модулей MMDetection
register_all_modules()

# Загрузка конфигурации
cfg = Config.fromfile('configs/deformable_detr.py')

# Инициализация и запуск обучения
runner = Runner.from_cfg(cfg)
runner.train()
