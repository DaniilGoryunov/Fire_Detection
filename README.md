# Детекция пожара для БПЛА — репозиторий диплома

Сравнение архитектур детекции огня и демонстрация на Streamlit

## Структура репозитория

```
.
├── README.md                 # этот файл
├── requirements.txt          # зависимости Python
├── .gitignore
│
├── app/                      # демонстрация (Streamlit)
│   ├── streamlit_app.py      # веб-интерфейс: карта маршрута БПЛА + видео/фото
│   └── inference.py          # загрузка моделей и инференс
│
├── training/                 # обучение четырёх архитектур
│   ├── common.py             # общие функции (YOLO, data.yaml)
│   ├── yolo_dataset.py       # конвертация YOLO → torchvision
│   ├── torchvision_train.py  # цикл обучения Faster R-CNN / SSD
│   ├── train_yolov8n.py      # YOLOv8n
│   ├── train_yolov8s.py      # YOLOv8s
│   ├── train_faster_rcnn.py  # Faster R-CNN
│   └── train_mobilenet_ssd.py# MobileNet-SSD (SSD Lite)
│
├── datasets/                 # данные (не в git)
│   ├── README.md
│   └── fire/                 # YOLO-датасет: data.yaml, train/, valid/, test/
│
├── weights/                  # экспорт лучших весов YOLO для демо
│   ├── yolov8n/best.pt
│   └── yolov8s/best.pt
│
├── runs/                     # артефакты обучения (создаётся скриптами)
│   ├── yolo/                 # Ultralytics: yolov8n, yolov8s
│   └── torchvision/          # model.pth для Faster R-CNN и SSD
│
├── config/
│   └── models_comparison.yaml  # справочник метрик из дипломной таблицы
│
└── temp_videos/              # временные файлы демо (создаётся приложением)
```

### Назначение каталогов

| Каталог | Назначение |
|---------|------------|
| `app/` | Запуск демонстрации для защиты диплома: загрузка видео/изображения, выбор модели, карта точек облёта |
| `training/` | Воспроизводимое обучение всех моделей из сравнительной таблицы |
| `datasets/` | Локальное хранение размеченного датасета в формате YOLO |
| `weights/` | Готовые веса для Streamlit без полного пути к `runs/` |
| `runs/` | Логи, метрики и чекпоинты после `train_*.py` |
| `config/` | Справочные параметры и ориентиры по mAP/FPS |

## Быстрый старт

```bash
cd git
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Датасет: datasets/fire/data.yaml + train/valid/test
python training/train_yolov8n.py --data-root datasets/fire

# Демо (из корня репозитория)
streamlit run app/streamlit_app.py
```

## Обучение моделей

```bash
python training/train_yolov8n.py
python training/train_yolov8s.py
python training/train_faster_rcnn.py
python training/train_mobilenet_ssd.py
```

После YOLO скопируйте `runs/yolo/yolov8n/weights/best.pt` → `weights/yolov8n/best.pt` для демо.
