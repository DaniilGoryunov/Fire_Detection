# Датасет

Положите сюда YOLO-датасет с разметкой пожара:

```
datasets/fire/
├── data.yaml
├── train/images, train/labels
├── valid/images, valid/labels
└── test/images, test/labels
```

Шаблон конфигурации: `fire/data.yaml.example`.

Данные в репозиторий не включаются (см. `.gitignore`).

Если датасет уже есть локально (например, `старый код/fire detection.v1i.yolov8`):

```bash
ln -s "../../старый код/fire detection.v1i.yolov8" datasets/fire
```
