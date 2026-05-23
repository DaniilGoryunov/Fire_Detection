import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common import default_project_root, resolve_data_yaml, train_yolo


def main() -> None:
    p = argparse.ArgumentParser(description="Обучение YOLOv8s")
    p.add_argument("--data-root", type=str, default="datasets/fire")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--project", type=str, default="runs/yolo")
    args = p.parse_args()

    root = default_project_root()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root
    project = Path(args.project)
    if not project.is_absolute():
        project = root / project

    data_yaml = resolve_data_yaml(data_root, root / "training" / ".cache")
    metrics = train_yolo(
        model_name="yolov8s.pt",
        data_yaml=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        run_name="yolov8s",
        project_dir=project,
        extra_kwargs={"augment": True, "patience": 20},
    )
    print("Метрики:", metrics)


if __name__ == "__main__":
    main()
