import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.torchvision_train import run_training


def main() -> None:
    p = argparse.ArgumentParser(description="Обучение Faster R-CNN")
    p.add_argument("--data-root", type=str, default="datasets/fire")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--project", type=str, default="runs/torchvision")
    args = p.parse_args()

    root = ROOT
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root
    project = Path(args.project)
    if not project.is_absolute():
        project = root / project

    metrics = run_training(
        arch="faster_rcnn",
        data_root=data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        run_name="faster_rcnn",
        project_dir=project,
    )
    print("Метрики:", metrics)


if __name__ == "__main__":
    main()
