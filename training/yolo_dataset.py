from __future__ import annotations

from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def _labels_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    if parts[-1] == "images":
        parts[-1] = "labels"
        return Path(*parts)
    return images_dir.parent / "labels"


def collect_image_label_pairs(data_root: Path, split: str) -> list[tuple[Path, Path]]:
    with (data_root / "data.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    key = split if split in cfg else ("valid" if split == "val" else split)
    rel = cfg.get(key, f"{split}/images")
    images_dir = (data_root / rel).resolve()
    if not images_dir.is_absolute():
        images_dir = (data_root / rel).resolve()
    if images_dir.name != "images" and (images_dir / "images").exists():
        images_dir = images_dir / "images"
    labels_dir = _labels_dir(images_dir)

    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        pairs.append((img_path, label_path))
    return pairs


class YoloDetectionDataset(Dataset):
    def __init__(self, data_root: Path, split: str = "train"):
        self.pairs = collect_image_label_pairs(data_root, split)
        with (data_root / "data.yaml").open("r", encoding="utf-8") as f:
            self.class_names = yaml.safe_load(f).get("names", ["fire"])
        self.num_classes = len(self.class_names) + 1

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, label_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        boxes, labels = [], []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                cls, xc, yc, bw, bh = map(float, line.split())
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)

        image = T.functional.to_tensor(image)
        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
        }
        return image, target


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
