from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssd import SSDLiteClassificationHead

from training.yolo_dataset import YoloDetectionDataset, detection_collate, get_transform


def _build_model(arch: str, num_classes: int):
    if arch == "faster_rcnn":
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    if arch == "mobilenet_ssd":
        model = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
        in_channels = model.head.classification_head.module_list[0].in_channels
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels, num_anchors, num_classes
        )
        return model
    raise ValueError(f"Не та архитекутра: {arch}")


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


@torch.inference_mode()
def evaluate_map50(model, loader, device, iou_threshold: float = 0.5) -> dict:
    model.eval()
    matched, total_gt = 0, 0

    def iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter + 1e-9
        return inter / union

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            gt_boxes = tgt["boxes"].cpu()
            total_gt += len(gt_boxes)
            pred_boxes = out["boxes"].cpu()
            pred_scores = out["scores"].cpu()
            keep = pred_scores >= 0.3
            pred_boxes = pred_boxes[keep]
            for gt in gt_boxes:
                best = 0.0
                for pred in pred_boxes:
                    best = max(best, iou(gt.tolist(), pred.tolist()))
                if best >= iou_threshold:
                    matched += 1

    recall_proxy = matched / max(total_gt, 1)
    return {"mAP@0.5_proxy": recall_proxy, "matched_gt": matched, "total_gt": total_gt}


def run_training(
    arch: str,
    data_root: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    run_name: str,
    project_dir: Path,
    device: str | None = None,
) -> dict:
    data_root = data_root.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = YoloDetectionDataset(data_root, "train", get_transform(train=True))
    val_ds = YoloDetectionDataset(data_root, "val", get_transform(train=False))
    num_classes = train_ds.num_classes

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=detection_collate,
    )

    model = _build_model(arch, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{epochs} — loss: {loss:.4f}")

    metrics = evaluate_map50(model, val_loader, device)
    out_dir = project_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pth")
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение torchvision-детекторов")
    p.add_argument("--arch", choices=["faster_rcnn", "mobilenet_ssd"], required=True)
    p.add_argument("--data-root", type=str, default="datasets/fire")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--project", type=str, default="runs/torchvision")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = root / data_root
    project = Path(args.project)
    if not project.is_absolute():
        project = root / project
    name = args.name or args.arch
    metrics = run_training(
        arch=args.arch,
        data_root=data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        run_name=name,
        project_dir=project,
        device=args.device,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
