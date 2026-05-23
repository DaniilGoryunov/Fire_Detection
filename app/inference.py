from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

MODEL_REGISTRY = {
    "YOLOv8n": {
        "type": "yolo",
        "default_weights": "weights/yolov8n/best.pt",
        "fallback": "yolov8n.pt",
    },
    "YOLOv8s": {
        "type": "yolo",
        "default_weights": "weights/yolov8s/best.pt",
        "fallback": "yolov8s.pt",
    },
    "Faster R-CNN": {
        "type": "torchvision",
        "arch": "faster_rcnn",
        "default_weights": "runs/torchvision/faster_rcnn/model.pth",
    },
    "MobileNet-SSD": {
        "type": "torchvision",
        "arch": "mobilenet_ssd",
        "default_weights": "runs/torchvision/mobilenet_ssd/model.pth",
    },
}


def resolve_weights(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def load_yolo(weights: Path, fallback: str) -> YOLO:
    if weights.exists():
        return YOLO(str(weights))
    return YOLO(fallback)


def load_torchvision(arch: str, weights: Path):
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        ssdlite320_mobilenet_v3_large,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.ssd import SSDLiteClassificationHead

    num_classes = 2  # fire + background
    if arch == "faster_rcnn":
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        model = ssdlite320_mobilenet_v3_large(weights=None)
        in_channels = model.head.classification_head.module_list[0].in_channels
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels, num_anchors, num_classes
        )
    if not weights.exists():
        raise FileNotFoundError(
            f"Веса не найдены: {weights}"
        )
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def predict_yolo(model: YOLO, source, conf: float = 0.25, **kwargs):
    return model.predict(source=source, conf=conf, save=True, **kwargs)


def predict_image_torchvision(model, image: Image.Image, conf: float = 0.3):
    import torchvision.transforms as T

    tensor = T.functional.to_tensor(image.convert("RGB"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    with torch.inference_mode():
        out = model([tensor.to(device)])[0]
    boxes = out["boxes"].cpu()
    scores = out["scores"].cpu()
    keep = scores >= conf
    return boxes[keep], scores[keep]
