from __future__ import annotations

import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def resolve_data_yaml(data_root: Path, temp_dir: Path) -> Path:
    source_yaml = data_root / "data.yaml"
    if not source_yaml.exists():
        raise FileNotFoundError(
            f"Не найден {source_yaml}"
        )

    with source_yaml.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    data_cfg["path"] = str(data_root.resolve())
    for split in ("train", "val", "valid", "test"):
        if split not in data_cfg:
            continue
        rel = Path(str(data_cfg[split]))
        if rel.is_absolute():
            continue
        images_dir = data_root / rel
        if images_dir.name != "images" and (images_dir / "images").exists():
            images_dir = images_dir / "images"
        data_cfg[split] = str(images_dir.resolve())

    if "val" not in data_cfg and "valid" in data_cfg:
        data_cfg["val"] = data_cfg["valid"]

    temp_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = temp_dir / "data_abs.yaml"
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f, allow_unicode=True, sort_keys=False)
    return out_yaml


def extract_yolo_metrics(val_result) -> dict:
    precision = float(val_result.box.mp)
    recall = float(val_result.box.mr)
    map50 = float(val_result.box.map50)
    map5095 = float(val_result.box.map)
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)
    return {
        "precision": precision,
        "recall": recall,
        "mAP@0.5": map50,
        "mAP@0.5:0.95": map5095,
        "f1": f1,
    }


def train_yolo(
    model_name: str,
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    run_name: str,
    project_dir: Path,
    extra_kwargs: dict | None = None,
) -> dict:
    model = YOLO(model_name)
    kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project_dir),
        name=run_name,
        verbose=True,
        exist_ok=True,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    model.train(**kwargs)
    val_result = model.val(data=str(data_yaml), split="val")
    metrics = extract_yolo_metrics(val_result)
    metrics_path = project_dir / run_name / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]
