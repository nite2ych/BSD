"""
Run BSD candidate training, validation, and ONNX export from a YAML config.

This script is intentionally small and explicit so candidate experiments remain
reproducible on the training server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


def apply_activation_override(cfg: dict[str, Any]) -> None:
    activation = cfg.get("model", {}).get("activation")
    if not activation:
        return

    import torch.nn as nn
    from ultralytics.nn.modules.conv import Conv

    choices = {
        "SiLU": nn.SiLU,
        "ReLU": nn.ReLU,
        "ReLU6": nn.ReLU6,
        "LeakyReLU": lambda: nn.LeakyReLU(0.1, inplace=True),
    }
    if activation not in choices:
        raise ValueError(f"Unsupported activation {activation!r}; choose one of {sorted(choices)}")
    Conv.default_act = choices[activation]()
    print(f"Conv.default_act={Conv.default_act}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def candidate_name(cfg: dict[str, Any]) -> str:
    return cfg["output"]["name"]


def artifact_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["output"]["artifact_dir"])


def best_weight(cfg: dict[str, Any]) -> Path:
    return artifact_dir(cfg) / candidate_name(cfg) / "weights" / "best.pt"


def train(cfg: dict[str, Any], device: str | None) -> Path:
    apply_activation_override(cfg)
    if "architecture" in cfg["model"]:
        model = YOLO(cfg["model"]["architecture"])
        pretrained = cfg["model"].get("pretrained")
        if pretrained:
            model.load(pretrained)
    else:
        model = YOLO(cfg["model"]["pretrained"])
    t = cfg["training"]
    a = cfg["augmentation"]
    train_data = cfg["data"].get("train_yaml", cfg["data"]["yaml"])
    model.train(
        data=train_data,
        imgsz=cfg["input"]["imgsz"],
        epochs=t["epochs"],
        batch=t["batch_size"],
        patience=t["patience"],
        optimizer=t["optimizer"],
        lr0=t["lr0"],
        lrf=t["lrf"],
        momentum=t["momentum"],
        weight_decay=t["weight_decay"],
        warmup_epochs=t["warmup_epochs"],
        seed=t["seed"],
        close_mosaic=t["close_mosaic"],
        hsv_h=a["hsv_h"],
        hsv_s=a["hsv_s"],
        hsv_v=a["hsv_v"],
        translate=a["translate"],
        scale=a["scale"],
        fliplr=a["fliplr"],
        mosaic=a["mosaic"],
        mixup=a["mixup"],
        copy_paste=a["copy_paste"],
        erasing=a["erasing"],
        project=str(artifact_dir(cfg)),
        name=candidate_name(cfg),
        exist_ok=True,
        device=device if device is not None else None,
        workers=8,
        plots=True,
        val=True,
    )
    return best_weight(cfg)


def val_one(cfg: dict[str, Any], weights: Path, data_yaml: str, suffix: str, device: str | None) -> dict[str, Any]:
    apply_activation_override(cfg)
    model = YOLO(str(weights))
    result = model.val(
        data=data_yaml,
        split="val",
        imgsz=cfg["input"]["imgsz"],
        batch=max(1, min(16, cfg["training"]["batch_size"])),
        conf=0.001,
        iou=0.7,
        project=str(artifact_dir(cfg) / f"{candidate_name(cfg)}_eval"),
        name=suffix,
        exist_ok=True,
        device=device if device is not None else None,
        plots=True,
        save_json=False,
        verbose=True,
    )
    return dict(result.results_dict)


def validate(cfg: dict[str, Any], weights: Path, device: str | None) -> Path:
    results = {
        "weights": str(weights),
        "imgsz": cfg["input"]["imgsz"],
        "coco_val": val_one(cfg, weights, cfg["data"]["yaml"], "coco_val", device),
        "bdd_proxy_val": val_one(cfg, weights, cfg["data"]["proxy_val"], "bdd_proxy_val", device),
    }
    out = artifact_dir(cfg) / candidate_name(cfg) / "v5_eval_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return out


def export_onnx(cfg: dict[str, Any], weights: Path) -> Path:
    apply_activation_override(cfg)
    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=cfg["input"]["imgsz"],
        opset=12,
        simplify=False,
        dynamic=False,
        nms=True,
        device="cpu",
    )
    print(f"exported={exported}")
    return Path(exported)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["train", "val", "export", "all"], default="all")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. 0 or cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    weights = args.weights
    if args.mode in {"train", "all"}:
        weights = train(cfg, args.device)
    if weights is None:
        weights = best_weight(cfg)
    if args.mode in {"val", "all"}:
        validate(cfg, weights, args.device)
    if args.mode in {"export", "all"}:
        export_onnx(cfg, weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
