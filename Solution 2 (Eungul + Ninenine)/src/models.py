"""
PyTorch Model Architecture for Thai Character Recognition.
Uses EfficientNet-B0 with Transfer Learning from ImageNet.
Layer names and structure strictly match the original notebook and Colab Fine-tuning:
  - features: EfficientNet-B0 Feature Extractor
  - avgpool: AdaptiveAvgPool2d(output_size=1)
  - classifier: Sequential(
      (0): Dropout(p=0.35, inplace=False)
      (1): Linear(in_features=1280, out_features=NUM_CLASSES, bias=True)
    )
Ensures 100% shape and key compatibility with best_thai_character_finetuned.pth.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

from src.config import DROPOUT_RATE, resolve_checkpoint_path


def create_model(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = DROPOUT_RATE
) -> nn.Module:
    """
    สร้างโมเดล EfficientNet-B0 พร้อม Classifier Head ใหม่สำหรับตัวอักษรไทย
    โครงสร้างและชื่อ Layer ตรงตามในโน้ตบุ๊ก:
      - features: EfficientNet-B0 Feature Extractor
      - avgpool: AdaptiveAvgPool2d(output_size=1)
      - classifier: Sequential(
          (0): Dropout(p=0.35, inplace=False)
          (1): Linear(in_features=1280, out_features=num_classes, bias=True)
        )
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    input_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(input_features, num_classes)
    )
    return model


def load_model_from_checkpoint(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    dropout: float = DROPOUT_RATE
) -> Tuple[nn.Module, Dict[str, int], Dict[int, str], Optional[float]]:
    """
    โหลดโมเดลจากไฟล์ Checkpoint (.pth)
    รองรับทั้ง best_thai_character_finetuned.pth และ best_thai_character_model_v2.pth
    ป้องกันปัญหา Key Mismatch (เช่น module. จาก DataParallel) และ Shape Mismatch
    คืนค่า: (model, class_to_idx, idx_to_class, best_score)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_path = resolve_checkpoint_path(checkpoint_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ Checkpoint ที่: {resolved_path.resolve()}")

    checkpoint: Dict[str, Any] = torch.load(resolved_path, map_location=device)

    # 1. ดึง State Dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        raw_state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and any(k.startswith("features.") for k in checkpoint.keys()):
        raw_state_dict = checkpoint
    else:
        raise KeyError("ไม่พบคีย์ 'model_state_dict' หรือโครงสร้างน้ำหนักโมเดลในไฟล์ Checkpoint")

    # คลีนคีย์ prefix 'module.' ในกรณีที่บันทึกมาจาก DataParallel บน Colab
    cleaned_state_dict = {}
    for k, v in raw_state_dict.items():
        clean_k = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[clean_k] = v

    # 2. ตรวจสอบจำนวนคลาสจากโครงสร้างน้ำหนัก Classifier หรือ class_to_idx
    if "class_to_idx" in checkpoint:
        class_to_idx = checkpoint["class_to_idx"]
        num_classes = len(class_to_idx)
    elif "classifier.1.weight" in cleaned_state_dict:
        num_classes = cleaned_state_dict["classifier.1.weight"].shape[0]
        class_to_idx = {str(i): i for i in range(num_classes)}
    else:
        num_classes = 72  # ค่าเริ่มต้นสำหรับ Thai Character Dataset
        class_to_idx = {str(i): i for i in range(num_classes)}

    # ดึง idx_to_class
    if "idx_to_class" in checkpoint:
        raw_idx = checkpoint["idx_to_class"]
        idx_to_class = {int(k): str(v) for k, v in raw_idx.items()}
    else:
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    best_score: Optional[float] = checkpoint.get("best_score", None)

    # 3. สร้างและโหลดน้ำหนักเข้าโมเดล
    model = create_model(num_classes=num_classes, pretrained=False, dropout=dropout)
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, class_to_idx, idx_to_class, best_score
