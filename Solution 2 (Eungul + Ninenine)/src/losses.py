"""
Loss Functions and Class Weighting for Thai Character Recognition.
Handles class imbalance using inverse square root weighting with clipping
and label smoothing.
"""

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import LABEL_SMOOTHING


def compute_class_weights(
    train_df: pd.DataFrame,
    num_classes: int,
    min_clip: float = 1.0,
    max_clip: float = 8.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    คำนวณ Class Weight จากจำนวนภาพของแต่ละคลาสใน Train Set
    สูตร: weight = clip(sqrt(max_count / count), min_clip, max_clip)
    ปรับ normalize ให้ค่าเฉลี่ยเป็น 1.0
    """
    train_class_counts = (
        train_df["label"]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .sort_index()
    )
    counts = train_class_counts.to_numpy(dtype=np.float32)
    maximum_count = counts.max()

    class_weights = np.sqrt(maximum_count / np.maximum(counts, 1.0))
    class_weights = np.clip(class_weights, min_clip, max_clip)
    class_weights = class_weights / class_weights.mean()

    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    if device is not None:
        weights_tensor = weights_tensor.to(device)

    return weights_tensor


def get_loss_function(
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = LABEL_SMOOTHING
) -> nn.CrossEntropyLoss:
    """
    สร้าง Loss Function (CrossEntropyLoss) พร้อม Class Weights และ Label Smoothing
    """
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )
