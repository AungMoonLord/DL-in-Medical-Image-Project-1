"""
Evaluation Metrics for Thai Character Recognition.
Calculates Accuracy, Macro-F1, Top-k Accuracy, Confusion Matrix,
and Confused Pairs Analysis.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import torch


def calculate_accuracy(targets: Sequence[int], predictions: Sequence[int]) -> float:
    """คำนวณ Accuracy Score ระหว่าง Targets และ Predictions"""
    return float(accuracy_score(targets, predictions))


def calculate_macro_f1(targets: Sequence[int], predictions: Sequence[int]) -> float:
    """คำนวณ Macro-averaged F1 Score"""
    return float(f1_score(targets, predictions, average="macro", zero_division=0))


def calculate_topk_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    top_k: Tuple[int, ...] = (1, 5)
) -> Dict[int, float]:
    """
    คำนวณ Top-k Accuracy สำหรับผลลัพธ์จากโมเดล (Logits/Softmax)
    outputs: Tensor ขนาด [N, C]
    targets: Tensor ขนาด [N]
    คืนค่า: Dict เช่น {1: 0.95, 5: 0.99}
    """
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = targets.size(0)

        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res: Dict[int, float] = {}
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[k] = float(correct_k.mul_(1.0 / batch_size).item())

        return res


def compute_metrics(
    targets: Sequence[int],
    predictions: Sequence[int]
) -> Dict[str, float]:
    """คำนวณทั้ง Accuracy และ Macro-F1 รวมกัน"""
    return {
        "accuracy": calculate_accuracy(targets, predictions),
        "macro_f1": calculate_macro_f1(targets, predictions),
    }


def compute_confusion_matrix(
    targets: Sequence[int],
    predictions: Sequence[int],
    labels: Optional[Sequence[int]] = None
) -> np.ndarray:
    """สร้างตาราง Confusion Matrix"""
    return confusion_matrix(targets, predictions, labels=labels)


def compute_classification_report(
    targets: Sequence[int],
    predictions: Sequence[int],
    labels: Optional[Sequence[int]] = None,
    target_names: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """สร้าง Classification Report ละเอียดระดับคลาส แปลงเป็น DataFrame"""
    report_dict = classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )
    return pd.DataFrame(report_dict).transpose()


def get_confused_pairs(
    confusion: np.ndarray,
    class_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    ดึงคู่ตัวอักษรที่โมเดลทายสับสนมากที่สุด (ไม่รวมแนวทแยงที่ทายถูก)
    เรียงจากคู่ที่ผิดบ่อยที่สุดลงมา
    """
    confusion_no_diag = confusion.copy()
    np.fill_diagonal(confusion_no_diag, 0)

    confused_pairs = []
    num_classes = len(class_names)
    for actual_idx in range(num_classes):
        for pred_idx in range(num_classes):
            count = confusion_no_diag[actual_idx, pred_idx]
            if count > 0:
                confused_pairs.append({
                    "actual": class_names[actual_idx],
                    "predicted": class_names[pred_idx],
                    "count": int(count)
                })

    df = pd.DataFrame(confused_pairs)
    if not df.empty:
        df = df.sort_values("count", ascending=False).reset_index(drop=True)
    return df.head(top_n)
