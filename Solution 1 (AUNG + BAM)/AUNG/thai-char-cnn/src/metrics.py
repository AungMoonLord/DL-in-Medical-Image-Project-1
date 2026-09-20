"""
Metrics — Macro-F1 คือตัวตัดสิน ไม่ใช่ Accuracy

เหตุผล: 12 คลาสใหญ่กิน 73% ของข้อมูล โมเดลที่ทิ้ง 23 คลาส tail ไปเลย
ยังได้ accuracy สูงได้ แต่ Macro-F1 จะร่วงทันที เพราะทุกคลาสมีน้ำหนักเท่ากัน
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support)


@dataclass
class MetricResult:
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    balanced_accuracy: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    top5_accuracy: float = 0.0
    head_recall: float = 0.0      # คลาส >= 200 ภาพ
    mid_recall: float = 0.0       # 50-199
    tail_recall: float = 0.0      # < 50
    per_class: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"Acc {self.accuracy*100:.2f}% | MacroF1 {self.macro_f1*100:.2f}% | "
                f"BalAcc {self.balanced_accuracy*100:.2f}% | Top5 {self.top5_accuracy*100:.2f}% | "
                f"head/mid/tail {self.head_recall*100:.1f}/{self.mid_recall*100:.1f}/"
                f"{self.tail_recall*100:.1f}%")

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "per_class"}
        return d


def topk_accuracy(probs: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
    if probs.ndim != 2 or probs.shape[1] < k:
        return 0.0
    topk = np.argpartition(-probs, k - 1, axis=1)[:, :k]
    return float(np.mean([t in row for t, row in zip(targets, topk)]))


def compute_metrics(targets, preds, probs=None, num_classes: int = 72,
                    class_counts=None, class_names=None,
                    tail_threshold: int = 50,
                    head_threshold: int = 200) -> MetricResult:
    targets = np.asarray(targets)
    preds = np.asarray(preds)

    p, r, f1, support = precision_recall_fscore_support(
        targets, preds, labels=np.arange(num_classes),
        average=None, zero_division=0)

    res = MetricResult(
        accuracy=float(accuracy_score(targets, preds)),
        macro_f1=float(f1_score(targets, preds, labels=np.arange(num_classes),
                                average="macro", zero_division=0)),
        weighted_f1=float(f1_score(targets, preds, labels=np.arange(num_classes),
                                   average="weighted", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(targets, preds)),
        macro_precision=float(p.mean()),
        macro_recall=float(r.mean()),
        top5_accuracy=topk_accuracy(probs, targets, 5) if probs is not None else 0.0,
    )

    names = class_names or [str(i) for i in range(num_classes)]
    res.per_class = {
        int(i): {"name": names[i], "precision": float(p[i]), "recall": float(r[i]),
                 "f1": float(f1[i]), "support": int(support[i]),
                 "train_count": int(class_counts[i]) if class_counts is not None else -1}
        for i in range(num_classes)
    }

    if class_counts is not None:
        c = np.asarray(class_counts)
        for key, mask in (("tail_recall", c < tail_threshold),
                          ("mid_recall", (c >= tail_threshold) & (c < head_threshold)),
                          ("head_recall", c >= head_threshold)):
            valid = mask & (support > 0)
            setattr(res, key, float(r[valid].mean()) if valid.any() else 0.0)
    return res


def get_confusion_matrix(targets, preds, num_classes: int = 72,
                         normalize: bool = True) -> np.ndarray:
    cm = confusion_matrix(targets, preds, labels=np.arange(num_classes))
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
    return cm


def top_confused_pairs(targets, preds, class_names, num_classes: int = 72,
                       top_k: int = 25) -> list[dict]:
    """คู่ที่สับสนมากที่สุด — เอาไปใส่สไลด์ได้ตรง ๆ."""
    cm = confusion_matrix(targets, preds, labels=np.arange(num_classes))
    support = cm.sum(axis=1)
    pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true": class_names[i], "pred": class_names[j],
                    "true_idx": i, "pred_idx": j, "count": int(cm[i, j]),
                    "rate": float(cm[i, j] / max(support[i], 1)),
                })
    pairs.sort(key=lambda d: (-d["count"], -d["rate"]))
    return pairs[:top_k]


def group_analysis(targets, preds, groups_as_labels, class_names) -> list[dict]:
    """
    วิเคราะห์เฉพาะกลุ่มที่หน้าตาคล้ายกัน:
    'ในบรรดาภาพจริงของกลุ่มนี้ โมเดลทายถูกกี่ % และหลุดไปนอกกลุ่มกี่ %'
    """
    targets, preds = np.asarray(targets), np.asarray(preds)
    out = []
    for labels in groups_as_labels:
        s = set(labels)
        mask = np.isin(targets, labels)
        n = int(mask.sum())
        if n == 0:
            continue
        t, p = targets[mask], preds[mask]
        out.append({
            "group": " ".join(class_names[i] for i in labels),
            "support": n,
            "accuracy": float((t == p).mean()),
            "within_group_error": float(np.mean([(a != b) and (b in s)
                                                 for a, b in zip(t, p)])),
            "outside_group_error": float(np.mean([b not in s for b in p])),
        })
    return sorted(out, key=lambda d: d["accuracy"])


def per_class_table(result: MetricResult, sort_by: str = "recall",
                    ascending: bool = True):
    import pandas as pd
    df = pd.DataFrame([
        {"label": k, "class": v["name"], "train_count": v["train_count"],
         "val_support": v["support"], "precision": v["precision"],
         "recall": v["recall"], "f1": v["f1"]}
        for k, v in result.per_class.items()
    ])
    return df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)


def format_report(result: MetricResult, worst_k: int = 15) -> str:
    lines = [
        "=" * 74,
        "EVALUATION REPORT",
        "=" * 74,
        f"  Accuracy           : {result.accuracy*100:>6.2f}%",
        f"  Macro-F1  (หลัก)   : {result.macro_f1*100:>6.2f}%   ← ตัวตัดสิน",
        f"  Weighted-F1        : {result.weighted_f1*100:>6.2f}%",
        f"  Balanced Accuracy  : {result.balanced_accuracy*100:>6.2f}%",
        f"  Top-5 Accuracy     : {result.top5_accuracy*100:>6.2f}%",
        "-" * 74,
        f"  Head recall (>=200): {result.head_recall*100:>6.2f}%",
        f"  Mid  recall (50-199): {result.mid_recall*100:>5.2f}%",
        f"  Tail recall (<50)  : {result.tail_recall*100:>6.2f}%   ← จุดชี้วัดว่าแก้ imbalance สำเร็จไหม",
        "-" * 74,
        f"  {worst_k} คลาสที่แย่ที่สุด:",
        f"  {'class':<8}{'train':>8}{'val':>6}{'recall':>9}{'f1':>8}",
    ]
    worst = sorted(result.per_class.values(),
                   key=lambda v: (v["recall"], v["f1"]))[:worst_k]
    for v in worst:
        lines.append(f"  {v['name']:<8}{v['train_count']:>8,}{v['support']:>6}"
                     f"{v['recall']*100:>8.1f}%{v['f1']*100:>7.1f}%")
    lines.append("=" * 74)
    return "\n".join(lines)