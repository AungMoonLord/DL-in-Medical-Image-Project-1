"""
Comprehensive Evaluation and Error Analysis Suite for Thai Character Classification.

Implements:
1. Multi-class metric extraction (Top-1/Top-5 Accuracy, Balanced Accuracy, Per-Class F1).
2. Confusion Matrix plotting with Thai character labels.
3. Top Confused Character Pairs analysis (e.g. ด vs ต, ข vs ช, ผ vs พ).
4. Training curve visualization.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
try:
    from .dataset import tis620_to_char
except (ImportError, ValueError):
    from dataset import tis620_to_char


@torch.no_grad()
def evaluate_model_full(
    model: nn.Module,
    test_loader: DataLoader,
    class_to_idx: Dict[int, int],
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    Runs full inference on test_loader and extracts complete performance statistics.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    idx_to_char = {idx: tis620_to_char(num) for num, idx in class_to_idx.items()}
    num_classes = len(class_to_idx)

    all_preds = []
    all_targets = []
    all_probs = []

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Top-1 and Top-5 accuracy
    top1_acc = np.mean(all_preds == all_targets)

    top5_correct = 0
    for i, target in enumerate(all_targets):
        top5 = np.argsort(all_probs[i])[-5:]
        if target in top5:
            top5_correct += 1
    top5_acc = top5_correct / len(all_targets)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

    # Per-class metrics
    report = classification_report(
        all_targets,
        all_preds,
        labels=list(range(num_classes)),
        target_names=[idx_to_char.get(i, str(i)) for i in range(num_classes)],
        output_dict=True,
        zero_division=0,
    )

    # Top confused pairs (excluding diagonal)
    confused_pairs = []
    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            if true_idx != pred_idx and cm[true_idx, pred_idx] > 0:
                confused_pairs.append({
                    "true_char": idx_to_char.get(true_idx, str(true_idx)),
                    "pred_char": idx_to_char.get(pred_idx, str(pred_idx)),
                    "true_idx": true_idx,
                    "pred_idx": pred_idx,
                    "count": cm[true_idx, pred_idx],
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    return {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": cm,
        "classification_report": report,
        "top_confusions": confused_pairs[:20],
        "idx_to_char": idx_to_char,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    idx_to_char: Dict[int, str],
    top_n: int = 30,
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None,
):
    """
    Plots a normalized confusion matrix heatmap for the most active classes.
    """
    # Normalize by row sum
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype("float") / row_sums

    # Slice top N active classes
    labels = [idx_to_char.get(i, str(i)) for i in range(min(top_n, cm.shape[0]))]
    sub_cm = cm_norm[:top_n, :top_n]

    plt.figure(figsize=figsize)
    sns.heatmap(
        sub_cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"Normalized Confusion Matrix (Top {top_n} Classes)", fontsize=14)
    plt.xlabel("Predicted Character", fontsize=12)
    plt.ylabel("True Character", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (16, 5),
    save_path: Optional[str] = None,
):
    """
    Plots training loss, validation accuracy, macro F1, and learning rate curves.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="tab:red", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="tab:blue", lw=2)
    axes[0].set_title("Cross-Entropy Loss", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Accuracy & Macro F1
    axes[1].plot(epochs, [x * 100 for x in history["train_acc"]], label="Train Acc", color="tab:orange", lw=2)
    axes[1].plot(epochs, [x * 100 for x in history["val_top1"]], label="Val Top-1 Acc", color="tab:green", lw=2)
    axes[1].plot(epochs, [x * 100 for x in history["val_macro_f1"]], label="Val Macro F1", color="tab:purple", lw=2, linestyle="--")
    axes[1].set_title("Accuracy & Macro F1 (%)", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3. Learning Rate
    axes[2].plot(epochs, history["lr"], label="Learning Rate", color="tab:cyan", lw=2)
    axes[2].set_title("Learning Rate Schedule (Warmup + Cosine)", fontsize=13)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
