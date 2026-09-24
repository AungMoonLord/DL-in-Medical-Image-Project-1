"""
Training and Validation Pipelines for Thai Character Recognition.
Includes 2-stage transfer learning (Frozen Backbone -> Fine-Tuning 35 Epochs),
mixed precision training, automatic checkpoint saving, and metric visualization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # เพื่อความเสถียรบน Terminal และ Server
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config import (
    BATCH_SIZE,
    CURVE_IMAGE_PATH,
    DEFAULT_CHECKPOINT_PATH,
    DEVICE,
    NUM_WORKERS,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    SCHEDULER_MODE,
    SCHEDULER_PATIENCE,
    STAGE1_EPOCHS,
    STAGE1_LR,
    STAGE2_BACKBONE_LR,
    STAGE2_CLASSIFIER_LR,
    STAGE2_EPOCHS,
    WEIGHT_DECAY,
    resolve_checkpoint_path,
)
from src.dataset import ThaiCharacterDataset
from src.metrics import calculate_accuracy, calculate_macro_f1


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: torch.device = DEVICE
) -> Dict[str, Any]:
    """
    รัน 1 Epoch สำหรับการฝึกสอน (เมื่อระบุ optimizer) หรือการประเมินผล (เมื่อไม่ระบุ optimizer)
    รองรับ Mixed Precision อัตโนมัติเมื่อใช้งานบน GPU (CUDA)
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    running_loss = 0.0
    all_targets: List[int] = []
    all_predictions: List[int] = []

    desc = "Training" if is_training else "Validation"
    pbar = tqdm(data_loader, desc=desc, leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)

            if is_training:
                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        all_targets.extend(targets.cpu().numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    total_samples = len(data_loader.dataset)
    epoch_loss = running_loss / max(total_samples, 1)
    epoch_accuracy = calculate_accuracy(all_targets, all_predictions)
    epoch_macro_f1 = calculate_macro_f1(all_targets, all_predictions)

    return {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
        "macro_f1": epoch_macro_f1,
        "targets": all_targets,
        "predictions": all_predictions,
    }


def train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    criterion: nn.Module,
    epochs: int,
    stage_name: str,
    checkpoint_path: Path,
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    best_score: float = -1.0,
    history: Optional[List[Dict[str, Any]]] = None,
    device: torch.device = DEVICE
) -> Tuple[nn.Module, float, List[Dict[str, Any]]]:
    """
    ฝึกสอนโมเดลใน Stage ที่ระบุ พร้อมบันทึก Checkpoint เมื่อได้คะแนน Macro-F1 ดีที่สุด
    บันทึกคีย์ model_state_dict, class_to_idx, idx_to_class และ best_score ครบถ้วน
    """
    if history is None:
        history = []

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(1, epochs + 1):
        train_result = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device
        )
        val_result = run_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device
        )

        scheduler.step(val_result["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "val_loss": val_result["loss"],
            "val_accuracy": val_result["accuracy"],
            "val_macro_f1": val_result["macro_f1"],
            "lr": current_lr,
        }
        history.append(row)

        print(
            f"[{stage_name}] Epoch {epoch}/{epochs} | "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_accuracy']*100:.2f}% val_f1={row['val_macro_f1']:.4f} "
            f"lr={current_lr:.2e}"
        )

        if val_result["macro_f1"] > best_score:
            best_score = val_result["macro_f1"]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "best_score": best_score,
            }, checkpoint_path)
            print(f"  -> [บันทึกโมเดลที่ดีที่สุด] (val_macro_f1={best_score:.4f}) ที่ {checkpoint_path}")

    return model, best_score, history


def setup_stage1(
    model: nn.Module,
    lr: float = STAGE1_LR,
    weight_decay: float = WEIGHT_DECAY
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """
    ตั้งค่า Stage 1: Freeze Backbone ทั้งหมด เทรนเฉพาะ Classifier Head
    """
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=SCHEDULER_MODE,
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR
    )
    return optimizer, scheduler


def setup_stage2(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device = DEVICE,
    backbone_lr: float = STAGE2_BACKBONE_LR,
    classifier_lr: float = STAGE2_CLASSIFIER_LR,
    weight_decay: float = WEIGHT_DECAY
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """
    ตั้งค่า Stage 2: โหลดโมเดลที่ดีที่สุดจาก Checkpoint และทำการ Fine-Tuning
    ปลดล็อค Backbone 3 บล็อกท้ายสุด (`features[-3:]`) และ Classifier Head
    """
    resolved_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(resolved_path, map_location=device)

    # ดึง state dict และคลีน module. prefix
    raw_state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    cleaned_state_dict = {}
    for k, v in raw_state_dict.items():
        clean_k = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[clean_k] = v

    model.load_state_dict(cleaned_state_dict)

    for param in model.features.parameters():
        param.requires_grad = False
    for block in model.features[-3:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": model.features[-3:].parameters(), "lr": backbone_lr},
        {"params": model.classifier.parameters(), "lr": classifier_lr},
    ], weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=SCHEDULER_MODE,
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR
    )
    return optimizer, scheduler


def plot_and_save_training_curves(
    history: List[Dict[str, Any]],
    output_path: Path = CURVE_IMAGE_PATH
) -> None:
    """
    วาดกราฟ Accuracy และ Loss ตลอดกระบวนการเทรน และบันทึกเป็นไฟล์ภาพ PNG
    """
    if not history:
        print("[คำเตือน] ไม่มีข้อมูลประวัติการเทรนสำหรับวาดกราฟ")
        return

    history_df = pd.DataFrame(history)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # กราฟ Accuracy
    axes[0].plot(history_df["train_accuracy"], marker="o", label="Train accuracy")
    axes[0].plot(history_df["val_accuracy"], marker="o", label="Validation accuracy")
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].set_xlabel("Epoch Step")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # กราฟ Loss
    axes[1].plot(history_df["train_loss"], marker="o", label="Train loss")
    axes[1].plot(history_df["val_loss"], marker="o", label="Validation loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].set_xlabel("Epoch Step")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[*] บันทึกกราฟประสิทธิภาพการเทรนเรียบร้อยที่: {output_path.resolve()}")


def evaluate_by_source(
    model: nn.Module,
    val_df: pd.DataFrame,
    val_transform: Any,
    criterion: nn.Module,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    device: torch.device = DEVICE
) -> Dict[str, Dict[str, float]]:
    """
    ประเมินผลแยกตามแหล่งที่มา (handwritten เทียบกับ printed)
    เพื่อดูว่าโมเดลมีจุดอ่อนในสไตล์ใดเป็นพิเศษหรือไม่
    """
    model.eval()
    results: Dict[str, Dict[str, float]] = {}

    for source_name in val_df["source"].unique():
        subset_df = val_df[val_df["source"] == source_name].reset_index(drop=True)
        if len(subset_df) == 0:
            continue

        subset_dataset = ThaiCharacterDataset(subset_df, transform=val_transform)
        subset_loader = DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        subset_result = run_epoch(
            model=model,
            data_loader=subset_loader,
            criterion=criterion,
            device=device
        )
        results[source_name] = {
            "count": len(subset_df),
            "accuracy": subset_result["accuracy"],
            "macro_f1": subset_result["macro_f1"],
        }
        print(
            f"== สไตล์ {source_name} (จำนวน {len(subset_df)} ภาพ) == "
            f"Accuracy: {subset_result['accuracy']*100:.2f}% | "
            f"Macro F1: {subset_result['macro_f1']:.4f}"
        )

    return results


def plot_and_save_confusion_matrix(
    confusion: Any,
    class_names: List[str],
    output_path: Path
) -> None:
    """
    วาด Heatmap ของ Confusion Matrix และบันทึกลงไฟล์
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(24, 20))
    sns.heatmap(
        confusion,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] บันทึก Confusion Matrix เรียบร้อยที่: {output_path.resolve()}")
