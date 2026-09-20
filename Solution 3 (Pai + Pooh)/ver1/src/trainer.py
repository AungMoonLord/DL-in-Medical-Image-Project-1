"""
Training and Optimization Engine for Thai Character Classification.

Implements:
1. Differential AdamW optimizer and Warmup + Cosine Annealing learning rate schedules.
2. Mixed-precision (AMP) training loop for GTX 1650 VRAM optimization.
3. Metric computation (Top-1, Top-5 accuracy, Macro F1).
4. Training history tracking and checkpoint management.
"""

import math
import os
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.optim import AdamW, RMSprop, SGD, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    OneCycleLR,
    ReduceLROnPlateau,
    _LRScheduler,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def build_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    base_lr: float = 1e-3,
    weight_decay: float = 1e-2,
    differential_lr: bool = False,
    backbone_lr: float = 1e-4,
) -> Optimizer:
    """
    Constructs optimizer with optional layer-wise differential learning rates
    for transfer learning backbones.
    """
    opt_name = optimizer_name.lower().strip()

    if differential_lr and hasattr(model, "features") and hasattr(model, "classifier"):
        params = [
            {"params": model.features.parameters(), "lr": backbone_lr},
            {"params": model.classifier.parameters(), "lr": base_lr},
        ]
    else:
        params = model.parameters()

    if opt_name == "adamw":
        return AdamW(params, lr=base_lr, weight_decay=weight_decay)
    elif opt_name == "rmsprop":
        return RMSprop(params, lr=base_lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return SGD(params, lr=base_lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine_warmup",
    total_epochs: int = 25,
    warmup_epochs: int = 3,
    min_lr: float = 1e-6,
) -> _LRScheduler:
    """
    Constructs adaptive learning rate scheduler with warmup phase.
    """
    stype = scheduler_type.lower().strip()

    if stype in ("cosine_warmup", "warmup_cosine"):
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                # Linear warmup
                return float(epoch + 1) / float(max(1, warmup_epochs))
            # Cosine decay
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return max(min_lr / 1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif stype == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)

    elif stype == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=min_lr)

    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")


class ThaiCharacterTrainer:
    """
    Full training pipeline supporting mixed precision, progressive augmentation,
    validation evaluation, and model artifact checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
        device: Optional[torch.device] = None,
        aug_controller: Optional[object] = None,
        save_dir: str = "checkpoints",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.aug_controller = aug_controller
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_top1": [],
            "val_top5": [],
            "val_macro_f1": [],
            "lr": [],
        }

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        if self.aug_controller and hasattr(self.aug_controller, "set_epoch"):
            self.aug_controller.set_epoch(epoch)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total_samples += batch_size

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / total_samples:.4f}",
            })

        epoch_loss = total_loss / max(1, total_samples)
        epoch_acc = correct / max(1, total_samples)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, float, float]:
        self.model.eval()
        total_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total_samples = 0

        all_preds = []
        all_targets = []

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size

            # Top 1
            preds = outputs.argmax(dim=1)
            correct_top1 += (preds == targets).sum().item()

            # Top 5
            maxk = min(5, outputs.size(1))
            _, top5_preds = outputs.topk(maxk, dim=1, largest=True, sorted=True)
            correct_top5 += top5_preds.eq(targets.view(-1, 1).expand_as(top5_preds)).sum().item()

            total_samples += batch_size
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

        val_loss = total_loss / max(1, total_samples)
        top1_acc = correct_top1 / max(1, total_samples)
        top5_acc = correct_top5 / max(1, total_samples)
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        return val_loss, top1_acc, top5_acc, macro_f1

    def fit(self, num_epochs: int = 20) -> Dict[str, List[float]]:
        best_f1 = 0.0
        start_time = time.time()

        print(f"\n[+] Starting Training on Device: {self.device} (AMP: {self.use_amp})")
        print(f"Total Epochs: {num_epochs} | Train Batches: {len(self.train_loader)} | Val Batches: {len(self.val_loader)}\n")

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_top1, val_top5, val_f1 = self.evaluate()

            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_f1)
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_top1"].append(val_top1)
            self.history["val_top5"].append(val_top5)
            self.history["val_macro_f1"].append(val_f1)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - t0
            is_best = val_f1 > best_f1
            if is_best:
                best_f1 = val_f1
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_top1": val_top1,
                }, self.save_dir / "best_model.pt")

            star = " [BEST]" if is_best else ""
            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] ({elapsed:.1f}s) "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} Top-1: {val_top1*100:.2f}% Top-5: {val_top5*100:.2f}% Macro-F1: {val_f1*100:.2f}% | "
                f"LR: {current_lr:.6f}{star}"
            )

        total_time = time.time() - start_time
        print(f"\n[OK] Training Complete in {total_time/60:.2f} min. Best Val Macro-F1: {best_f1*100:.2f}%\n")
        return self.history
