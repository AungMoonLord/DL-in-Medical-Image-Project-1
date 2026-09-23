"""
Training and Benchmark Evaluation Suite for Thai Character Classification (ver2).

Supports:
1. Training across CustomGlyphCNN, AdaptedResNet18, AdaptedMobileNetV3, EfficientNet-B0.
2. Dataset comparison: Original (round2) vs Cleaned (round2-cleaned).
3. Mixed precision training and evaluation with Macro-F1, Top-1, Top-3 metrics.
4. Comprehensive comparative benchmark report generation.
"""

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .dataset import build_dataloaders, tis620_to_char
    from .losses import build_loss_function
    from .models import build_model
except (ImportError, ValueError):
    from dataset import build_dataloaders, tis620_to_char
    from losses import build_loss_function
    from models import build_model


class Trainer:
    """
    End-to-End trainer for Thai Character Classification models.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        model_name: str = "custom_cnn",
        use_amp: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir) / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        self.model = self.model.to(self.device)
        self.best_macro_f1 = 0.0

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for tensors, targets, _ in self.train_loader:
            tensors = tensors.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()

            if self.use_amp and self.scaler:
                with torch.amp.autocast('cuda'):
                    logits = self.model(tensors)
                    loss = self.criterion(logits, targets)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(tensors)
                loss = self.criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

            total_loss += loss.item() * tensors.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        if self.scheduler:
            self.scheduler.step()

        return total_loss / max(1, total), correct / max(1, total)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        top3_correct = 0
        total = 0

        for tensors, targets, _ in self.val_loader:
            tensors = tensors.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(tensors)
            loss = self.criterion(logits, targets)

            total_loss += loss.item() * tensors.size(0)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

            # Top-3 Accuracy
            top3 = torch.topk(logits, k=min(3, logits.size(-1)), dim=-1).indices
            top3_correct += (top3 == targets.unsqueeze(1)).any(dim=-1).sum().item()
            total += targets.size(0)

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        top1_acc = float((all_preds == all_targets).mean())
        top3_acc = float(top3_correct / max(1, total))
        macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))

        return {
            "val_loss": total_loss / max(1, total),
            "top1_acc": top1_acc,
            "top3_acc": top3_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

    def train(self, epochs: int = 30) -> Dict[str, Any]:
        history = []
        print(f"🚀 Starting training for {self.model_name} on {self.device} ({epochs} epochs)...")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch()
            val_metrics = self.evaluate()
            elapsed = time.time() - t0

            val_f1 = val_metrics["macro_f1"]
            val_acc = val_metrics["top1_acc"]

            if val_f1 > self.best_macro_f1:
                self.best_macro_f1 = val_f1
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "macro_f1": val_f1,
                        "top1_acc": val_acc,
                        "model_name": self.model_name,
                    },
                    self.checkpoint_dir / "best_model.pt",
                )
                saved_mark = "⭐ (Saved Best)"
            else:
                saved_mark = ""

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f}, Top1: {val_acc:.4f}, Macro-F1: {val_f1:.4f} {saved_mark}"
            )

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                **val_metrics,
            })

        return {"best_macro_f1": self.best_macro_f1, "history": history}
