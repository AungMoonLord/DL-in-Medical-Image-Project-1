"""Training / validation loop ที่ใช้ร่วมกันทั้งสอง stage."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .losses import LogitAdjustedLoss, SoftTargetCE, smooth_soft_target
from .metrics import MetricResult, compute_metrics
from .utils import AverageMeter


class EarlyStopping:
    def __init__(self, patience: int = 12, min_delta: float = 1e-3,
                 mode: str = "max"):
        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.best = -np.inf if mode == "max" else np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = (value > self.best + self.min_delta) if self.mode == "max" \
            else (value < self.best - self.min_delta)
        if improved:
            self.best, self.counter = value, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def _autocast(device: torch.device, enabled: bool, dtype: torch.dtype):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def train_one_epoch(model, loader, criterion, optimizer, device, *,
                    epoch: int = 0, total_epochs: int = 0, scheduler=None,
                    scaler=None, amp: bool = True,
                    amp_dtype: torch.dtype = torch.bfloat16,
                    channels_last: bool = True, mixup=None,
                    max_grad_norm: float = 5.0, grad_accum: int = 1,
                    label_smoothing: float = 0.1, log_interval: int = 20,
                    logger=None, stage: int = 1) -> dict:
    model.train()
    loss_m, acc_m = AverageMeter("loss"), AverageMeter("acc")
    soft_ce = SoftTargetCE()
    use_la_soft = isinstance(criterion, LogitAdjustedLoss)

    pbar = tqdm(loader, desc=f"[S{stage}] train {epoch+1}/{total_epochs}",
                ncols=110, leave=False)
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()

    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        soft = None
        if mixup is not None:
            images, soft, applied = mixup(images, targets)
            if not applied:
                soft = None

        with _autocast(device, amp, amp_dtype):
            head_labels = targets if soft is None else soft
            logits = model(images, head_labels)

            # if soft is None:
            #     loss = criterion(logits, targets)
            # else:
            #     soft = smooth_soft_target(soft, label_smoothing)
            #     loss = (criterion.forward_soft(logits, soft) if use_la_soft
            #             else soft_ce(logits, soft))
            
            # ส่ง head_labels (รองรับทั้ง hard target และ soft target จาก mixup) เข้า criterion ได้โดยตรง
            loss = criterion(logits, head_labels)
            loss = loss / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0:
            if max_grad_norm and max_grad_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g["params"]],
                    max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and getattr(scheduler, "_per_step", False):
                scheduler.step()

        bs = targets.size(0)
        acc = (logits.detach().argmax(1) == targets).float().mean().item()
        loss_m.update(loss.item() * grad_accum, bs)
        acc_m.update(acc, bs)

        if step % log_interval == 0:
            pbar.set_postfix(loss=f"{loss_m.avg:.4f}", acc=f"{acc_m.avg*100:.2f}%",
                             lr=f"{optimizer.param_groups[-1]['lr']:.2e}")

    pbar.close()
    return {"loss": loss_m.avg, "acc": acc_m.avg,
            "lr": optimizer.param_groups[-1]["lr"],
            "time": time.perf_counter() - t0}


@torch.no_grad()
def validate(model, loader, criterion, device, *, num_classes: int = 72,
             class_counts=None, class_names=None, amp: bool = True,
             amp_dtype: torch.dtype = torch.bfloat16,
             channels_last: bool = True, return_raw: bool = False,
             desc: str = "valid") -> tuple[MetricResult, dict]:
    model.eval()
    loss_m = AverageMeter("loss")
    all_probs, all_targets = [], []

    for images, targets in tqdm(loader, desc=desc, ncols=110, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        with _autocast(device, amp, amp_dtype):
            logits = model(images, None)          # ไม่ใส่ margin ตอน eval
            try:
                loss = criterion(logits.float(), targets)
                loss_m.update(loss.item(), targets.size(0))
            except Exception:
                pass

        all_probs.append(F.softmax(logits.float(), dim=1).cpu())
        all_targets.append(targets.cpu())

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    preds = probs.argmax(1)

    result = compute_metrics(targets, preds, probs, num_classes,
                             class_counts, class_names)
    raw = {"loss": loss_m.avg}
    if return_raw:
        raw.update({"probs": probs, "preds": preds, "targets": targets})
    return result, raw


@torch.no_grad()
def validate_tta(model, dataset, tta_transforms, device, *, batch_size: int = 128,
                 num_workers: int = 8, num_classes: int = 72,
                 class_counts=None, class_names=None, amp: bool = True,
                 amp_dtype: torch.dtype = torch.bfloat16,
                 channels_last: bool = True) -> tuple[MetricResult, dict]:
    """เฉลี่ย probability จากหลาย view — มักได้ +0.5 ถึง +1.5% Macro-F1 ฟรี."""
    from copy import copy

    from torch.utils.data import DataLoader

    model.eval()
    acc_probs, targets = None, None

    for vi, tf in enumerate(tta_transforms):
        ds = copy(dataset)
        ds.transform = tf
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        probs_v, tgt_v = [], []
        for images, t in tqdm(loader, desc=f"TTA view {vi+1}/{len(tta_transforms)}",
                              ncols=110, leave=False):
            images = images.to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            with _autocast(device, amp, amp_dtype):
                logits = model(images, None)
            probs_v.append(F.softmax(logits.float(), 1).cpu())
            tgt_v.append(t)
        p = torch.cat(probs_v).numpy()
        acc_probs = p if acc_probs is None else acc_probs + p
        targets = torch.cat(tgt_v).numpy()

    probs = acc_probs / len(tta_transforms)
    preds = probs.argmax(1)
    result = compute_metrics(targets, preds, probs, num_classes,
                             class_counts, class_names)
    return result, {"probs": probs, "preds": preds, "targets": targets}


# ----------------------------------------------------------------------------
# Optimizer / Scheduler
# ----------------------------------------------------------------------------
def build_optimizer(model, cfg, stage: str):
    lr = cfg.get_path(f"train.{stage}.lr", 3e-4)
    wd = cfg.get_path(f"train.{stage}.weight_decay", 0.01)
    mult = cfg.get_path(f"train.{stage}.backbone_lr_mult", 0.1)
    name = cfg.get_path(f"train.{stage}.optimizer", "adamw").lower()

    groups = model.param_groups(lr, mult, wd)
    if not groups:
        raise RuntimeError("ไม่มีพารามิเตอร์ที่ trainable — ตรวจการ freeze")

    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr)
    raise ValueError(f"ไม่รู้จัก optimizer: {name}")


def build_scheduler(optimizer, cfg, stage: str, steps_per_epoch: int):
    """Cosine + linear warmup แบบ per-step (นุ่มกว่า per-epoch)."""
    epochs = cfg.get_path(f"train.{stage}.epochs", 30)
    warmup = cfg.get_path(f"train.{stage}.warmup_epochs", 3)
    min_lr = cfg.get_path(f"train.{stage}.min_lr", 1e-6)
    kind = cfg.get_path(f"train.{stage}.scheduler", "cosine")

    if kind == "none":
        return None

    total = max(epochs * steps_per_epoch, 1)
    warm = max(warmup * steps_per_epoch, 1)
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lr_lambda(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(total - warm, 1)
        cos = 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))
        floor = min_lr / max(base_lrs[-1], 1e-12)
        return floor + (1 - floor) * cos

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    sched._per_step = True
    return sched