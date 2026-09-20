"""
Loss functions สำหรับ long-tail + class similarity

Logit Adjustment (Menon et al., ICLR 2021) คือหัวใจ:
  ตอนเทรน  : logits' = logits + τ·log(π_c)
  ตอน infer: ใช้ logits ดิบ
  ผล: โมเดลถูกบังคับให้ยก logit ของคลาสหายากให้สูงพอจะชนะ prior ที่บวกเข้าไป
  ข้อดีเหนือ oversampling คือไม่ทำให้เห็นภาพเดิมซ้ำจนจำ noise
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitAdjustedLoss(nn.Module):
    """Cross-entropy + logit adjustment + label smoothing."""

    def __init__(self, class_counts, tau: float = 1.0,
                 label_smoothing: float = 0.1):
        super().__init__()
        counts = torch.as_tensor(np.asarray(class_counts), dtype=torch.float32)
        prior = counts.clamp(min=1.0)
        prior = prior / prior.sum()
        self.register_buffer("adjustment", tau * torch.log(prior + 1e-12))
        self.label_smoothing = label_smoothing
        self.tau = tau

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.adjustment, target,
                               label_smoothing=self.label_smoothing)

    # def forward_soft(self, logits: torch.Tensor,
    #                  soft_target: torch.Tensor) -> torch.Tensor:
    #     """สำหรับ mixup (target เป็น distribution ไม่ใช่ index)."""
    #     logp = F.log_softmax(logits + self.adjustment, dim=-1)
    #     return -(soft_target * logp).sum(dim=-1).mean()
    forward_soft = forward # เผื่อมีโค้ดส่วนอื่นเคยเรียกชื่อ forward_soft ไว้ จะได้ไม่เกิด AttributeError


class FocalLoss(nn.Module):
    """ลดน้ำหนักตัวอย่างง่าย ให้โมเดลโฟกัสคู่ที่สับสน (ข/ฃ, ท/ฑ)."""

    def __init__(self, gamma: float = 2.0, class_weights=None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            w = torch.as_tensor(np.asarray(class_weights), dtype=torch.float32)
            self.register_buffer("weight", w)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # คำนวณ CE แบบ unweighted ก่อนเพื่อหาค่า pt ที่แท้จริง
        ce = F.cross_entropy(logits, target, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce.clamp(max=20))
        focal_weight = (1.0 - pt) ** self.gamma

        if self.weight is not None:
            if target.dim() > 1:  # รองรับ soft target จาก mixup
                alpha = (target * self.weight).sum(dim=-1)
            else:
                alpha = self.weight[target]
            return (alpha * focal_weight * ce).mean()

        return (focal_weight * ce).mean()


class SoftTargetCE(nn.Module):
    """Cross-entropy กับ soft target — ใช้คู่ mixup เมื่อไม่ใช้ logit adjustment."""

    def forward(self, logits: torch.Tensor,
                soft_target: torch.Tensor) -> torch.Tensor:
        return -(soft_target * F.log_softmax(logits, dim=-1)).sum(-1).mean()


# ----------------------------------------------------------------------------
# Mixup — tail-aware
# ----------------------------------------------------------------------------
class TailAwareMixup:
    """
    Mixup ที่จับคู่เฉพาะ batch ที่มี tail class อยู่ด้วย
    ทำให้ decision boundary รอบคลาสหายากนุ่มขึ้นโดยไม่รบกวนคลาสใหญ่มากเกินไป
    """

    def __init__(self, num_classes: int, alpha: float = 0.2,
                 tail_labels: list[int] | None = None,
                 tail_aware: bool = True, prob: float = 0.5):
        self.num_classes = num_classes
        self.alpha = alpha
        self.tail_aware = tail_aware
        self.prob = prob
        
        # เก็บเป็น Tensor Boolean Mask แทน Python set เพื่อเช็คบน GPU ได้ในคำสั่งเดียว
        self.tail_mask = torch.zeros(num_classes, dtype=torch.bool)
        if tail_labels:
            self.tail_mask[tail_labels] = True

    def _one_hot(self, target: torch.Tensor) -> torch.Tensor:
        return F.one_hot(target, self.num_classes).float()

    def __call__(self, images: torch.Tensor, target: torch.Tensor):
        """คืน (images, soft_target, applied)."""
        if self.alpha <= 0 or torch.rand(1).item() > self.prob:
            return images, self._one_hot(target), False

        perm = torch.randperm(images.size(0), device=images.device)
        lam = float(np.random.beta(self.alpha, self.alpha))
        lam = max(lam, 1 - lam)  # ให้ภาพหลักเด่นกว่าเสมอ

        y1 = self._one_hot(target)
        y2 = y1[perm]

        if self.tail_aware and self.tail_mask.any():
            # ย้าย mask ไปอยู่ device เดียวกับ images/target อัตโนมัติ (GPU)
            if self.tail_mask.device != target.device:
                self.tail_mask = self.tail_mask.to(target.device)

            # เช็คแบบ Vectorized: เร็วกว่าลูป Python และไม่มี CUDA Sync
            is_tail = self.tail_mask[target]
            
            # mix เฉพาะตำแหน่งที่อย่างน้อยฝั่งหนึ่งเป็น tail
            mask = (is_tail | is_tail[perm]).float().view(-1, 1, 1, 1)
            mixed = images * (lam * mask + (1 - mask)) \
                + images[perm] * ((1 - lam) * mask)
            m2 = mask.view(-1, 1)
            soft = y1 * (lam * m2 + (1 - m2)) + y2 * ((1 - lam) * m2)
        else:
            mixed = lam * images + (1 - lam) * images[perm]
            soft = lam * y1 + (1 - lam) * y2

        return mixed, soft, True


def smooth_soft_target(soft: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0:
        return soft
    n = soft.size(-1)
    return soft * (1 - eps) + eps / n


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
def build_loss(cfg, class_counts) -> nn.Module:
    from .sampler import class_weights as cw_fn

    t = cfg.get_path("loss.type", "ce")
    smooth = cfg.get_path("loss.label_smoothing", 0.1)

    if t == "logit_adjusted":
        return LogitAdjustedLoss(class_counts,
                                 tau=cfg.get_path("loss.tau", 1.0),
                                 label_smoothing=smooth)
    if t == "focal":
        return FocalLoss(gamma=cfg.get_path("loss.focal_gamma", 2.0),
                         class_weights=cw_fn(np.asarray(class_counts), "sqrt"),
                         label_smoothing=smooth)
    return nn.CrossEntropyLoss(label_smoothing=smooth)


def build_mixup(cfg, num_classes: int, class_counts) -> TailAwareMixup | None:
    if not cfg.get_path("loss.mixup.enabled", False):
        return None
    thr = cfg.get_path("augmentation.offline.tail_threshold", 50)
    tail = [i for i, c in enumerate(class_counts) if 0 < c < thr]
    return TailAwareMixup(
        num_classes=num_classes,
        alpha=cfg.get_path("loss.mixup.alpha", 0.2),
        tail_labels=tail,
        tail_aware=cfg.get_path("loss.mixup.tail_aware", True),
    )