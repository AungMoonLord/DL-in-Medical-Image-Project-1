"""
Loss Functions and Objectives for Imbalanced Thai Character Classification (ver2).

Implements:
1. ClassBalancedFocalLoss (Cui et al., CVPR 2019).
2. CrossEntropy with Label Smoothing.
3. Logit Adjustment Loss.
4. Loss factory `build_loss_function`.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss based on effective number of samples (Cui et al., CVPR 2019)
    with numerical stabilization and weight clipping to prevent gradient explosion on rare classes.
    """

    def __init__(
        self,
        samples_per_cls: List[int],
        num_classes: int = 72,
        beta: float = 0.999,
        gamma: float = 2.0,
        label_smoothing: float = 0.08,
        max_weight: float = 10.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

        samples_arr = np.maximum(np.array(samples_per_cls, dtype=np.float64), 1.0)
        effective_num = 1.0 - np.power(beta, samples_arr)
        effective_num = np.maximum(effective_num, 1e-6)
        weights = (1.0 - beta) / effective_num
        weights = weights / np.sum(weights) * num_classes

        # Clip extreme weights to avoid gradient destabilization
        weights = np.clip(weights, 0.1, max_weight)
        weights = weights / np.mean(weights)

        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Label smoothing one-hot encoding
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / max(1, self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # Focal modulating factor: (1 - p_t)^gamma based on true class probability
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = torch.pow((1.0 - pt).clamp(min=0.0, max=1.0), self.gamma)

        # Batch class weights
        batch_weights = self.class_weights[targets]

        # Cross entropy loss
        ce_loss = -torch.sum(smooth_targets * log_probs, dim=-1)
        loss = batch_weights * focal_weight * ce_loss
        return loss.mean()


def build_loss_function(
    loss_name: str = "cb_focal",
    samples_per_cls: Optional[List[int]] = None,
    num_classes: int = 72,
    beta: float = 0.999,
    gamma: float = 2.0,
    label_smoothing: float = 0.08,
) -> nn.Module:
    """Factory function for building loss functions."""
    name = loss_name.lower().strip()
    if name in ("cb_focal", "class_balanced_focal", "focal"):
        if samples_per_cls is None:
            samples_per_cls = [100] * num_classes
        return ClassBalancedFocalLoss(
            samples_per_cls=samples_per_cls,
            num_classes=num_classes,
            beta=beta,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )
    elif name in ("ce", "cross_entropy"):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unknown loss type: {loss_name}")
