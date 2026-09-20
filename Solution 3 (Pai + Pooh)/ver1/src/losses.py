"""
Loss Functions for Imbalanced Thai Character Classification.

Implements:
1. ClassBalancedFocalLoss (Combines effective sample weighting with focal hard-example mining).
2. LabelSmoothingCrossEntropy (Mitigates overconfidence between visually similar Thai glyph pairs).
3. Factory function `build_loss_fn`.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-Entropy loss with Label Smoothing.
    Prevents overconfidence between confusing character pairs (e.g. ด vs ต, ข vs ช).
    """

    def __init__(self, epsilon: float = 0.08, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_preds = F.log_softmax(logits, dim=-1)

        # Smooth targets: (1 - eps) * one_hot + eps / num_classes
        with torch.no_grad():
            smooth_targets = torch.full_like(log_preds, self.epsilon / num_classes)
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.epsilon + (self.epsilon / num_classes))

        loss = (-smooth_targets * log_preds).sum(dim=-1)

        if self.weight is not None:
            sample_weights = self.weight[targets]
            loss = loss * sample_weights
            return loss.sum() / sample_weights.sum()

        return loss.mean()


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss (Cui et al., CVPR 2019).
    Combines effective sample numbers with focal modulation (1 - p_t)^gamma.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather target probabilities
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal modulating factor: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - target_probs + 1e-8, self.gamma)

        # Standard cross-entropy per sample with optional smoothing
        if self.label_smoothing > 0:
            smooth_targets = torch.full_like(log_probs, self.label_smoothing / num_classes)
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + (self.label_smoothing / num_classes))
            ce_loss = (-smooth_targets * log_probs).sum(dim=-1)
        else:
            ce_loss = F.nll_loss(log_probs, targets, reduction="none")

        loss = focal_weight * ce_loss

        # Apply class balancing weights if provided
        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]
            loss = loss * sample_weights
            return loss.sum() / (sample_weights.sum() + 1e-8)

        return loss.mean()


def build_loss_fn(
    loss_type: str = "class_balanced_focal",
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.08,
) -> nn.Module:
    """
    Factory function for loss modules.
    
    Supported types:
    - 'class_balanced_focal' / 'cbfocal'
    - 'label_smoothing' / 'ls_ce'
    - 'cross_entropy' / 'ce'
    """
    name = loss_type.lower().strip()
    if name in ("class_balanced_focal", "cbfocal", "focal"):
        return ClassBalancedFocalLoss(
            class_weights=class_weights,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )
    elif name in ("label_smoothing", "ls_ce"):
        return LabelSmoothingCrossEntropy(
            epsilon=label_smoothing,
            weight=class_weights,
        )
    elif name in ("cross_entropy", "ce"):
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
