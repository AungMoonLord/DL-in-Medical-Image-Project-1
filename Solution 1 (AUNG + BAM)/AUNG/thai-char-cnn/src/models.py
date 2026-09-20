"""
Backbone (timm, ImageNet pretrained) + classifier head

ArcFace แก้ได้สองปัญหาพร้อมกัน:
 1. Similarity — angular margin m บังคับให้ ข กับ ฃ ต้องห่างกันเชิงมุม
    อย่างน้อย m ทำให้ decision boundary คมขึ้นมาก
 2. Imbalance — การ normalize weight ทำให้ ‖W_c‖ ของคลาสใหญ่ไม่ท่วมคลาสเล็ก
    (ปัญหาคลาสสิกของ linear head บน long-tail)

Adaptive margin: คลาสที่มีภาพเดียวได้ margin น้อยลง เพราะบังคับ margin สูง
กับข้อมูลที่ไม่มี intra-class variation จะทำให้ไม่ converge
"""
from __future__ import annotations

import math

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Heads
# ----------------------------------------------------------------------------
class LinearHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int,
                 embedding_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.embedding_dim = embedding_dim

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.neck(x)

    def forward(self, x: torch.Tensor, labels=None) -> torch.Tensor:
        return self.fc(self.neck(x))


class ArcFaceHead(nn.Module):
    """
    ArcFace: L = -log( e^{s·cos(θ_y+m)} / (e^{s·cos(θ_y+m)} + Σ e^{s·cosθ_c}) )
    ถ้า labels=None (ตอน eval/inference) จะคืน s·cosθ ตรง ๆ ไม่ใส่ margin
    """

    def __init__(self, in_features: int, num_classes: int,
                 embedding_dim: int = 512, dropout: float = 0.3,
                 scale: float = 30.0, margin: float = 0.30,
                 easy_margin: bool = False, adaptive_margin: bool = False,
                 class_counts=None, margin_min: float = 0.10,
                 margin_max: float = 0.35):
        super().__init__()
        self.neck = nn.Sequential(
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout),
        )
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)

        self.scale = scale
        self.easy_margin = easy_margin
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        if adaptive_margin and class_counts is not None:
            c = np.asarray(class_counts, dtype=np.float64).clip(min=1.0)
            # margin ∝ n^{1/4} (Ada-ArcFace style) — คลาสใหญ่ได้ margin มากกว่า
            r = c ** 0.25
            r = (r - r.min()) / max(r.max() - r.min(), 1e-8)
            m = margin_min + r * (margin_max - margin_min)
        else:
            m = np.full(num_classes, margin, dtype=np.float64)

        m_t = torch.as_tensor(m, dtype=torch.float32)
        self.register_buffer("margin", m_t)
        self.register_buffer("cos_m", torch.cos(m_t))
        self.register_buffer("sin_m", torch.sin(m_t))
        self.register_buffer("th", torch.cos(math.pi - m_t))
        self.register_buffer("mm", torch.sin(math.pi - m_t) * m_t)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.neck(x)

    def cosine(self, x: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(self.neck(x), dim=1)
        w = F.normalize(self.weight, dim=1)
        return F.linear(emb, w).clamp(-1 + 1e-7, 1 - 1e-7)

    def forward(self, x: torch.Tensor, labels=None) -> torch.Tensor:
        cos = self.cosine(x)
        if labels is None or not self.training:
            return cos * self.scale

        # labels อาจเป็น soft target จาก mixup → ใช้ argmax เป็นตัวแทน
        if labels.dim() > 1:
            labels = labels.argmax(dim=1)

        cos_m = self.cos_m[labels].unsqueeze(1)
        sin_m = self.sin_m[labels].unsqueeze(1)
        sin = torch.sqrt((1.0 - cos.pow(2)).clamp(min=1e-9))
        phi = cos * cos_m - sin * sin_m          # cos(θ + m)

        if self.easy_margin:
            phi = torch.where(cos > 0, phi, cos)
        else:
            th = self.th[labels].unsqueeze(1)
            mm = self.mm[labels].unsqueeze(1)
            phi = torch.where(cos > th, phi, cos - mm)

        onehot = F.one_hot(labels, self.num_classes).to(cos.dtype)
        return (onehot * phi + (1 - onehot) * cos) * self.scale


# ----------------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------------
class ThaiCharModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int = 72,
                 pretrained: bool = True, head_type: str = "linear",
                 embedding_dim: int = 512, dropout: float = 0.3,
                 drop_path: float = 0.1, arcface_cfg: dict | None = None,
                 class_counts=None, in_chans: int = 3):
        super().__init__()
        self.backbone_name = backbone_name
        self.head_type = head_type

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0,
            global_pool="avg", drop_path_rate=drop_path, in_chans=in_chans,
        )
        feat_dim = self.backbone.num_features

        if head_type == "arcface":
            a = arcface_cfg or {}
            self.head = ArcFaceHead(
                feat_dim, num_classes, embedding_dim, dropout,
                scale=a.get("scale", 30.0), margin=a.get("margin", 0.30),
                easy_margin=a.get("easy_margin", False),
                adaptive_margin=a.get("adaptive_margin", False),
                class_counts=class_counts,
                margin_min=a.get("margin_min", 0.10),
                margin_max=a.get("margin_max", 0.35),
            )
        else:
            self.head = LinearHead(feat_dim, num_classes, embedding_dim, dropout)

    def forward(self, x: torch.Tensor, labels=None) -> torch.Tensor:
        return self.head(self.backbone(x), labels)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.head.embed(self.backbone(x))

    # ---------------- freeze control ----------------
    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        # BN ต้องอยู่ใน eval mode ด้วย ไม่งั้น running stats จะเพี้ยนตอน stage 2
        if freeze:
            self.backbone.eval()
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()  # คง frozen BN ไว้
        return self

    def param_groups(self, head_lr: float, backbone_mult: float = 0.1,
                     weight_decay: float = 0.01) -> list[dict]:
        """
        แยก lr ของ backbone กับ head + ไม่ใส่ weight decay กับ bias/norm
        (มาตรฐานการ fine-tune ที่ช่วยให้ converge นิ่งขึ้น)
        """
        def split(module):
            decay, no_decay = [], []
            for n, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                # ไม่ใส่ weight decay ให้ bias, 1D norm และ ArcFace class weights
                if p.ndim <= 1 or n.endswith(".bias") or n == "weight":
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        bd, bn = split(self.backbone)
        hd, hn = split(self.head)
        groups = []
        if bd: groups.append({"params": bd, "lr": head_lr * backbone_mult,
                              "weight_decay": weight_decay, "name": "backbone_decay"})
        if bn: groups.append({"params": bn, "lr": head_lr * backbone_mult,
                              "weight_decay": 0.0, "name": "backbone_nodecay"})
        if hd: groups.append({"params": hd, "lr": head_lr,
                              "weight_decay": weight_decay, "name": "head_decay"})
        if hn: groups.append({"params": hn, "lr": head_lr,
                              "weight_decay": 0.0, "name": "head_nodecay"})
        return groups

    def get_cam_target_layer(self):
        """ชั้นสุดท้ายของ backbone สำหรับ Grad-CAM."""
        for attr in ("conv_head", "stages", "blocks", "layer4", "features"):
            if hasattr(self.backbone, attr):
                m = getattr(self.backbone, attr)
                return m[-1] if isinstance(m, (nn.Sequential, nn.ModuleList)) else m
        return list(self.backbone.modules())[-2]


def build_model(cfg, class_counts=None) -> ThaiCharModel:
    return ThaiCharModel(
        backbone_name=cfg.get_path("model.backbone", "tf_efficientnet_b0.ns_jft_in1k"),
        num_classes=cfg.get_path("project.num_classes", 72),
        pretrained=cfg.get_path("model.pretrained", True),
        head_type=cfg.get_path("model.head", "linear"),
        embedding_dim=cfg.get_path("model.embedding_dim", 512),
        dropout=cfg.get_path("model.dropout", 0.3),
        drop_path=cfg.get_path("model.drop_path", 0.1),
        arcface_cfg=cfg.get_path("model.arcface", {}) or {},
        class_counts=class_counts,
        in_chans=cfg.get_path("data.in_channels", 3),
    )