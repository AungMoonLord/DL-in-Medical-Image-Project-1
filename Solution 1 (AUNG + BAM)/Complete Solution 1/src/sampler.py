"""
Sampling strategies สำหรับ long-tail

ทำไมต้อง sqrt ไม่ใช่ 1/n:
  dataset นี้มี ratio 5,025:1 ถ้าใช้ w = 1/n ภาพเดียวของ 'ฃ' จะถูกสุ่มมา
  บ่อยเท่ากับภาพทั้ง 5,025 ใบของ 'า' รวมกัน → โมเดลจะจำ noise ของภาพนั้น
  แทนที่จะเรียนรูปร่างตัวอักษร
  sqrt (w ∝ n^-0.5) ให้ tail class ได้โอกาสมากขึ้นอย่างมีนัยสำคัญ
  แต่ยังไม่สุดโต่งจนเกิด overfit
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def class_weights(counts: np.ndarray, mode: str = "sqrt",
                  beta: float = 0.999) -> np.ndarray:
    """คำนวณน้ำหนักต่อคลาส (normalize ให้ mean = 1)."""
    counts = np.asarray(counts, dtype=np.float64)
    safe = np.clip(counts, 1.0, None)

    if mode == "instance":
        w = np.ones_like(safe)
    elif mode == "sqrt":
        w = safe ** -0.5
    elif mode == "balanced":
        w = safe ** -1.0
    elif mode == "effective":
        # Class-Balanced Loss (Cui et al., CVPR 2019)
        eff = (1.0 - np.power(beta, safe)) / (1.0 - beta)
        w = 1.0 / eff
    else:
        raise ValueError(f"ไม่รู้จัก sampler mode: {mode}")

    w[counts == 0] = 0.0
    m = w[w > 0].mean() if (w > 0).any() else 1.0
    return w / m


def build_sampler(labels: np.ndarray, num_classes: int, mode: str = "sqrt",
                  beta: float = 0.999, num_samples: int | None = None,
                  generator: torch.Generator | None = None):
    """
    คืน WeightedRandomSampler หรือ None (ถ้า mode='instance' ให้ใช้ shuffle ปกติ)
    """
    if mode == "instance":
        return None

    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes)
    cw = class_weights(counts, mode=mode, beta=beta)
    sample_w = cw[labels]

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=int(num_samples or len(labels)),
        replacement=True,
        generator=generator,
    )


def describe_sampler(labels: np.ndarray, num_classes: int, mode: str,
                     class_names: list[str] | None = None, top_k: int = 5) -> str:
    """สรุปว่า sampler เปลี่ยนสัดส่วนการสุ่มไปอย่างไร — ใช้ใส่สไลด์ได้เลย."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes)
    cw = class_weights(counts, mode=mode)
    expected = cw * counts
    expected = expected / expected.sum() * len(labels)

    order = np.argsort(counts)
    lines = [f"Sampler = '{mode}'",
             f"{'class':<10}{'ก่อน':>10}{'หลัง(คาด)':>14}{'×':>9}"]
    for i in list(order[:top_k]) + list(order[-top_k:]):
        name = class_names[i] if class_names else str(i)
        before, after = counts[i], expected[i]
        ratio = after / max(before, 1)
        lines.append(f"{name:<10}{before:>10,}{after:>14,.0f}{ratio:>8.1f}x")
    return "\n".join(lines)