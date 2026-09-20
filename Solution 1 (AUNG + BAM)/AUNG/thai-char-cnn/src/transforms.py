"""
Augmentation pipeline

หลักสำคัญ 3 ข้อ:
1. Pad-to-square ก่อน resize เสมอ — การยืดภาพทำให้สัดส่วนหัวอักษรเพี้ยน
   ซึ่งเป็นจุดชี้ขาดระหว่าง ก/ถ/ภ และ ิ/ี/ึ/ื
2. ห้าม flip ทุกแกน — ตัวอักษรไทยพลิกแล้วกลายเป็นตัวอื่นหรือไม่มีความหมาย
3. Morphology (dilate/erode) คือ augmentation ที่ตรงกับโดเมนลายมือที่สุด
   เพราะจำลอง "ความหนาของปากกา" ที่ต่างกันจริงในข้อมูล
"""
from __future__ import annotations

import cv2
import numpy as np

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------
# Custom transforms
# ----------------------------------------------------------------------------
class PadToSquare(ImageOnlyTransform):
    """เติมขอบให้เป็นสี่เหลี่ยมจัตุรัสโดยรักษา aspect ratio เดิม."""

    def __init__(self, pad_value: int = 255, margin: float = 0.06,
                 always_apply: bool = True, p: float = 1.0):
        super().__init__(p=p)
        self.pad_value = pad_value
        self.margin = margin

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        side = int(max(h, w) * (1.0 + self.margin))
        top = (side - h) // 2
        bottom = side - h - top
        left = (side - w) // 2
        right = side - w - left
        pad_val = (self.pad_value, self.pad_value, self.pad_value) if img.ndim == 3 else self.pad_value
        return cv2.copyMakeBorder(img, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=pad_val)

    def get_transform_init_args_names(self):
        return ("pad_value", "margin")


class AutoInvert(ImageOnlyTransform):
    """
    บังคับ polarity ให้เป็น 'ตัวอักษรเข้มบนพื้นสว่าง' เสมอ
    ตรวจจากค่ามุมภาพ 4 มุม — ถ้ามุมเข้ม แปลว่าพื้นหลังดำ ต้อง invert
    """

    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = g.shape[:2]
        k = max(2, min(h, w) // 12)
        corners = np.concatenate([
            g[:k, :k].ravel(), g[:k, -k:].ravel(),
            g[-k:, :k].ravel(), g[-k:, -k:].ravel(),
        ])
        return 255 - img if corners.mean() < 110 else img


class RandomMorphology(ImageOnlyTransform):
    """
    สุ่ม dilate/erode เพื่อจำลองความหนาปากกา
    เป็น augmentation ที่ได้ผลสูงมากกับลายมือ เพราะเป็น variation ที่เกิดจริง
    (สมมติภาพเป็นตัวอักษรเข้มบนพื้นสว่าง → erode ทำให้เส้นหนาขึ้น)
    """

    def __init__(self, max_kernel: int = 3, p: float = 0.3):
        super().__init__(p=p)
        self.max_kernel = max_kernel

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        k = np.random.choice([2, 3, self.max_kernel])
        kernel = np.ones((int(k), int(k)), np.uint8)
        return (cv2.erode(img, kernel, iterations=1) if np.random.rand() < 0.5
                else cv2.dilate(img, kernel, iterations=1))

    def get_transform_init_args_names(self):
        return ("max_kernel",)


class ToRGB(ImageOnlyTransform):
    """grayscale → 3 channel (pretrained backbone ต้องการ 3 ch)."""

    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        if img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img


# ----------------------------------------------------------------------------
# Pipeline builders
# ----------------------------------------------------------------------------
def _pre(cfg) -> list:
    """ขั้นตอนที่ต้องทำเหมือนกันทุก split."""
    steps = [ToRGB()]
    if cfg.get_path("data.auto_invert", True):
        steps.append(AutoInvert())
    steps.append(PadToSquare(pad_value=cfg.get_path("data.pad_color", 255)))
    return steps


def _post(cfg) -> list:
    norm = cfg.get_path("data.normalize", {})
    return [
        A.Normalize(mean=tuple(norm.get("mean", IMAGENET_MEAN)),
                    std=tuple(norm.get("std", IMAGENET_STD))),
        ToTensorV2(),
    ]


def build_train_transform(cfg):
    """Online augmentation — ใช้ทุก batch ระหว่างเทรน."""
    size = cfg.get_path("data.image_size", 224)
    o = cfg.get_path("augmentation.online", {}) or {}
    pad = cfg.get_path("data.pad_color", 255)

    aug: list = []
    aug.append(A.Affine(
        scale=(1 - o.get("scale_limit", 0.15), 1 + o.get("scale_limit", 0.15)),
        rotate=(-o.get("rotate_limit", 10), o.get("rotate_limit", 10)),
        shear={"x": (-o.get("shear_limit", 8), o.get("shear_limit", 8)),
               "y": (-o.get("shear_limit", 8) / 2, o.get("shear_limit", 8) / 2)},
        translate_percent=(-o.get("translate_percent", 0.06),
                           o.get("translate_percent", 0.06)),
        cval=pad, mode=cv2.BORDER_CONSTANT, p=0.85,
    ))

    el = o.get("elastic", {}) or {}
    gd = o.get("grid_distortion", {}) or {}
    warps = []
    if el.get("enabled", True):
        warps.append(A.ElasticTransform(
            alpha=el.get("alpha", 30), sigma=el.get("sigma", 6),
            border_mode=cv2.BORDER_CONSTANT, value=pad, p=1.0))
    # if gd.get("enabled", True):
    #     warps.append(A.GridDistortion(
    #         num_steps=5, 
    #         distort_limit=(-float(gd.get("distort_limit", 0.2)), float(gd.get("distort_limit", 0.2))),
    #         normalized=True,
    #         border_mode=cv2.BORDER_CONSTANT, 
    #         value=pad, 
    #         p=1.0
    #     ))
    if gd.get("enabled", True):
        warps.append(A.GridDistortion(
            num_steps=5, distort_limit=gd.get("distort_limit", 0.2),
            border_mode=cv2.BORDER_CONSTANT, value=pad, p=1.0))
    if warps:
        aug.append(A.OneOf(warps, p=max(el.get("p", 0.4), gd.get("p", 0.3))))

    mo = o.get("morphology", {}) or {}
    if mo.get("enabled", True):
        aug.append(RandomMorphology(max_kernel=mo.get("max_kernel", 3),
                                    p=mo.get("p", 0.3)))

    bc = o.get("brightness_contrast", 0.2)
    aug.append(A.RandomBrightnessContrast(brightness_limit=bc,
                                          contrast_limit=bc, p=0.5))
    aug.append(A.OneOf([
        A.GaussNoise(var_limit=(5, 30)),
        A.GaussianBlur(blur_limit=(3, 5)),
        A.MotionBlur(blur_limit=5),
    ], p=0.25))

    cd = o.get("coarse_dropout", {}) or {}
    if cd.get("enabled", True):
        max_h = max(4, int(cd.get("max_size", 12)))
        max_w = max(4, int(cd.get("max_size", 12)))
        max_holes = max(1, int(cd.get("max_holes", 4)))
        aug.append(A.CoarseDropout(
            num_holes_range=(1, max_holes),
            hole_height_range=(2, max_h),
            hole_width_range=(2, max_w),
            fill_value=pad,    # หรือ fill=pad ตามเวอร์ชันของ albumentations
            p=float(cd.get("p", 0.2)),
        ))

    return A.Compose(_pre(cfg) + aug
                     + [A.Resize(size, size, interpolation=cv2.INTER_AREA)]
                     + _post(cfg))


def build_tail_transform(cfg):
    """
    Offline augmentation สำหรับคลาส tail — แรงกว่า online อย่างมีนัยสำคัญ
    เพราะภาพต้นฉบับมีน้อยมาก (บางคลาสมีใบเดียว) ต้องสร้าง variation ให้กว้างที่สุด
    เท่าที่ยังคงเป็นตัวอักษรเดิมอยู่
    คืน pipeline ที่ยังเป็น uint8 image (ไม่ normalize) เพื่อเซฟเป็นไฟล์
    """
    size = cfg.get_path("data.image_size", 224)
    pad = cfg.get_path("data.pad_color", 255)

    return A.Compose(_pre(cfg) + [
        A.Affine(scale=(0.80, 1.20), rotate=(-13, 13),
                 shear={"x": (-12, 12), "y": (-7, 7)},
                 translate_percent=(-0.09, 0.09),
                 cval=pad, mode=cv2.BORDER_CONSTANT, p=0.95),
        A.OneOf([
            A.ElasticTransform(alpha=45, sigma=7,
                               border_mode=cv2.BORDER_CONSTANT, value=pad),
            A.GridDistortion(num_steps=6, distort_limit=0.30,
                             border_mode=cv2.BORDER_CONSTANT, value=pad),
            A.OpticalDistortion(distort_limit=0.30,
                                border_mode=cv2.BORDER_CONSTANT, value=pad),
        ], p=0.85),
        A.Perspective(scale=(0.02, 0.07), pad_val=pad, p=0.4),
        RandomMorphology(max_kernel=4, p=0.6),
        A.RandomBrightnessContrast(0.30, 0.30, p=0.6),
        A.OneOf([A.GaussNoise(var_limit=(10, 45)),
                 A.GaussianBlur(blur_limit=(3, 7))], p=0.35),
        A.Resize(size, size, interpolation=cv2.INTER_AREA),
    ])


def build_val_transform(cfg):
    """Validation — deterministic ล้วน ห้ามมีความสุ่ม."""
    size = cfg.get_path("data.image_size", 224)
    return A.Compose(_pre(cfg)
                     + [A.Resize(size, size, interpolation=cv2.INTER_AREA)]
                     + _post(cfg))


def build_tta_transforms(cfg, n_views: int = 5) -> list:
    """
    TTA — view แรกเป็น clean เสมอ ที่เหลือเป็น perturbation เบา ๆ
    ใช้ scale/shift/rotate เล็กน้อยเท่านั้น เพราะอักษรไทยไวต่อการบิดเบือน
    """
    size = cfg.get_path("data.image_size", 224)
    pad = cfg.get_path("data.pad_color", 255)
    views = [build_val_transform(cfg)]

    params = [
        dict(scale=(1.06, 1.06), rotate=(0, 0), tx=0.0),
        dict(scale=(0.94, 0.94), rotate=(0, 0), tx=0.0),
        dict(scale=(1.0, 1.0), rotate=(4, 4), tx=0.02),
        dict(scale=(1.0, 1.0), rotate=(-4, -4), tx=-0.02),
        dict(scale=(1.03, 1.03), rotate=(2, 2), tx=0.015),
        dict(scale=(0.97, 0.97), rotate=(-2, -2), tx=-0.015),
    ]
    for p in params[: max(0, n_views - 1)]:
        views.append(A.Compose(_pre(cfg) + [
            A.Affine(scale=p["scale"], rotate=p["rotate"],
                     translate_percent=(p["tx"], p["tx"]),
                     cval=pad, mode=cv2.BORDER_CONSTANT, p=1.0),
            A.Resize(size, size, interpolation=cv2.INTER_AREA),
        ] + _post(cfg)))
    return views


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """สำหรับแสดงภาพใน Grad-CAM / notebook."""
    import torch
    m = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    s = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    return (tensor * s + m).clamp(0, 1)