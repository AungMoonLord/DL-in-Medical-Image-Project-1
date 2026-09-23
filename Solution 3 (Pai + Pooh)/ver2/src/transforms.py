"""
Domain-Specific Data Augmentation Pipeline for Thai Character Glyphs (ver2).

Features:
1. Morphological Dilation & Erosion (pen pressure, ink bleed, stroke thickness).
2. Pedestal Augmentation for Classes 173 (ญ) & 176 (ฐ) (simulates footed vs footless typographical forms).
3. Random Letterbox Padding (trains model to be scale-invariant).
4. Strictly bounded geometric transforms (prevents character inversion and loop distortion).
5. Dynamic / Progressive augmentation controller (curriculum-based training).
"""

import math
import random
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.v2 as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MorphologicalTransform:
    """
    Applies morphological dilation or erosion on character images.
    Operates on inverted (0-background, bright foreground) glyphs.
    - Dilation: thickens bright strokes.
    - Erosion: thins bright strokes.

    Ensures the image dimension is at least `min_size` (default 32) on all sides
    while maintaining aspect ratio before morphology to avoid destructive erosion
    on tiny crops (e.g. 17x12).
    """

    def __init__(
        self,
        p_dilate: float = 0.3,
        p_erode: float = 0.3,
        kernel_size: int = 2,
        min_size: int = 32,
    ):
        self.p_dilate = p_dilate
        self.p_erode = p_erode
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.min_size = min_size

    def __call__(self, img: Image.Image) -> Image.Image:
        r = random.random()
        if r >= self.p_dilate + self.p_erode:
            return img

        w, h = img.size
        # Rescale keeping aspect ratio so all sides are at least min_size pixels
        if self.min_size and (w < self.min_size or h < self.min_size):
            scale = max(self.min_size / max(1, w), self.min_size / max(1, h))
            new_w = max(self.min_size, int(round(w * scale)))
            new_h = max(self.min_size, int(round(h * scale)))
            processed_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        else:
            processed_img = img

        np_img = np.array(processed_img)
        if r < self.p_dilate:
            dilated = cv2.dilate(np_img, self.kernel, iterations=1)
            return Image.fromarray(dilated)
        else:
            eroded = cv2.erode(np_img, self.kernel, iterations=1)
            return Image.fromarray(eroded)


class PedestalAugmentation:
    """
    Specialized augmentation for Class 173 (ญ) and Class 176 (ฐ).

    Thai characters ญ and ฐ appear in two typographical variants:
    1. Standard form with lower pedestal ('เชิง' / 'ตีน')
    2. Sub-vowel form without pedestal (when combined with lower vowels ุ, ู to prevent collision).

    This transform simulates the footless representation by zero-filling the bottom
    25-35% of the glyph with probability p.
    """

    def __init__(self, p: float = 0.5, bottom_ratio_range: Tuple[float, float] = (0.22, 0.35)):
        self.p = p
        self.bottom_ratio_range = bottom_ratio_range

    def __call__(self, img: Image.Image, class_number: Optional[int] = None) -> Image.Image:
        # Only apply to classes 173 (ญ) and 176 (ฐ)
        if class_number is not None and class_number not in (173, 176):
            return img

        if random.random() > self.p:
            return img

        np_img = np.array(img)
        h, w = np_img.shape[:2]

        # Determine bottom slice height
        slice_ratio = random.uniform(*self.bottom_ratio_range)
        cut_y = int(h * (1.0 - slice_ratio))

        # Zero-fill bottom pedestal portion (background is 0)
        np_img[cut_y:, :] = 0
        return Image.fromarray(np_img)


class RandomLetterboxPad:
    """
    Applies random zero-padding ratio to train character size invariance.
    """

    def __init__(self, pad_ratio_range: Tuple[float, float] = (0.05, 0.25)):
        self.pad_ratio_range = pad_ratio_range

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        pad_ratio = random.uniform(*self.pad_ratio_range)
        pad_w = int(w * pad_ratio)
        pad_h = int(h * pad_ratio)

        new_w = w + 2 * pad_w
        new_h = h + 2 * pad_h
        target_dim = max(new_w, new_h)

        padded = Image.new("RGB", (target_dim, target_dim), (0, 0, 0))
        paste_x = (target_dim - w) // 2
        paste_y = (target_dim - h) // 2
        padded.paste(img.convert("RGB"), (paste_x, paste_y))
        return padded


class ProgressiveAugmentationController:
    """
    Dynamic augmentation controller that increases transformation strength
    progressively across training epochs (Curriculum Learning).
    """

    def __init__(self, total_epochs: int = 30, max_angle: float = 8.0, max_scale: float = 0.08):
        self.total_epochs = max(1, total_epochs)
        self.max_angle = max_angle
        self.max_scale = max_scale
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = min(epoch, self.total_epochs)

    def get_progress(self) -> float:
        return self.current_epoch / float(self.total_epochs)

    def get_transform(self, is_train: bool = True) -> T.Compose:
        if not is_train:
            return get_val_transform()

        progress = self.get_progress()
        current_angle = self.max_angle * (0.3 + 0.7 * progress)
        current_scale = self.max_scale * (0.3 + 0.7 * progress)
        p_morph = 0.15 + 0.35 * progress

        return T.Compose([
            MorphologicalTransform(p_dilate=p_morph / 2, p_erode=p_morph / 2, kernel_size=2),
            T.ToImage(),
            T.RandomAffine(
                degrees=(-current_angle, current_angle),
                translate=(current_scale, current_scale),
                scale=(1.0 - current_scale, 1.0 + current_scale),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.ColorJitter(
                brightness=0.1 + 0.1 * progress,
                contrast=0.1 + 0.1 * progress,
            ),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_train_transform(
    max_angle: float = 8.0,
    max_translate: float = 0.06,
    scale_range: Tuple[float, float] = (0.92, 1.08),
    p_morphology: float = 0.3,
) -> T.Compose:
    """
    Constructs domain-appropriate training transforms for Thai character glyphs.
    Strictly excludes horizontal/vertical flips to preserve semantic glyph orientation.
    """
    return T.Compose([
        MorphologicalTransform(
            p_dilate=p_morphology / 2,
            p_erode=p_morphology / 2,
            kernel_size=2,
        ),
        T.ToImage(),
        T.RandomAffine(
            degrees=(-max_angle, max_angle),
            translate=(max_translate, max_translate),
            scale=scale_range,
            shear=(-3.0, 3.0),
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        T.ColorJitter(brightness=0.12, contrast=0.12),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform() -> T.Compose:
    """
    Deterministic evaluation transform (ToImage -> ToDtype -> Normalize).
    """
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
