"""
Domain-Specific Data Augmentation Pipeline for Thai Character Glyphs.

Features:
1. Morphological Dilation & Erosion (simulates pen pressure, ink bleed, stroke thickness).
2. Strictly bounded geometric transforms (prevents character inversion and loop distortion).
3. Dynamic / Progressive augmentation controller (curriculum-based training).
4. ImageNet & custom normalization presets.
"""

import math
import random
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
import torchvision.transforms.v2 as T

# Standard normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MorphologicalTransform:
    """
    Applies morphological dilation or erosion on character images.
    Simulates handwriting variations like thick marker bleed or faint pencil strokes.
    """

    def __init__(self, p_dilate: float = 0.3, p_erode: float = 0.3, kernel_size: int = 2):
        self.p_dilate = p_dilate
        self.p_erode = p_erode
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def __call__(self, img: Image.Image) -> Image.Image:
        r = random.random()
        if r < self.p_dilate:
            # Dilation (thicker strokes for dark glyphs on white bg -> erosion of white space)
            np_img = np.array(img)
            # Detect if dark-on-light or light-on-dark
            if np.mean(np_img) > 127: # Dark text on light background
                dilated = cv2.erode(np_img, self.kernel, iterations=1)
            else: # Light text on dark background
                dilated = cv2.dilate(np_img, self.kernel, iterations=1)
            return Image.fromarray(dilated)
        elif r < self.p_dilate + self.p_erode:
            np_img = np.array(img)
            if np.mean(np_img) > 127:
                eroded = cv2.dilate(np_img, self.kernel, iterations=1)
            else:
                eroded = cv2.erode(np_img, self.kernel, iterations=1)
            return Image.fromarray(eroded)
        return img


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
    scale_range: Tuple[float, float] = (0.94, 1.06),
    p_morphology: float = 0.3,
) -> T.Compose:
    """
    Constructs standard domain-appropriate training transforms for Thai character glyphs.
    Strictly excludes horizontal/vertical flips to preserve semantic correctness.
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
            shear=(-4.0, 4.0),
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
