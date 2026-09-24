"""
Thai Character Classification - Solution 3 (ver2) Core Pipeline Package.
"""

from .dataset import build_split_dataframes, letterbox_pad, tis620_to_char
from .focal_cleaner import FocalElementCleaner, clean_glyph_artifacts
from .inference import THAI_CHAR_METADATA, ThaiCharacterInferenceEngine, UniversalModelLoader
from .losses import ClassBalancedFocalLoss, build_loss_function
from .models import AdaptedEfficientNetB0, AdaptedMobileNetV3, AdaptedResNet18, CustomGlyphCNN, build_model
from .prototype_filter import ClassPrototypeEngine
from .transforms import (
    MorphologicalTransform,
    PedestalAugmentation,
    ProgressiveAugmentationController,
    get_train_transform,
    get_val_transform,
)

__all__ = [
    "build_split_dataframes",
    "letterbox_pad",
    "tis620_to_char",
    "FocalElementCleaner",
    "clean_glyph_artifacts",
    "ClassPrototypeEngine",
    "PedestalAugmentation",
    "MorphologicalTransform",
    "ProgressiveAugmentationController",
    "get_train_transform",
    "get_val_transform",
    "CustomGlyphCNN",
    "AdaptedResNet18",
    "AdaptedMobileNetV3",
    "AdaptedEfficientNetB0",
    "build_model",
    "ClassBalancedFocalLoss",
    "build_loss_function",
    "THAI_CHAR_METADATA",
    "ThaiCharacterInferenceEngine",
    "UniversalModelLoader",
]
