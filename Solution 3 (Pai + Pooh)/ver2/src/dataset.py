"""
Dataset and Deterministic Splitting Pipeline for Thai Character Classification (ver2).

Implements:
1. TIS-620 character mapping.
2. Zero-leakage deterministic group-aware 80/20 train-test splitting.
3. Aspect-ratio preserving letterbox square padding with configurable margin.
4. PyTorch Dataset with optional focal cleaning and pedestal augmentation.
5. Balanced DataLoader builders.
"""

from collections import Counter, defaultdict
import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from .focal_cleaner import FocalElementCleaner
    from .transforms import PedestalAugmentation, get_train_transform, get_val_transform
except (ImportError, ValueError):
    from focal_cleaner import FocalElementCleaner
    from transforms import PedestalAugmentation, get_train_transform, get_val_transform

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def tis620_to_char(number: Union[int, str]) -> str:
    """Convert a decimal TIS-620 byte value to its corresponding Thai character."""
    val = int(number)
    if not 0 <= val <= 255:
        raise ValueError(f"TIS-620 byte value must be between 0 and 255, got {val}")
    try:
        return bytes([val]).decode("tis-620")
    except UnicodeDecodeError:
        return f"Unknown({val})"


def get_group_key(filename: str) -> str:
    """
    Extracts a grouping key from character filenames to prevent data leakage.
    e.g.: 'bc_001sg_3_118.jpg' and 'bc_001tg_3_118.jpg' are unified under 'bc_001_p3'.
    """
    parts = filename.split("_")
    if len(parts) >= 3:
        doc_prefix = parts[0]
        sub_doc = parts[1]
        if sub_doc.endswith(("sg", "tg")):
            sub_doc = sub_doc[:-2]
        doc_base = f"{doc_prefix}_{sub_doc}"
        page_id = parts[2]
        return f"{doc_base}_p{page_id}"
    elif len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def invert_glyph_if_needed(img: Image.Image) -> Image.Image:
    """
    Inverts document scan images (dark text on light paper) so that:
    - Background is strictly 0 (black / zero-intensity).
    - Character glyph strokes are bright/foreground (>0).

    This ensures standard zero-padding (fill=0) and convolutional networks
    naturally treat empty margins as 0-valued signal.
    """
    np_img = np.array(img.convert("L"))
    corners = np.array([np_img[0, 0], np_img[0, -1], np_img[-1, 0], np_img[-1, -1]])
    # All raw scans in the dataset have light background (>100) and dark ink strokes (<100).
    # If corners, mean, or upper percentile indicates a light background document image, invert it.
    if np.median(corners) > 100 or np.percentile(np_img, 70) > 100 or np.mean(np_img) > 100:
        inverted_np = 255 - np_img
        return Image.fromarray(inverted_np).convert("RGB" if img.mode == "RGB" else "L")
    return img


def letterbox_pad(
    img: Image.Image,
    target_size: Tuple[int, int] = (32, 32),
    pad_ratio: float = 0.10,
    fill_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Pads an inverted (0-background) image into a square letterbox while preserving aspect ratio.
    Defaults to zero-padding (0, 0, 0) matching the black background.
    """
    w, h = img.size
    target_w, target_h = target_size

    # Compute scale factor to fit target size with safety padding
    effective_w = int(target_w * (1.0 - 2 * pad_ratio))
    effective_h = int(target_h * (1.0 - 2 * pad_ratio))

    scale = min(effective_w / max(1, w), effective_h / max(1, h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    padded = Image.new("RGB", target_size, fill_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded.paste(resized.convert("RGB"), (paste_x, paste_y))
    return padded


def build_split_dataframes(
    dataset_dir: Union[str, Path],
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int]]:
    """
    Builds zero-leakage deterministic train and test DataFrames.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    class_dirs = sorted(
        [d for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit() and not d.name.startswith(".")],
        key=lambda d: int(d.name),
    )

    class_to_idx = {int(d.name): idx for idx, d in enumerate(class_dirs)}
    
    train_records: List[dict] = []
    test_records: List[dict] = []

    for class_dir in class_dirs:
        class_number = int(class_dir.name)
        class_idx = class_to_idx[class_number]
        character = tis620_to_char(class_number)

        files = sorted(
            [
                f.name
                for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS and not f.name.startswith(".")
            ]
        )
        n_samples = len(files)

        if n_samples == 0:
            continue

        groups = defaultdict(list)
        for f in files:
            groups[get_group_key(f)].append(f)

        for g_key, g_files in groups.items():
            hash_val = int(hashlib.sha256(g_key.encode("utf-8")).hexdigest(), 16)
            assigned_split = "train" if (hash_val % 100) < int(train_ratio * 100) else "test"

            target_records = train_records if assigned_split == "train" else test_records
            for f_name in g_files:
                target_records.append({
                    "filepath": str(class_dir / f_name),
                    "filename": f_name,
                    "class_number": class_number,
                    "class_idx": class_idx,
                    "character": character,
                    "group_key": g_key,
                    "split": assigned_split,
                })

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)
    return train_df, test_df, class_to_idx


class ThaiCharacterDataset(Dataset):
    """
    PyTorch Dataset for Thai Character Glyphs with input inversion,
    optional focal artifact cleaning, and typography pedestal augmentation.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_size: Tuple[int, int] = (32, 32),
        transform: Optional[Callable] = None,
        use_focal_cleaner: bool = False,
        use_pedestal_aug: bool = False,
        pad_ratio: float = 0.10,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.target_size = target_size
        self.transform = transform
        self.pad_ratio = pad_ratio
        self.focal_cleaner = FocalElementCleaner() if use_focal_cleaner else None
        self.pedestal_aug = PedestalAugmentation(p=0.5) if use_pedestal_aug else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        class_idx = int(row["class_idx"])
        class_number = int(row["class_number"])

        img = Image.open(img_path).convert("RGB")

        # Invert first thing: background = 0 (black), glyph strokes = 255 (bright)
        img = invert_glyph_if_needed(img)

        # Optional Focal Element Artifact Cleaning (operates on 0-background)
        if self.focal_cleaner is not None:
            img = self.focal_cleaner.clean(img)

        # Optional Pedestal Augmentation for ญ (173) and ฐ (176)
        if self.pedestal_aug is not None and class_number in (173, 176):
            img = self.pedestal_aug(img, class_number=class_number)

        # Letterbox pad to target square size with zero padding (0, 0, 0)
        letterboxed = letterbox_pad(img, target_size=self.target_size, pad_ratio=self.pad_ratio, fill_color=(0, 0, 0))

        if self.transform:
            tensor = self.transform(letterboxed)
        else:
            tensor = get_val_transform()(letterboxed)

        return tensor, class_idx, class_number


def build_dataloaders(
    dataset_dir: Union[str, Path],
    batch_size: int = 64,
    target_size: Tuple[int, int] = (32, 32),
    num_workers: int = 0,
    use_balanced_sampler: bool = False,
    use_focal_cleaner: bool = False,
    use_pedestal_aug: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict[int, int], pd.DataFrame, pd.DataFrame]:
    """
    Constructs train and test DataLoaders with zero data leakage.
    Note: When using class-balanced loss (CB Focal), set use_balanced_sampler=False
    to prevent quadratic over-weighting of rare classes.
    """
    train_df, test_df, class_to_idx = build_split_dataframes(dataset_dir, train_ratio=0.8)

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_dataset = ThaiCharacterDataset(
        dataframe=train_df,
        target_size=target_size,
        transform=train_transform,
        use_focal_cleaner=use_focal_cleaner,
        use_pedestal_aug=use_pedestal_aug,
    )

    test_dataset = ThaiCharacterDataset(
        dataframe=test_df,
        target_size=target_size,
        transform=val_transform,
        use_focal_cleaner=use_focal_cleaner,
        use_pedestal_aug=False,
    )

    if use_balanced_sampler:
        class_counts = Counter(train_df["class_idx"])
        total_samples = len(train_df)
        class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
        sample_weights = [class_weights[idx] for idx in train_df["class_idx"]]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader, class_to_idx, train_df, test_df
