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
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

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

    Optimized for high-throughput batch loading (eliminates slow percentile sort).
    """
    np_img = np.array(img.convert("L"))
    corners = np.array([np_img[0, 0], np_img[0, -1], np_img[-1, 0], np_img[-1, -1]])
    # All raw scans in the dataset have light background (>100) and dark ink strokes (<100).
    # If corners or global mean indicates a light background document image, invert it.
    if np.median(corners) > 100 or np_img.mean() > 100:
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


def _normalize_paths(paths: Optional[Union[str, Path, Sequence[Union[str, Path]]]]) -> List[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def is_valid_glyph_sample(img_path: Union[str, Path], min_pixels: int = 5) -> bool:
    """
    Checks if an image file is readable and contains valid character stroke content
    (not pure black/white blank or unreadable zero-size corruption).
    """
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            if w <= 0 or h <= 0:
                return False
            inv = invert_glyph_if_needed(img)
            arr = np.array(inv.convert("L"))
            # Count bright stroke pixels (threshold at 30/255)
            stroke_pixels = np.count_nonzero(arr > 30)
            return stroke_pixels >= min_pixels
    except Exception:
        return False


def build_split_dataframes(
    dataset_dir: Union[str, Path, Sequence[Union[str, Path]]],
    val_dir: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    train_ratio: float = 0.8,
    use_auto_reject: bool = False,
    min_foreground_pixels: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int]]:
    """
    Builds train and test DataFrames supporting single or multiple dataset sources.
    
    Features:
    - Multiple Paths: Accepts a single directory or a list/tuple of directories.
    - Sparse Classes: Individual directories do not need to contain all classes;
      the union of all available classes across all provided paths is computed.
    - Zero-Leakage: Document page / scanner counterpart grouping applies across all paths.
    - Auto-Reject Sample: When use_auto_reject=True, scans and filters out corrupted,
      empty, or non-character blank crops. When False (default for cleaned sets),
      accepts all samples with maximum I/O indexing speed.
    """
    train_roots = _normalize_paths(dataset_dir)
    val_roots = _normalize_paths(val_dir)

    if not train_roots:
        raise ValueError("At least one dataset directory must be provided.")

    for p in train_roots:
        if not p.exists():
            raise FileNotFoundError(f"Training dataset directory not found: {p}")
    for p in val_roots:
        if not p.exists():
            raise FileNotFoundError(f"Validation dataset directory not found: {p}")

    # Check if train_roots have train/ and val/ (or test/) subfolders
    if not val_roots:
        sub_train, sub_val = [], []
        for p in train_roots:
            if (p / "train").is_dir() and ((p / "val").is_dir() or (p / "test").is_dir()):
                sub_train.append(p / "train")
                sub_val.append(p / "val" if (p / "val").is_dir() else p / "test")
        if len(sub_train) == len(train_roots):
            train_roots = sub_train
            val_roots = sub_val

    # Discover the global union of all class directories across all sources
    all_class_numbers = set()
    for root in (train_roots + val_roots):
        for d in root.iterdir():
            if d.is_dir() and d.name.isdigit() and not d.name.startswith("."):
                all_class_numbers.add(int(d.name))

    if not all_class_numbers:
        raise ValueError(f"No numeric class folders (e.g. 161/, 162/) found in provided paths.")

    class_numbers = sorted(list(all_class_numbers))
    class_to_idx = {c_num: idx for idx, c_num in enumerate(class_numbers)}

    train_records: List[dict] = []
    test_records: List[dict] = []
    rejected_count = 0

    # Case A: Separate validation directories provided
    if val_roots:
        for split_name, roots, records in [("train", train_roots, train_records), ("test", val_roots, test_records)]:
            for root in roots:
                for class_num in class_numbers:
                    c_dir = root / str(class_num)
                    if not c_dir.is_dir():
                        continue
                    class_idx = class_to_idx[class_num]
                    character = tis620_to_char(class_num)
                    for f in sorted(c_dir.iterdir()):
                        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS and not f.name.startswith("."):
                            if use_auto_reject and not is_valid_glyph_sample(f, min_pixels=min_foreground_pixels):
                                rejected_count += 1
                                continue
                            records.append({
                                "filepath": str(f),
                                "filename": f.name,
                                "class_number": class_num,
                                "class_idx": class_idx,
                                "character": character,
                                "source_root": root.name,
                                "group_key": f"{root.stem}_{get_group_key(f.name)}",
                                "split": split_name,
                            })

        if use_auto_reject and rejected_count > 0:
            print(f"🧹 Auto-Reject Filter: Filtered out {rejected_count} corrupted/blank samples.")

        train_df = pd.DataFrame(train_records)
        test_df = pd.DataFrame(test_records)
        return train_df, test_df, class_to_idx

    # Case B: Single/Multiple directories with Zero-Leakage Deterministic Hash Split
    for root in train_roots:
        for class_num in class_numbers:
            c_dir = root / str(class_num)
            if not c_dir.is_dir():
                continue
            class_idx = class_to_idx[class_num]
            character = tis620_to_char(class_num)

            files = sorted(
                [
                    f.name
                    for f in c_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS and not f.name.startswith(".")
                ]
            )
            if not files:
                continue

            groups = defaultdict(list)
            for f in files:
                f_path = c_dir / f
                if use_auto_reject and not is_valid_glyph_sample(f_path, min_pixels=min_foreground_pixels):
                    rejected_count += 1
                    continue
                g_key = f"{root.stem}_{get_group_key(f)}"
                groups[g_key].append(f)

            for g_key, g_files in groups.items():
                hash_val = int(hashlib.sha256(g_key.encode("utf-8")).hexdigest(), 16)
                assigned_split = "train" if (hash_val % 100) < int(train_ratio * 100) else "test"

                target_records = train_records if assigned_split == "train" else test_records
                for f_name in g_files:
                    target_records.append({
                        "filepath": str(c_dir / f_name),
                        "filename": f_name,
                        "class_number": class_num,
                        "class_idx": class_idx,
                        "character": character,
                        "source_root": root.name,
                        "group_key": g_key,
                        "split": assigned_split,
                    })

    if use_auto_reject and rejected_count > 0:
        print(f"🧹 Auto-Reject Filter: Filtered out {rejected_count} corrupted/blank samples.")

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)
    return train_df, test_df, class_to_idx


class ThaiCharacterDataset(Dataset):
    """
    PyTorch Dataset for Thai Character Glyphs with input inversion,
    optional focal artifact cleaning, and typography pedestal augmentation.
    Supports high-speed in-RAM caching for massive datasets (e.g. 300k+ images).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_size: Tuple[int, int] = (32, 32),
        transform: Optional[Callable] = None,
        use_focal_cleaner: bool = False,
        use_pedestal_aug: bool = False,
        pad_ratio: float = 0.10,
        cache_in_ram: bool = False,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.target_size = target_size
        self.transform = transform
        self.pad_ratio = pad_ratio
        self.focal_cleaner = FocalElementCleaner() if use_focal_cleaner else None
        self.pedestal_aug = PedestalAugmentation(p=0.5) if use_pedestal_aug else None
        self.cache_in_ram = cache_in_ram

        self.labels = np.array(self.df["class_idx"].values, dtype=np.int64)
        self.class_numbers = np.array(self.df["class_number"].values, dtype=np.int64)
        self.cached_images: Optional[np.ndarray] = None

        if self.cache_in_ram:
            import time
            n_samples = len(self.df)
            print(f"[CACHE] Pre-caching {n_samples:,} images into RAM ({target_size[0]}x{target_size[1]} uint8)...")
            t_start = time.time()
            self.cached_images = np.zeros((n_samples, target_size[1], target_size[0]), dtype=np.uint8)

            for idx, row in enumerate(self.df.itertuples()):
                img = Image.open(row.filepath).convert("RGB")
                img = invert_glyph_if_needed(img)
                if self.focal_cleaner is not None:
                    img = self.focal_cleaner.clean(img)
                letterboxed = letterbox_pad(img, target_size=self.target_size, pad_ratio=self.pad_ratio, fill_color=(0, 0, 0))
                self.cached_images[idx] = np.array(letterboxed.convert("L"), dtype=np.uint8)

            elapsed = time.time() - t_start
            mem_mb = self.cached_images.nbytes / (1024 * 1024)
            print(f"[CACHE] RAM caching complete: {n_samples:,} images in {elapsed:.1f}s ({mem_mb:.1f} MB RAM). Training throughput is now at maximum speed.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        class_idx = int(self.labels[idx])
        class_number = int(self.class_numbers[idx])

        if self.cached_images is not None:
            arr = self.cached_images[idx]
            img = Image.fromarray(arr).convert("RGB")
        else:
            row = self.df.iloc[idx]
            img = Image.open(row["filepath"]).convert("RGB")
            img = invert_glyph_if_needed(img)
            if self.focal_cleaner is not None:
                img = self.focal_cleaner.clean(img)
            img = letterbox_pad(img, target_size=self.target_size, pad_ratio=self.pad_ratio, fill_color=(0, 0, 0))

        # Optional Pedestal Augmentation for ญ (173) and ฐ (176)
        if self.pedestal_aug is not None and class_number in (173, 176):
            img = self.pedestal_aug(img, class_number=class_number)

        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = get_val_transform()(img)

        return tensor, class_idx, class_number


def build_dataloaders(
    dataset_dir: Union[str, Path, Sequence[Union[str, Path]]],
    val_dir: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    train_ratio: float = 0.8,
    batch_size: int = 64,
    target_size: Tuple[int, int] = (32, 32),
    num_workers: int = 0,
    use_balanced_sampler: bool = False,
    use_focal_cleaner: bool = False,
    use_pedestal_aug: bool = True,
    use_auto_reject: bool = False,
    cache_in_ram: bool = False,
) -> Tuple[DataLoader, DataLoader, Dict[int, int], pd.DataFrame, pd.DataFrame]:
    """
    Constructs train and test DataLoaders with zero data leakage.
    Note: When using class-balanced loss (CB Focal), set use_balanced_sampler=False
    to prevent quadratic over-weighting of rare classes.
    """
    train_df, test_df, class_to_idx = build_split_dataframes(
        dataset_dir=dataset_dir,
        val_dir=val_dir,
        train_ratio=train_ratio,
        use_auto_reject=use_auto_reject,
    )

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_dataset = ThaiCharacterDataset(
        dataframe=train_df,
        target_size=target_size,
        transform=train_transform,
        use_focal_cleaner=use_focal_cleaner,
        use_pedestal_aug=use_pedestal_aug,
        cache_in_ram=cache_in_ram,
    )

    test_dataset = ThaiCharacterDataset(
        dataframe=test_df,
        target_size=target_size,
        transform=val_transform,
        use_focal_cleaner=use_focal_cleaner,
        use_pedestal_aug=False,
        cache_in_ram=cache_in_ram,
    )

    persistent = num_workers > 0

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
            persistent_workers=persistent,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=persistent,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
    )

    return train_loader, test_loader, class_to_idx, train_df, test_df
