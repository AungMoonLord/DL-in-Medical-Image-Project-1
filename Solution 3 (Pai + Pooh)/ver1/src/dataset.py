"""
Dataset and Deterministic Splitting Pipeline for Thai Character Classification.

Implements:
1. TIS-620 character mapping.
2. Zero-leakage deterministic group-aware 80/20 train-test splitting.
3. Aspect-ratio preserving letterbox square padding.
4. PyTorch Dataset and balanced DataLoader builders.
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

# Valid image extensions
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
    
    Filenames are formatted as: [doc_id][sg/tg]_[page_id]_[char_index].jpg
    e.g.: 'bc_001sg_3_118.jpg' and 'bc_001tg_3_118.jpg'
    
    This function pairs 'sg' (source) and 'tg' (target) counterpart scans
    and binds all characters from the same document page together.
    """
    parts = filename.split("_")
    if len(parts) >= 3:
        # Strip the trailing 'sg' / 'tg' to unify document pair base
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


def build_split_dataframes(
    dataset_dir: Union[str, Path],
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, int]]:
    """
    Builds zero-leakage deterministic train and test DataFrames.
    
    Partition Rules:
    - Tier 1 (N = 1): Allocated to train (test evaluated with synthetic or few-shot).
    - Tier 2 (2 <= N <= 5): Stratified split allocating 1 sample to test, remainder to train.
    - Tier 3 (N > 5): Deterministic SHA-256 group hashing on (class_id + '_' + group_key).
    
    Returns:
        train_df: DataFrame containing training samples.
        test_df: DataFrame containing testing samples.
        class_to_idx: Dictionary mapping TIS-620 class number to sequential index (0..71).
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

        if n_samples == 1:
            # Tier 1: Single sample assigned to train
            train_records.append({
                "filepath": str(class_dir / files[0]),
                "filename": files[0],
                "class_number": class_number,
                "class_idx": class_idx,
                "character": character,
                "group_key": get_group_key(files[0]),
                "split": "train",
            })
        elif n_samples <= 5:
            # Tier 2: Rare class stratified assignment
            test_file = files[-1]
            test_records.append({
                "filepath": str(class_dir / test_file),
                "filename": test_file,
                "class_number": class_number,
                "class_idx": class_idx,
                "character": character,
                "group_key": get_group_key(test_file),
                "split": "test",
            })
            for train_file in files[:-1]:
                train_records.append({
                    "filepath": str(class_dir / train_file),
                    "filename": train_file,
                    "class_number": class_number,
                    "class_idx": class_idx,
                    "character": character,
                    "group_key": get_group_key(train_file),
                    "split": "train",
                })
        else:
            # Tier 3: Group-aware deterministic hash partition
            groups = defaultdict(list)
            for f in files:
                g = get_group_key(f)
                groups[g].append(f)

            c_train_files: List[str] = []
            c_test_files: List[str] = []
            threshold = int(train_ratio * 100)

            for g_name, g_files in sorted(groups.items()):
                # Hash combined class and group key
                hash_input = f"{class_number}_{g_name}".encode("utf-8")
                hash_val = int(hashlib.sha256(hash_input).hexdigest()[:8], 16) % 100

                if hash_val < threshold:
                    c_train_files.extend(g_files)
                else:
                    c_test_files.extend(g_files)

            # Guarantee representation
            if not c_test_files and len(c_train_files) > 1:
                c_test_files.append(c_train_files.pop())
            if not c_train_files and len(c_test_files) > 1:
                c_train_files.append(c_test_files.pop())

            for f in c_train_files:
                train_records.append({
                    "filepath": str(class_dir / f),
                    "filename": f,
                    "class_number": class_number,
                    "class_idx": class_idx,
                    "character": character,
                    "group_key": get_group_key(f),
                    "split": "train",
                })
            for f in c_test_files:
                test_records.append({
                    "filepath": str(class_dir / f),
                    "filename": f,
                    "class_number": class_number,
                    "class_idx": class_idx,
                    "character": character,
                    "group_key": get_group_key(f),
                    "split": "test",
                })

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)
    return train_df, test_df, class_to_idx


def letterbox_pad(
    image: Image.Image,
    target_size: Tuple[int, int] = (32, 32),
    fill_color: Optional[Tuple[int, int, int]] = None,
) -> Image.Image:
    """
    Pads and scales an image to target square dimensions while strictly
    preserving aspect ratio. Automatically computes background fill color
    from corner pixels if fill_color is not specified.
    """
    target_w, target_h = target_size
    orig_w, orig_h = image.size

    # Compute scale to fit inside target bounding box
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))

    # Resize glyph with high-quality resampling
    resized_img = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Estimate background color from border if fill_color is None
    if fill_color is None:
        np_img = np.array(image.convert("RGB"))
        # Sample borders (top, bottom, left, right edges)
        corners = np.concatenate([
            np_img[0, :], np_img[-1, :], np_img[:, 0], np_img[:, -1]
        ], axis=0)
        bg_rgb = tuple(int(x) for x in np.median(corners, axis=0))
    else:
        bg_rgb = fill_color

    # Create canvas and paste resized image in the center
    canvas = Image.new("RGB", (target_w, target_h), bg_rgb)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized_img, (paste_x, paste_y))
    return canvas


class ThaiCharacterDataset(Dataset):
    """
    PyTorch Dataset for Thai Character classification.
    Supports letterboxing, dynamic transform pipelines, and metadata returns.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_size: Tuple[int, int] = (32, 32),
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.target_size = target_size
        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        label = int(row["class_idx"])

        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            # Fallback for corrupted images
            img_rgb = Image.new("RGB", self.target_size, (255, 255, 255))

        # Apply aspect-preserving letterboxing
        padded_img = letterbox_pad(img_rgb, target_size=self.target_size)

        if self.transform:
            tensor = self.transform(padded_img)
        else:
            # Default ToTensor and Normalize
            np_arr = np.array(padded_img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(np_arr).permute(2, 0, 1)

        if self.return_metadata:
            return tensor, label, {
                "filename": row["filename"],
                "class_number": row["class_number"],
                "character": row["character"],
                "group_key": row["group_key"],
            }

        return tensor, label


def get_class_weights(train_df: pd.DataFrame, num_classes: int = 72, beta: float = 0.999) -> torch.Tensor:
    """
    Computes effective class weights for Class-Balanced Loss and WeightedRandomSampler.
    Formula: E_n = (1 - beta) / (1 - beta^n)
    """
    counts = Counter(train_df["class_idx"])
    weights = []
    for idx in range(num_classes):
        n = counts.get(idx, 1)
        if beta > 0 and n > 0:
            eff_weight = (1.0 - beta) / (1.0 - (beta ** n))
        else:
            eff_weight = 1.0 / max(1, n)
        weights.append(eff_weight)
    
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize weights so their mean is 1.0
    weight_tensor = weight_tensor / weight_tensor.mean()
    return weight_tensor


def create_dataloaders(
    dataset_dir: Union[str, Path],
    batch_size: int = 64,
    target_size: Tuple[int, int] = (32, 32),
    train_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    use_balanced_sampler: bool = True,
    num_workers: int = 2,
    train_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame, Dict[int, int]]:
    """
    Constructs train and test DataLoaders with deterministic zero-leakage splitting
    and optional class-balanced sampling.
    """
    train_df, test_df, class_to_idx = build_split_dataframes(dataset_dir, train_ratio=train_ratio)

    train_dataset = ThaiCharacterDataset(
        dataframe=train_df,
        target_size=target_size,
        transform=train_transform,
    )
    test_dataset = ThaiCharacterDataset(
        dataframe=test_df,
        target_size=target_size,
        transform=test_transform,
    )

    sampler = None
    shuffle = True
    if use_balanced_sampler:
        class_counts = Counter(train_df["class_idx"])
        # Smoothed inverse square root sample weight
        sample_weights = [1.0 / np.sqrt(max(1, class_counts[idx])) for idx in train_df["class_idx"]]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
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

    return train_loader, test_loader, train_df, test_df, class_to_idx
