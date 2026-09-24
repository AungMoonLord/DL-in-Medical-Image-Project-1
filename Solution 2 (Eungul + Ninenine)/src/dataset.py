"""
Dataset and DataLoader management for Thai Character Recognition.
Includes dataset directory scanning, style detection, stratified splitting,
train-only offline data augmentation, and custom PyTorch Dataset implementation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import (
    AUGMENTED_TRAIN_DIR,
    BATCH_SIZE,
    DATA_DIR,
    MIN_TRAIN_SAMPLES_PER_CLASS,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED,
    STYLE_FOLDER_MAP,
    VALID_EXTENSIONS,
    VAL_RATIO,
)
from src.transforms import get_train_transforms, get_val_transforms


def numeric_sort(path: Path) -> Tuple[int, int, str]:
    """เรียงลำดับโฟลเดอร์ที่เป็นตัวเลขให้ถูกต้องตามค่าตัวเลข (เช่น 161 ก่อน 170)"""
    name = path.name
    if name.isdigit():
        return (0, int(name), "")
    return (1, 0, name)


def detect_source(file_path: Path, data_dir: Path) -> str:
    """
    ตรวจสอบแหล่งที่มาของภาพ (handwritten หรือ printed) จากโครงสร้างพาธย่อย
    """
    try:
        relative_parts = file_path.relative_to(data_dir).parts
    except ValueError:
        return "unknown"

    for part in relative_parts:
        lower_part = part.lower()
        if lower_part in STYLE_FOLDER_MAP:
            return STYLE_FOLDER_MAP[lower_part]

    return "unknown"


def scan_dataset(
    data_dir: Union[str, Path] = DATA_DIR
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str], int]:
    """
    สแกนโฟลเดอร์ Dataset เพื่อรวบรวมไฟล์ภาพทั้งหมด
    สร้างตาราง DataFrame พร้อม Label และตรวจจับ Source (สไตล์การเขียน)
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ Dataset ที่: {data_dir.resolve()}")

    # ค้นหาโฟลเดอร์ของแต่ละคลาส (ข้าม __MACOSX และโฟลเดอร์ที่ขึ้นต้นด้วย .)
    class_dirs = sorted(
        [
            path for path in data_dir.rglob("*")
            if path.is_dir()
            and path.name != "__MACOSX"
            and not path.name.startswith(".")
            and any(
                child.is_file() and child.suffix.lower() in VALID_EXTENSIONS
                for child in path.iterdir()
            )
        ],
        key=numeric_sort
    )

    class_dirs = [path for path in class_dirs if path.name]
    if not class_dirs:
        raise ValueError(f"ไม่พบโฟลเดอร์รูปภาพที่ถูกต้องใน {data_dir.resolve()}")

    class_names = sorted(
        {path.name for path in class_dirs},
        key=lambda x: (int(x) if x.isdigit() else 0, x)
    )
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    num_classes = len(class_names)

    records: List[Dict[str, Union[str, int]]] = []
    for class_dir in class_dirs:
        class_name = class_dir.name
        label = class_to_idx[class_name]

        for file_path in class_dir.iterdir():
            if (
                file_path.is_file()
                and not file_path.name.startswith(".")
                and file_path.suffix.lower() in VALID_EXTENSIONS
                and file_path.stat().st_size > 0  # กรองไฟล์ 0 ไบต์
            ):
                records.append({
                    "path": str(file_path.resolve()),
                    "class_name": class_name,
                    "label": label,
                    "source": detect_source(file_path, data_dir),
                })

    data_df = pd.DataFrame(records)
    return data_df, class_to_idx, idx_to_class, num_classes


def stratified_split_per_group(
    dataframe: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    แบ่งชุดข้อมูลเป็น Train และ Validation โดยกระจายสัดส่วนตามกลุ่ม (label และ source)
    เพื่อรักษาสัดส่วนลายมือและตัวพิมพ์ในทุกๆ ตัวอักษร
    [สำคัญ] กรณีคลาส/กลุ่มที่มีเพียง 1 ภาพ จะส่งเข้า Train ทั้งหมด เพื่อป้องกัน Error จาก Stratified Split
    """
    if group_cols is None:
        group_cols = ["label", "source"]

    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for _, group in dataframe.groupby(group_cols):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)

        num_images = len(indices)
        # หากมีภาพเพียง 1 ภาพ ให้ส่งเข้า Train ทั้งหมด ห้ามแบ่งเข้า Val
        if num_images <= 1:
            train_indices.extend(indices)
            continue

        num_val = max(1, int(round(num_images * val_ratio)))
        num_val = min(num_val, num_images - 1)

        val_indices.extend(indices[:num_val])
        train_indices.extend(indices[num_val:])

    train_df = dataframe.loc[train_indices].reset_index(drop=True)
    val_df = dataframe.loc[val_indices].reset_index(drop=True)

    train_paths = set(train_df["path"])
    val_paths = set(val_df["path"])
    assert train_paths.isdisjoint(val_paths), "พบภาพซ้ำซ้อนระหว่าง Train และ Validation set!"

    return train_df, val_df


def augment_train_set_only(
    train_df: pd.DataFrame,
    target_count: int = MIN_TRAIN_SAMPLES_PER_CLASS,
    output_dir: Optional[Union[str, Path]] = None,
    seed: int = SEED
) -> pd.DataFrame:
    """
    ทำ Data Augmentation (RandomRotation, RandomAffine, ColorJitter)
    เติมเฉพาะภาพในฝั่ง Train Set ให้มีขั้นต่ำ target_count ภาพต่อคลาส (ค่าเริ่มต้น: 80 ภาพ)
    [สำคัญ]: ฝั่ง Validation Set ต้องคงเป็นภาพต้นฉบับเพียวๆ ห้าม Augment เด็ดขาด เพื่อป้องกัน Data Leakage
    """
    if output_dir is None:
        output_dir = AUGMENTED_TRAIN_DIR
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Pipeline สำหรับสร้างภาพสังเคราะห์เพิ่มเติม
    aug_pipeline = transforms.Compose([
        transforms.RandomRotation(degrees=15, fill=255),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.90, 1.10),
            fill=255
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])

    new_records: List[Dict[str, Union[str, int]]] = []
    classes_augmented = 0
    total_new_images = 0

    # จัดกลุ่มตามคลาส
    for label, group in train_df.groupby("label"):
        current_count = len(group)
        if current_count < target_count:
            needed = target_count - current_count
            class_name = group.iloc[0]["class_name"]
            class_out_dir = out_path / str(class_name)
            class_out_dir.mkdir(parents=True, exist_ok=True)

            classes_augmented += 1
            sample_indices = np.random.default_rng(seed + int(label)).choice(
                len(group), size=needed, replace=True
            )

            for i, idx in enumerate(sample_indices):
                src_row = group.iloc[idx]
                src_file = Path(src_row["path"])

                try:
                    with Image.open(src_file) as img:
                        img_rgb = img.convert("RGB")
                    aug_img = aug_pipeline(img_rgb)

                    aug_filename = f"aug_{i}_{src_file.stem}.png"
                    aug_filepath = class_out_dir / aug_filename
                    aug_img.save(aug_filepath, "PNG")

                    new_records.append({
                        "path": str(aug_filepath.resolve()),
                        "class_name": class_name,
                        "label": int(label),
                        "source": "augmented",
                    })
                    total_new_images += 1
                except Exception as e:
                    # หากเกิดข้อผิดพลาดในการเปิดหรือบันทึกภาพ ให้ข้ามไป
                    continue

    if new_records:
        augmented_df = pd.concat([train_df, pd.DataFrame(new_records)], ignore_index=True)
        print(
            f"[*] ทำ Train-only Data Augmentation สำเร็จ: เพิ่มภาพสังเคราะห์ {total_new_images} ภาพ "
            f"ใน {classes_augmented} คลาส (ขั้นต่ำ {target_count} ภาพ/คลาส) "
            f"บันทึกไว้ที่: {out_path.resolve()}"
        )
        return augmented_df

    print(f"[*] ทุกคลาสใน Train Set มีภาพมากกว่าหรือเท่ากับ {target_count} ภาพอยู่แล้ว ไม่ต้อง Augment เพิ่ม")
    return train_df.copy()


class ThaiCharacterDataset(Dataset):
    """
    Custom Dataset สำหรับโหลดภาพตัวอักษรไทย
    มีระบบ Fallback ในกรณีที่ไฟล์ภาพเสียหายหรือไม่สมบูรณ์
    """
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.dataframe.iloc[index]
        image_path = row["path"]

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception:
            try:
                import cv2
                img_cv = cv2.imread(image_path)
                if img_cv is not None:
                    image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                else:
                    image = Image.new("RGB", (224, 224), (255, 255, 255))
            except Exception:
                image = Image.new("RGB", (224, 224), (255, 255, 255))

        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_transform=None,
    val_transform=None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY
) -> Tuple[DataLoader, DataLoader]:
    """
    สร้าง DataLoader สำหรับ Train และ Validation
    """
    if train_transform is None:
        train_transform = get_train_transforms()
    if val_transform is None:
        val_transform = get_val_transforms()

    # กรองเฉพาะไฟล์ที่มีอยู่จริงบนดิสก์
    train_df_clean = train_df[train_df["path"].apply(os.path.exists)].reset_index(drop=True)
    val_df_clean = val_df[val_df["path"].apply(os.path.exists)].reset_index(drop=True)

    train_dataset = ThaiCharacterDataset(train_df_clean, transform=train_transform)
    val_dataset = ThaiCharacterDataset(val_df_clean, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader
