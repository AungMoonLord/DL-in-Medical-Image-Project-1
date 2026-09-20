"""ThaiCharDataset — อ่านจาก CSV ที่ make_splits.py สร้างไว้."""
from __future__ import annotations
from .sampler import build_sampler
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .class_map import ClassMap
from .transforms import (build_train_transform, build_val_transform)
from .utils import worker_init_fn

REQUIRED_COLUMNS = {"filepath", "folder_id", "class_name",
                    "label", "is_augmented", "origin_id"}


def read_image(path: str | Path) -> np.ndarray:
    """อ่านภาพเป็น RGB uint8 (รองรับ path ภาษาไทย/ยูนิโค้ดบน Windows)."""
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:  # fallback สำหรับ path ที่ cv2 อ่านไม่ได้
        try:
            buf = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        except Exception:
            img = None
    if img is None:
        raise FileNotFoundError(f"อ่านภาพไม่ได้: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        # รวม alpha กับพื้นขาว (PNG โปร่งใสจะกลายเป็นดำสนิทถ้าไม่ทำ)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (rgb * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(img)


class ThaiCharDataset(Dataset):
    """
    คืน (image_tensor, label) หรือ (image_tensor, label, index) ถ้า return_index=True
    แถวเสียจะถูกข้ามอัตโนมัติ (แทนด้วยภาพขาว) แทนที่จะทำให้ training ล้ม
    """

    def __init__(self, csv_path: str | Path, transform=None,
                 root: str | Path = ".", return_index: bool = False,
                 verify: bool = False):
        self.df = pd.read_csv(csv_path)
        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"{csv_path} ขาดคอลัมน์: {sorted(missing)}")

        self.root = Path(root)
        self.transform = transform
        self.return_index = return_index

        self.paths = self.df["filepath"].astype(str).tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.is_aug = self.df["is_augmented"].astype(int).tolist()

        if verify:
            self._verify_files()

    def _verify_files(self) -> None:
        missing = [p for p in self.paths if not (self.root / p).exists()]
        if missing:
            raise FileNotFoundError(
                f"ไม่พบไฟล์ {len(missing)} รายการ เช่น {missing[:5]}\n"
                "ลองรัน make_splits.py / offline_augment.py ใหม่"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        try:
            img = read_image(self.root / self.paths[idx])
        except Exception as e:
            print(f"[warn] ข้ามไฟล์เสีย {self.paths[idx]}: {e}")
            size = 224
            img = np.full((size, size, 3), 255, np.uint8)

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return (img, label, idx) if self.return_index else (img, label)

    # ---------------- helpers ----------------
    def get_labels(self) -> np.ndarray:
        return np.asarray(self.labels, dtype=np.int64)

    def class_counts(self, num_classes: int) -> np.ndarray:
        return np.bincount(self.get_labels(), minlength=num_classes)

    def summary(self) -> str:
        n_aug = int(sum(self.is_aug))
        return (f"{len(self):,} ภาพ | ต้นฉบับ {len(self)-n_aug:,} | "
                f"augmented {n_aug:,} | {len(set(self.labels))} คลาส")


class InferenceDataset(Dataset):
    """สำหรับ inference บนโฟลเดอร์ภาพที่ไม่มี label."""

    def __init__(self, paths: list[str | Path], transform=None):
        self.paths = [Path(p) for p in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = read_image(self.paths[idx])
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, str(self.paths[idx])


# ----------------------------------------------------------------------------
# DataLoader factory
# ----------------------------------------------------------------------------
def build_dataloaders(cfg, class_map: ClassMap, stage: str = "stage1",
                      sampler=None) -> tuple[DataLoader, DataLoader,
                                             ThaiCharDataset, ThaiCharDataset]:
    train_ds = ThaiCharDataset(cfg.get_path("paths.train_csv"),
                               transform=build_train_transform(cfg))
    val_ds = ThaiCharDataset(cfg.get_path("paths.val_csv"),
                             transform=build_val_transform(cfg))
    sampler_mode = cfg.get_path(f"train.{stage}.sampler", "instance")
    if sampler is None and sampler_mode != "instance":
        sampler = build_sampler(
            labels=train_ds.get_labels(),
            num_classes=class_map.num_classes,
            mode=sampler_mode,
        )

    bs = cfg.get_path(f"train.{stage}.batch_size", 64)
    nw = cfg.get_path("hardware.num_workers", 8)
    common = dict(
        num_workers=nw,
        pin_memory=cfg.get_path("hardware.pin_memory", True),
        persistent_workers=bool(nw) and cfg.get_path("hardware.persistent_workers", True),
        prefetch_factor=cfg.get_path("hardware.prefetch_factor", 4) if nw else None,
        worker_init_fn=worker_init_fn,
    )

    train_loader = DataLoader(train_ds, batch_size=bs,
                              shuffle=(sampler is None), sampler=sampler,
                              drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False,
                            drop_last=False, **common)
    return train_loader, val_loader, train_ds, val_ds