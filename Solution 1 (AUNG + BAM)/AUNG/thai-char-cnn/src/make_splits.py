"""
แบ่ง train/val 80:20 พร้อมป้องกัน data leakage

ปัญหาเฉพาะของ dataset นี้:
  • 'ฃ' และ 'ฑ' มีภาพเดียว → แบ่ง 80:20 ตามปกติไม่ได้
  • 'ฬ' มี 3 ภาพ, '๗' มี 4 ภาพ → 20% = 0 ภาพ ถ้าปัดลง

นโยบาย (singleton_policy = val_first):
  1. คลาสที่มี 1 ภาพ → ภาพจริงไปอยู่ VAL (เพื่อให้วัดผลได้จริง)
     แล้ว offline_augment จะสร้างสำเนา augmented ไปอยู่ TRAIN
  2. ทุกคลาสต้องมีอย่างน้อย min_val_per_class ภาพใน val
  3. origin_id ห้ามซ้ำข้าม split — มี assertion ตรวจก่อนเขียนไฟล์
  4. val.csv ต้องมี is_augmented == 0 ทุกแถว

⚠️ ข้อ 3 คือจุดที่พลาดแล้วได้ accuracy ปลอม ๆ 99% แล้วโดนอาจารย์จับได้
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .class_map import ClassMap
from .config import load_config
from .utils import ensure_dir, list_images, seed_everything, setup_logger

COLUMNS = ["filepath", "folder_id", "class_name", "label",
           "is_augmented", "origin_id"]


def scan_raw(raw_dir: str | Path, class_map: ClassMap) -> pd.DataFrame:
    raw = Path(raw_dir)
    rows = []
    for folder in class_map.folders:
        label = class_map.folder_to_label[folder]
        char = class_map.label_to_char[label]
        for p in list_images(raw / str(folder)):
            rows.append({
                "filepath": p.as_posix(),
                "folder_id": folder,
                "class_name": char,
                "label": label,
                "is_augmented": 0,
                "origin_id": f"{folder}_{p.stem}",
            })
    if not rows:
        raise RuntimeError(f"ไม่พบภาพใด ๆ ใน {raw}")
    return pd.DataFrame(rows, columns=COLUMNS)


def stratified_split(df: pd.DataFrame, ratio: float = 0.8, seed: int = 42,
                     min_val: int = 1, singleton_policy: str = "val_first",
                     logger=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts, val_parts, notes = [], [], []

    for label, group in df.groupby("label", sort=True):
        g = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(g)
        char = g["class_name"].iloc[0]

        if n == 1:
            if singleton_policy == "val_first":
                val_parts.append(g)
                notes.append(f"  '{char}' มี 1 ภาพ → VAL (train รอ augmented)")
            else:
                train_parts.append(g)
                notes.append(f"  '{char}' มี 1 ภาพ → TRAIN (ไม่มี val)")
            continue

        n_val = max(min_val, int(round(n * (1 - ratio))))
        n_val = min(n_val, n - 1)          # train ต้องเหลืออย่างน้อย 1
        val_parts.append(g.iloc[:n_val])
        train_parts.append(g.iloc[n_val:])
        if n < 10:
            notes.append(f"  '{char}' มี {n} ภาพ → train {n-n_val} / val {n_val}")

    train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=COLUMNS)
    val = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=COLUMNS)

    train = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = val.sort_values(["label", "filepath"]).reset_index(drop=True)

    if logger and notes:
        logger.info("คลาสที่ต้องจัดการพิเศษ:\n" + "\n".join(notes))
    return train, val


def assert_no_leakage(train: pd.DataFrame, val: pd.DataFrame) -> None:
    overlap_origin = set(train["origin_id"]) & set(val["origin_id"])
    if overlap_origin:
        raise AssertionError(
            f"DATA LEAKAGE: origin_id ซ้ำ {len(overlap_origin)} รายการ "
            f"เช่น {sorted(overlap_origin)[:5]}"
        )
    overlap_path = set(train["filepath"]) & set(val["filepath"])
    if overlap_path:
        raise AssertionError(f"DATA LEAKAGE: filepath ซ้ำ {len(overlap_path)} รายการ")
    if (val["is_augmented"] != 0).any():
        raise AssertionError("val.csv ต้องมีเฉพาะภาพต้นฉบับ (is_augmented == 0)")


def report(train: pd.DataFrame, val: pd.DataFrame, class_map: ClassMap) -> str:
    n_cls = class_map.num_classes
    t = np.bincount(train["label"], minlength=n_cls)
    v = np.bincount(val["label"], minlength=n_cls)

    lines = [
        "=" * 68, "SPLIT SUMMARY", "=" * 68,
        f"  train : {len(train):>7,} ภาพ  ({len(train)/(len(train)+len(val))*100:.1f}%)",
        f"  val   : {len(val):>7,} ภาพ  ({len(val)/(len(train)+len(val))*100:.1f}%)",
        f"  รวม   : {len(train)+len(val):>7,} ภาพ",
        "-" * 68,
        f"  คลาสที่ไม่มีข้อมูลใน train : {int((t == 0).sum())}  "
        f"→ {' '.join(class_map.label_to_char[i] for i in np.where(t == 0)[0])}",
        f"  คลาสที่ไม่มีข้อมูลใน val   : {int((v == 0).sum())}",
        "-" * 68,
        "  10 คลาสที่เล็กที่สุดใน train:",
        f"  {'class':<8}{'train':>8}{'val':>8}",
    ]
    for i in np.argsort(t)[:10]:
        lines.append(f"  {class_map.label_to_char[i]:<8}{t[i]:>8,}{v[i]:>8,}")
    lines.append("=" * 68)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="แบ่ง train/val แบบกัน leakage")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ratio", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--no-strict", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = args.raw_dir or cfg.get_path("paths.raw_dir", "data/raw")
    ratio = args.ratio if args.ratio is not None else cfg.get_path("split.ratio", 0.8)
    seed = args.seed if args.seed is not None else cfg.get_path("project.seed", 42)

    logger = setup_logger("make_splits",
                          Path(cfg.get_path("paths.log_dir", "outputs/logs")) / "make_splits.log")
    seed_everything(seed)

    logger.info(f"สแกน {raw_dir} ...")
    cm = ClassMap.from_raw_dir(raw_dir, strict=not args.no_strict)
    cm.save(cfg.get_path("paths.class_map", "data/splits/class_map.json"))
    logger.info(f"✓ class_map.json | {cm.num_classes} คลาส | "
                f"{sum(cm.counts.values()):,} ภาพ")

    df = scan_raw(raw_dir, cm)
    train, val = stratified_split(
        df, ratio=ratio, seed=seed,
        min_val=cfg.get_path("split.min_val_per_class", 1),
        singleton_policy=cfg.get_path("split.singleton_policy", "val_first"),
        logger=logger)

    if cfg.get_path("split.enforce_no_leakage", True):
        assert_no_leakage(train, val)
        logger.info("✓ ตรวจ leakage ผ่าน")

    train_csv = cfg.get_path("paths.train_csv", "data/splits/train.csv")
    val_csv = cfg.get_path("paths.val_csv", "data/splits/val.csv")
    ensure_dir(train_csv)
    train.to_csv(train_csv, index=False, encoding="utf-8-sig")
    val.to_csv(val_csv, index=False, encoding="utf-8-sig")

    logger.info("\n" + report(train, val, cm))
    logger.info(f"✓ บันทึก {train_csv} และ {val_csv}")
    logger.info("ขั้นถัดไป: python -m src.offline_augment --min-count 200")


if __name__ == "__main__":
    main()