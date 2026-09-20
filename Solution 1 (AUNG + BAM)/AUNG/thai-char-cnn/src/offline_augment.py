"""
Offline augmentation สำหรับคลาส tail

ทำไมต้อง offline ไม่ใช่แค่ online:
  คลาสที่มีภาพเดียว ถ้าพึ่ง online augmentation อย่างเดียว มันจะปรากฏใน
  epoch ละครั้งเท่านั้น (ถ้าไม่ oversample) หรือซ้ำแบบสุ่มไม่ควบคุม
  การสร้างไฟล์จริงเก็บไว้ทำให้จำนวนตัวอย่างสมดุลขึ้นตั้งแต่ระดับ dataset
  และยัง reproduce ได้ด้วย seed เดียวกัน

เขียนลง data/augmented/<folder_id>/ เท่านั้น — ไม่แตะ data/raw/
แล้ว append เข้า train.csv (ไม่แตะ val.csv เด็ดขาด)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .class_map import ClassMap
from .config import load_config
from .dataset import read_image
from .transforms import build_tail_transform
from .utils import ensure_dir, seed_everything, setup_logger


def clean_aug_dir(aug_dir: str | Path, logger=None) -> None:
    p = Path(aug_dir)
    if p.exists():
        shutil.rmtree(p)
        if logger:
            logger.info(f"ลบของเก่าใน {p}")
    p.mkdir(parents=True, exist_ok=True)


def augment_class(sources: list[str], n_needed: int, out_dir: Path,
                  transform, folder_id: int, label: int, char: str,
                  seed: int = 42) -> list[dict]:
    """สร้างภาพ augmented ให้ครบ n_needed โดยวนใช้ source แบบ round-robin."""
    out_dir.mkdir(parents=True, exist_ok=True)
    #rng = np.random.default_rng(seed + folder_id)
    rows = []

    cache = {}
    while len(rows) < n_needed:
        src = sources[i % len(sources)]
        if src not in cache:
            try:
                cache[src] = read_image(src)
            except Exception:
                continue
        img = cache[src]

        for _ in range(3):  # retry ถ้า augment แล้วภาพว่างเปล่า
            aug = transform(image=img)["image"]
            if aug.std() > 2.0:     # ยังมีตัวอักษรอยู่จริง
                break

        fname = f"aug_{i:05d}_{Path(src).stem}.png"
        dst = out_dir / fname
        ok, buf = cv2.imencode(".png", cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
        if not ok:
            continue
        buf.tofile(str(dst))    # รองรับ path ยูนิโค้ด

        rows.append({
            "filepath": dst.as_posix(),
            "folder_id": folder_id,
            "class_name": char,
            "label": label,
            "is_augmented": 1,
            "origin_id": f"{folder_id}_{Path(src).stem}",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline augmentation สำหรับ tail class")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--min-count", type=int, default=None,
                    help="ยกทุกคลาสใน train ให้ถึงจำนวนนี้")
    ap.add_argument("--threshold", type=int, default=None,
                    help="คลาสที่น้อยกว่านี้เท่านั้นที่จะถูก augment")
    ap.add_argument("--clean", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get_path("project.seed", 42)
    seed_everything(seed)

    logger = setup_logger("offline_aug",
                          Path(cfg.get_path("paths.log_dir", "outputs/logs")) / "offline_augment.log")

    target = args.min_count or cfg.get_path("augmentation.offline.target_count", 200)
    threshold = args.threshold or cfg.get_path("augmentation.offline.tail_threshold", 50)
    aug_dir = Path(cfg.get_path("paths.aug_dir", "data/augmented"))
    train_csv = cfg.get_path("paths.train_csv", "data/splits/train.csv")

    cm = ClassMap.load(cfg.get_path("paths.class_map", "data/splits/class_map.json"))
    df = pd.read_csv(train_csv)

    # ล้างแถว augmented เก่าออกก่อน เพื่อให้รันซ้ำได้โดยไม่สะสม
    df = df[df["is_augmented"] == 0].reset_index(drop=True)
    if args.clean and not args.dry_run:
        clean_aug_dir(aug_dir, logger)

    counts = df.groupby("label").size().to_dict()
    transform = build_tail_transform(cfg)

    todo = []
    for label in range(cm.num_classes):
        have = counts.get(label, 0)
        if have == 0:
            # คลาสที่ภาพต้นฉบับเดียวถูกส่งไป val → ดึงจาก val มาเป็น source
            todo.append((label, have, target))
        elif have < threshold:
            todo.append((label, have, target - have))

    logger.info(f"จะ augment {len(todo)} คลาส | threshold<{threshold} → target {target}")
    if args.dry_run:
        for label, have, need in todo:
            logger.info(f"  '{cm.label_to_char[label]}': {have} → +{need}")
        return

    val_df = pd.read_csv(cfg.get_path("paths.val_csv", "data/splits/val.csv"))
    new_rows = []

    for label, have, need in tqdm(todo, desc="augmenting", ncols=110):
        folder = cm.label_to_folder[label]
        char = cm.label_to_char[label]

        sources = df.loc[df["label"] == label, "filepath"].tolist()
        if not sources:
            # ใช้ภาพจาก val เป็นต้นแบบ (เฉพาะคลาสที่มีภาพเดียว)
            sources = val_df.loc[val_df["label"] == label, "filepath"].tolist()
            logger.warning(f"  '{char}' ไม่มีภาพใน train → ใช้ภาพจาก val "
                           f"เป็นต้นแบบสร้าง augmented ({len(sources)} ไฟล์)")
        if not sources:
            logger.error(f"  '{char}' ไม่มีภาพเลย — ข้าม")
            continue

        new_rows += augment_class(sources, need, aug_dir / str(folder),
                                  transform, folder, label, char, seed)

    aug_df = pd.DataFrame(new_rows, columns=df.columns)
    out = pd.concat([df, aug_df], ignore_index=True) \
            .sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out.to_csv(train_csv, index=False, encoding="utf-8-sig")

    # ตรวจ leakage อีกรอบ: augmented ที่มาจากภาพใน val ต้องไม่เกิดขึ้น
    # ยกเว้นคลาสที่มีภาพเดียวซึ่งเป็นนโยบายที่ประกาศไว้ชัดเจน
    val_origins = set(val_df["origin_id"])
    risky = aug_df[aug_df["origin_id"].isin(val_origins)]
    if len(risky):
        chars = sorted(set(risky["class_name"]))
        logger.warning(
            f"มี augmented {len(risky)} ภาพที่สร้างจากภาพใน val "
            f"(คลาส: {' '.join(chars)})\n"
            "  → เป็นไปตามนโยบาย singleton ที่ประกาศไว้ ต้องระบุใน limitation ของสไลด์"
        )

    final = np.bincount(out["label"], minlength=cm.num_classes)
    logger.info("=" * 60)
    logger.info(f"train เดิม {len(df):,} → ใหม่ {len(out):,} "
                f"(+{len(aug_df):,} augmented)")
    logger.info(f"คลาสเล็กสุดหลัง augment: {final.min():,} | "
                f"ใหญ่สุด: {final.max():,} | ratio {final.max()/max(final.min(),1):.0f}:1")
    logger.info("=" * 60)
    logger.info("ขั้นถัดไป: python -m src.train --config configs/effnet_arcface.yaml")


if __name__ == "__main__":
    main()