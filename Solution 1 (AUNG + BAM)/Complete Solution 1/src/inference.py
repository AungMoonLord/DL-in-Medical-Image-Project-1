"""
Inference — ไฟล์ที่ต้องส่งอาจารย์ (ปรับปรุงการแปลง folder -> class_id)

ใช้งาน:
  python -m src.inference --ckpt outputs/checkpoints/best_model.pth \
      --input path/to/images --output outputs/predictions.csv --tta 5

  --input รับได้ทั้ง: ไฟล์ภาพเดียว / โฟลเดอร์ / โฟลเดอร์ซ้อนโฟลเดอร์
  รองรับโฟลเดอร์ที่เป็น class_id (เช่น 101), folder_id (เช่น 161) หรือตัวอักษรภาษาไทย (เช่น ก)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .class_map import ClassMap
from .config import Config, load_config
from .dataset import InferenceDataset
from .models import build_model
from .transforms import build_tta_transforms, build_val_transform
from .utils import (IMG_EXTS, amp_dtype_from_str, describe_device, ensure_dir,
                    get_device, load_checkpoint)

# ตารางจับคู่ Folder ID -> Class ID ตาม Project_1-data_dict.txt
FOLDER_TO_CLASS_ID: dict[int, int] = {
    161: 101, 162: 102, 163: 103, 164: 104, 167: 105, 168: 106, 169: 107, 170: 108,
    171: 109, 173: 110, 175: 111, 176: 112, 177: 113, 178: 114, 179: 115, 180: 116,
    181: 117, 182: 118, 183: 119, 184: 120, 185: 121, 186: 122, 187: 123, 188: 124,
    189: 125, 190: 126, 191: 127, 192: 128, 193: 129, 194: 130, 195: 131, 196: 132,
    197: 133, 199: 134, 200: 135, 201: 136, 202: 137, 203: 138, 204: 139, 205: 140,
    206: 141, 207: 201, 209: 202, 210: 203, 212: 204, 213: 205, 214: 206, 215: 207,
    216: 208, 217: 209, 224: 210, 225: 211, 226: 212, 227: 213, 228: 214, 229: 215,
    230: 216, 231: 217, 232: 218, 233: 219, 234: 220, 236: 221, 240: 301, 241: 302,
    242: 303, 243: 304, 244: 305, 245: 306, 246: 307, 247: 308, 248: 309, 249: 310,
}
CLASS_ID_TO_FOLDER: dict[int, int] = {v: k for k, v in FOLDER_TO_CLASS_ID.items()}


def collect_images(input_path: str | Path) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"ไม่พบ: {p}")
    files = sorted(f for f in p.rglob("*")
                   if f.is_file() and f.suffix.lower() in IMG_EXTS)
    if not files:
        raise RuntimeError(f"ไม่พบไฟล์ภาพใน {p}")
    return files


class ThaiCharPredictor:
    """ห่อหุ้มโมเดลให้เรียกใช้ง่าย — ใช้ใน notebook หรือ import ไปใช้ต่อได้."""

    def __init__(self, ckpt_path: str | Path, config_path: str | None = None,
                 device: str = "cuda", tta: int = 1):
        ckpt = load_checkpoint(ckpt_path)
        self.cfg = load_config(config_path) if config_path else Config(ckpt["config"])
        self.device = get_device(device)

        cm_data = ckpt.get("class_map") or {}
        if cm_data:
            counts = {int(k): int(v) for k, v in cm_data.get("class_counts", {}).items()}
            self.class_map = ClassMap([int(f) for f in cm_data["folders"]], counts)
        else:
            self.class_map = ClassMap.load(self.cfg.get_path("paths.class_map"))

        infer_cfg = Config(self.cfg.to_dict())
        infer_cfg.set_path("model.pretrained", False)

        counts_by_label = np.asarray(self.class_map.counts_by_label())
        self.model = build_model(infer_cfg, class_counts=counts_by_label).to(self.device)

        state_dict = ckpt["model_state"]
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        if self.cfg.get_path("hardware.channels_last", True):
            self.model = self.model.to(memory_format=torch.channels_last)

        self.tta = max(1, int(tta))
        self.transforms = (build_tta_transforms(self.cfg, self.tta) if self.tta > 1
                           else [build_val_transform(self.cfg)])
        self.amp = bool(self.cfg.get_path("hardware.amp", True)) and (self.device.type == "cuda")
        self.amp_dtype = amp_dtype_from_str(
            self.cfg.get_path("hardware.amp_dtype", "bfloat16"))

    @torch.no_grad()
    def predict_paths(self, paths: list[Path], batch_size: int = 128,
                      num_workers: int = 8, top_k: int = 3) -> pd.DataFrame:
        acc_probs = None
        for vi, tf in enumerate(self.transforms):
            ds = InferenceDataset(paths, transform=tf)
            actual_workers = 0 if len(paths) <= 4 else min(num_workers, 8)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=actual_workers, pin_memory=(self.device.type == "cuda"))

            probs_v = []
            desc = f"predict (view {vi+1}/{len(self.transforms)})"
            for images, _ in tqdm(loader, desc=desc, ncols=110):
                images = images.to(self.device, non_blocking=True)
                if self.cfg.get_path("hardware.channels_last", True):
                    images = images.contiguous(memory_format=torch.channels_last)
                with torch.autocast(self.device.type, dtype=self.amp_dtype,
                                    enabled=self.amp):
                    logits = self.model(images, None)
                probs_v.append(F.softmax(logits.float(), dim=1).cpu())
            p = torch.cat(probs_v).numpy()
            acc_probs = p if acc_probs is None else acc_probs + p

        probs = acc_probs / len(self.transforms)
        preds = probs.argmax(1)
        k = min(top_k, probs.shape[1])
        topk = np.argsort(-probs, axis=1)[:, :k]

        rows = []
        for i, path in enumerate(paths):
            pred_lbl = int(preds[i])
            folder_id = self.class_map.label_to_folder[pred_lbl]
            class_id = FOLDER_TO_CLASS_ID.get(folder_id, folder_id)

            row = {
                "filepath": str(path),
                "pred_class_id": class_id,       # ค่า class_id สำหรับส่งตรวจ (เช่น 101)
                "pred_label": pred_lbl,
                "pred_class": self.class_map.label_to_char[pred_lbl],
                "pred_folder_id": folder_id,    # หมายเลข folder เดิม (เช่น 161)
                "confidence": float(probs[i, pred_lbl]),
            }
            for j in range(k):
                c = int(topk[i, j])
                c_folder = self.class_map.label_to_folder[c]
                row[f"top{j+1}_class"] = self.class_map.label_to_char[c]
                row[f"top{j+1}_class_id"] = FOLDER_TO_CLASS_ID.get(c_folder, c_folder)
                row[f"top{j+1}_prob"] = float(probs[i, c])
            rows.append(row)
        return pd.DataFrame(rows)

    @torch.no_grad()
    def predict_single(self, image_path: str | Path) -> dict:
        return self.predict_paths([Path(image_path)], batch_size=1,
                                  num_workers=0).iloc[0].to_dict()


def infer_ground_truth(paths: list[Path], class_map: ClassMap) -> list[int] | None:
    """
    ดึง Ground Truth จากชื่อโฟลเดอร์แม่:
    รองรับทั้ง:
      - class_id (เช่น โฟลเดอร์ 101)
      - folder_id (เช่น โฟลเดอร์ 161)
      - ตัวอักษรไทย (เช่น โฟลเดอร์ ก)
    """
    labels = []
    for p in paths:
        name = p.parent.name
        if name.isdigit():
            val = int(name)
            # กรณีที่ 1: ชื่อโฟลเดอร์เป็น class_id (101) ให้แปลงเป็น folder_id ก่อน
            if val in CLASS_ID_TO_FOLDER:
                fid = CLASS_ID_TO_FOLDER[val]
                if fid in class_map.folder_to_label:
                    labels.append(class_map.folder_to_label[fid])
                    continue
            # กรณีที่ 2: ชื่อโฟลเดอร์เป็น folder_id ตรงๆ (161)
            if val in class_map.folder_to_label:
                labels.append(class_map.folder_to_label[val])
                continue
            return None
        elif name in class_map.char_to_label:
            labels.append(class_map.char_to_label[name])
        else:
            return None
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description="ทำนายตัวอักษรไทยจากภาพ")
    ap.add_argument("--ckpt", default="outputs/checkpoints/best_model.pth")
    ap.add_argument("--input", required=True, help="ไฟล์ภาพ หรือโฟลเดอร์")
    ap.add_argument("--output", default="outputs/predictions.csv")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"Device: {describe_device(device)}")

    predictor = ThaiCharPredictor(args.ckpt, args.config, args.device, args.tta)
    paths = collect_images(args.input)
    print(f"พบภาพ {len(paths):,} ไฟล์ | TTA {args.tta} view(s)")

    df = predictor.predict_paths(paths, args.batch_size, args.num_workers, args.top_k)

    gt = infer_ground_truth(paths, predictor.class_map)
    if gt is not None:
        from sklearn.metrics import accuracy_score, f1_score
        df["true_label"] = gt
        df["true_class"] = [predictor.class_map.label_to_char[l] for l in gt]
        df["true_folder_id"] = [predictor.class_map.label_to_folder[l] for l in gt]
        df["true_class_id"] = [FOLDER_TO_CLASS_ID.get(f, f) for f in df["true_folder_id"]]
        
        # ตรวจสอบความถูกต้องโดยเทียบ class_id โดยตรง
        df["correct"] = df["pred_class_id"] == df["true_class_id"]
        acc = accuracy_score(df["true_class_id"], df["pred_class_id"])
        mf1 = f1_score(gt, df["pred_label"],
                       labels=np.arange(predictor.class_map.num_classes),
                       average="macro", zero_division=0)
        print(f"\n  Accuracy : {acc*100:.2f}%")
        print(f"  Macro-F1 : {mf1*100:.2f}%")

    ensure_dir(args.output)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n✓ บันทึกผลที่ {args.output}")

    if len(paths) <= 20:
        print("\nผลการทำนาย:")
        for _, r in df.iterrows():
            print(f"  {Path(r['filepath']).name:<30} → {r['pred_class']} "
                  f"(Class ID: {r['pred_class_id']}, Folder: {r['pred_folder_id']}) "
                  f"({r['confidence']*100:.1f}%)")


if __name__ == "__main__":
    main()
