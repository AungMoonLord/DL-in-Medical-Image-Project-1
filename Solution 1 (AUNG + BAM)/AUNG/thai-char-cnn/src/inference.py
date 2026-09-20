"""
Inference — ไฟล์ที่ต้องส่งอาจารย์

ใช้งาน:
  python -m src.inference --ckpt outputs/checkpoints/best_model.pth \
      --input path/to/images --output outputs/predictions.csv --tta 5

  --input รับได้ทั้ง: ไฟล์ภาพเดียว / โฟลเดอร์ / โฟลเดอร์ซ้อนโฟลเดอร์
  ถ้าโครงสร้างเป็น <root>/<folder_id>/*.png จะคำนวณ accuracy ให้อัตโนมัติ
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

        # counts_by_label = np.asarray(self.class_map.counts_by_label())
        # self.model = build_model(self.cfg, class_counts=counts_by_label).to(self.device)
        # self.model.load_state_dict(ckpt["model_state"])
        # self.model.eval()

        # 1. สำเนา config แล้วปิด pretrained=False เพื่อไม่ให้ timm ต่อเน็ตดาวน์โหลด ImageNet ซ้ำตอนสอบ
        infer_cfg = Config(self.cfg.to_dict())
        infer_cfg.set_path("model.pretrained", False)

        counts_by_label = np.asarray(self.class_map.counts_by_label())
        self.model = build_model(infer_cfg, class_counts=counts_by_label).to(self.device)

        # 2. ปลด prefix '_orig_mod.' ป้องกัน KeyError กรณี checkpoint ถูกเซฟตอนเปิด torch.compile
        state_dict = ckpt["model_state"]
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        if self.cfg.get_path("hardware.channels_last", True):
            self.model = self.model.to(memory_format=torch.channels_last)

        self.tta = max(1, int(tta))
        self.transforms = (build_tta_transforms(self.cfg, self.tta) if self.tta > 1
                           else [build_val_transform(self.cfg)])
        # เปิด AMP เฉพาะเมื่อรันบน GPU (CUDA) เท่านั้น
        self.amp = bool(self.cfg.get_path("hardware.amp", True)) and (self.device.type == "cuda")

        self.amp_dtype = amp_dtype_from_str(
            self.cfg.get_path("hardware.amp_dtype", "bfloat16"))

    @torch.no_grad()
    def predict_paths(self, paths: list[Path], batch_size: int = 128,
                      num_workers: int = 8, top_k: int = 3) -> pd.DataFrame:
        acc_probs = None
        for vi, tf in enumerate(self.transforms):
            ds = InferenceDataset(paths, transform=tf)

            # ---- ส่วนที่แก้ไข (จุดที่ 1.4) ----
            actual_workers = 0 if len(paths) <= 4 else min(num_workers, 8)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=actual_workers, pin_memory=(self.device.type == "cuda"))
            # ---------------------------------

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
            row = {
                "filepath": str(path),
                "pred_label": int(preds[i]),
                "pred_class": self.class_map.label_to_char[int(preds[i])],
                "pred_folder_id": self.class_map.label_to_folder[int(preds[i])],
                "confidence": float(probs[i, preds[i]]),
            }
            for j in range(k):
                c = int(topk[i, j])
                row[f"top{j+1}_class"] = self.class_map.label_to_char[c]
                row[f"top{j+1}_prob"] = float(probs[i, c])
            rows.append(row)
        return pd.DataFrame(rows)

    @torch.no_grad()
    def predict_single(self, image_path: str | Path) -> dict:
        return self.predict_paths([Path(image_path)], batch_size=1,
                                  num_workers=0).iloc[0].to_dict()


def infer_ground_truth(paths: list[Path], class_map: ClassMap) -> list[int] | None:
    """รองรับทั้งกรณี parent folder เป็น folder_id (161) หรือเป็นตัวอักษรไทย (ก)."""
    labels = []
    for p in paths:
        name = p.parent.name
        if name.isdigit() and int(name) in class_map.folder_to_label:
            labels.append(class_map.folder_to_label[int(name)])
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
        df["correct"] = df["pred_label"] == df["true_label"]
        acc = accuracy_score(gt, df["pred_label"])
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
                  f"({r['confidence']*100:.1f}%)")


if __name__ == "__main__":
    main()