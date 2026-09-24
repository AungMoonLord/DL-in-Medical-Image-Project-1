"""
Inference Pipeline for Thai Character Recognition.
Loads trained checkpoints (best_thai_character_finetuned.pth) and performs
single-image or batch predictions.
Maps numeric folder IDs to readable Thai characters and TIS-620 codes.

Usage from Terminal:
    # Single image:
    python -m src.inference --image "path/to/image.png" --top-k 3

    # Batch folder:
    python -m src.inference --folder "path/to/folder" --output-csv "predictions.csv"
"""

import argparse
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Union

# เพิ่ม project root ใน sys.path เพื่อให้ import src.* ได้อย่างเสถียรทุกรูปแบบการรัน
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from PIL import Image
import torch

from src.config import (
    DEFAULT_CHECKPOINT_PATH,
    DEVICE,
    FOLDER_TO_CLASS_ID,
    VALID_EXTENSIONS,
    folder_to_char,
    folder_to_info,
    folder_to_tis620_code,
    resolve_checkpoint_path,
)
from src.models import load_model_from_checkpoint
from src.transforms import get_inference_transforms


def natural_sort_key(path):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', path.name)]


class ThaiCharacterPredictor:
    """
    Pipeline สำหรับทำนายตัวอักษรไทยจากภาพ
    รองรับทั้งการทำนายทีละภาพ และทำนายทั้งโฟลเดอร์ (Batch Inference)
    """
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None
    ):
        if device is None:
            self.device = DEVICE
        else:
            self.device = device

        resolved_checkpoint = resolve_checkpoint_path(checkpoint_path)
        self.checkpoint_path = Path(resolved_checkpoint)
        print(f"[*] กำลังโหลด Checkpoint จาก: {self.checkpoint_path.resolve()}")

        (
            self.model,
            self.class_to_idx,
            self.idx_to_class,
            self.best_score
        ) = load_model_from_checkpoint(self.checkpoint_path, device=self.device)

        self.num_classes = len(self.class_to_idx)
        # Preprocessing Transform ตัวเดียวกับตอนเทรน (PadToSquare 255 + Resize 224x224 + Normalize ImageNet)
        self.transform = get_inference_transforms()
        print(f"[*] โหลดโมเดลสำเร็จ! จำนวนคลาส: {self.num_classes} | Validation Macro-F1: {self.best_score}")

    @torch.no_grad()
    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        ทำนายตัวอักษรไทยจากภาพเดี่ยว
        คืนค่า: รายการ Top-k อันดับ พร้อม folder, class_id, thai_char และ confidence
        """
        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            if not image_path.exists():
                raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {image_path.resolve()}")
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise TypeError("image_input ต้องเป็น Path, str หรือ PIL.Image.Image เท่านั้น")

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        k = min(top_k, self.num_classes)
        top_indices = probabilities.argsort()[::-1][:k]

        results: List[Dict[str, Any]] = []
        for rank, index in enumerate(top_indices, start=1):
            folder = str(self.idx_to_class[int(index)])
            info = folder_to_info(folder)
            confidence = float(probabilities[index])

            results.append({
                "rank": rank,
                "folder": info["folder"],
                "class_id": info["class_id"],
                "thai_char": info["thai_char"],
                "confidence": confidence,
            })

        return results

    def predict_batch(
        self,
        image_folder: Union[str, Path],
        output_csv: Optional[Union[str, Path]] = None,
        top_k: int = 1
    ) -> pd.DataFrame:
        """
        ทำนายภาพทั้งหมดภายในโฟลเดอร์ (ค้นหาโฟลเดอร์ย่อยแบบ Recursive)
        จัดเรียงลำดับไฟล์ด้วย Natural Sort (เช่น ts_img_1 ถึง ts_img_500)
        บันทึกตารางผลลัพธ์เป็นไฟล์ CSV ด้วย 4 คอลัมน์หลักตามรูปแบบของเพื่อนเป๊ะๆ
        """
        folder_path = Path(image_folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"ไม่พบโฟลเดอร์ภาพ: {folder_path.resolve()}")

        # จัดเรียงลำดับไฟล์ภาพแบบ Natural Sort เพื่อการันตีว่าลำดับแถว (ts_img_1 ถึง ts_img_500) จะตรงกับ Google Sheet 100%
        image_paths = sorted(
            [
                path for path in folder_path.rglob("*")
                if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
            ],
            key=natural_sort_key
        )

        print(f"[*] พบไฟล์ภาพทั้งหมด {len(image_paths)} ภาพ ใน {folder_path.resolve()} (จัดเรียงแบบ Natural Sort)")

        batch_results = []
        for image_path in image_paths:
            try:
                predictions = self.predict_image(image_path, top_k=top_k)
                best = predictions[0]

                batch_results.append({
                    "file_path": str(image_path),
                    "predicted_class": best["class_id"],
                    "predicted_char": best["thai_char"],
                    "confidence": best["confidence"],
                })
            except Exception as e:
                print(f"[Warning] ข้ามไฟล์ภาพเสีย ({image_path.name}): {e}", file=sys.stderr)
                batch_results.append({
                    "file_path": str(image_path),
                    "predicted_class": "101",  # กำหนด Fallback Class พื้นฐานไว้
                    "predicted_char": "ก",
                    "confidence": 0.0,
                })

        df = pd.DataFrame(batch_results)

        if output_csv is not None:
            csv_path = Path(output_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"[*] บันทึกผลลัพธ์ Batch Inference เรียบร้อยที่: {csv_path.resolve()}")

        return df


def print_prediction_table(results: List[Dict[str, Any]], image_name: str = "") -> None:
    """แสดงตารางผลลัพธ์การทำนาย Class ID (3 หลัก), ตัวอักษรไทย, โฟลเดอร์เดิม และค่า Confidence (%)"""
    print("\n" + "=" * 65)
    if image_name:
        print(f" ผลการทำนาย: {image_name}")
    else:
        print(" ผลการทำนายตัวอักษรไทย")
    print("=" * 65)
    print(f" {'อันดับ':<6} {'Class ID':<12} {'ตัวอักษรไทย':<14} {'Folder':<10} {'Confidence (%)':<15}")
    print("-" * 65)
    for res in results:
        rank_str = f"#{res['rank']}"
        cid_str = str(res.get('class_id', ''))
        char_str = str(res.get('thai_char', ''))
        fld_str = str(res.get('folder', ''))
        conf_str = f"{res['confidence'] * 100:.2f}%"
        print(f" {rank_str:<6} {cid_str:<12} {char_str:<14} {fld_str:<10} {conf_str:<15}")
    print("=" * 65 + "\n")



def parse_args():
    parser = argparse.ArgumentParser(description="Thai Character Recognition - Direct Inference")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path ของไฟล์รูปภาพเดี่ยวที่ต้องการทำนาย (เช่น path/to/image.png)"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Path ของโฟลเดอร์รูปภาพที่ต้องการทำนายแบบ Batch (ค้นหาไฟล์ภาพแบบ Recursive)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="predictions.csv",
        help="Path สำหรับบันทึกผลการทำนายแบบ Batch (.csv) (ค่าเริ่มต้น: predictions.csv)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="จำนวนอันดับผลลัพธ์ที่ต้องการแสดง (ค่าเริ่มต้น: 3)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path ของโมเดล Checkpoint (ค่าเริ่มต้น: ค้นหาอัตโนมัติ เริ่มจาก best_thai_character_finetuned.pth)"
    )
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_args()

    if not args.image and not args.folder:
        print("[ข้อผิดพลาด]: กรุณาระบุอย่างน้อย 1 อย่างระหว่าง --image (ภาพเดี่ยว) หรือ --folder (โฟลเดอร์ภาพ)", file=sys.stderr)
        print("ตัวอย่างคำสั่งใช้งาน:", file=sys.stderr)
        print('  python inference.py --image "path/to/image.png" --top-k 3', file=sys.stderr)
        print('  python inference.py --folder "path/to/dir" --output-csv "predictions.csv"', file=sys.stderr)
        sys.exit(1)

    try:
        predictor = ThaiCharacterPredictor(checkpoint_path=args.checkpoint)

        # 1. ทำนายภาพเดี่ยว
        if args.image:
            results = predictor.predict_image(args.image, top_k=args.top_k)
            print_prediction_table(results, image_name=Path(args.image).name)

        # 2. ทำนายทั้งโฟลเดอร์ (Batch Folder)
        if args.folder:
            df = predictor.predict_batch(
                image_folder=args.folder,
                output_csv=args.output_csv,
                top_k=args.top_k
            )
            print(f"[*] ทำนายภาพทั้งหมด {len(df)} ภาพเสร็จสิ้น")
            if not df.empty:
                print("\n[*] ตัวอย่างผลลัพธ์ 5 แถวแรก (พร้อมสำหรับการคัดลอกลง Google Sheet ช่อง 'กลุ่ม 2'):")
                cols_to_show = [c for c in ["file_path", "predicted_class", "predicted_char", "confidence"] if c in df.columns]
                print(df[cols_to_show].head(5).to_string(index=False))

    except Exception as e:
        print(f"[ข้อผิดพลาด]: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
