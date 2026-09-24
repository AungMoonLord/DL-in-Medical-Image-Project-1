"""
Inference Pipeline for Thai Character Recognition.
Loads trained checkpoints (best_thai_character_finetuned.pth) and performs
single-image or batch predictions.
Maps numeric folder IDs to readable Thai characters and TIS-620 codes.

Usage from Terminal:
    python -m src.inference --image "path/to/image.png" --top-k 3
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from PIL import Image
import torch

from src.config import (
    DEFAULT_CHECKPOINT_PATH,
    DEVICE,
    VALID_EXTENSIONS,
    folder_to_char,
    folder_to_tis620_code,
    resolve_checkpoint_path,
)
from src.models import load_model_from_checkpoint
from src.transforms import get_inference_transforms


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
        คืนค่า: รายการ Top-k อันดับ พร้อมตัวอักษรไทย รหัส TIS-620 และระดับความมั่นใจ (%)
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
            class_name = self.idx_to_class[int(index)]
            thai_char = folder_to_char(class_name)
            tis_code = folder_to_tis620_code(class_name)
            confidence = float(probabilities[index])

            results.append({
                "rank": rank,
                "class_name": class_name,
                "tis620_code": tis_code,
                "thai_char": thai_char,
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
        สามารถบันทึกตารางผลลัพธ์เป็นไฟล์ CSV ด้วย encoding 'utf-8-sig' สำหรับเปิดใน Excel
        """
        folder_path = Path(image_folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"ไม่พบโฟลเดอร์ภาพ: {folder_path.resolve()}")

        image_paths = sorted([
            path for path in folder_path.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        ])

        print(f"[*] พบไฟล์ภาพทั้งหมด {len(image_paths)} ภาพ ใน {folder_path.resolve()}")

        batch_records = []
        for img_path in image_paths:
            predictions = self.predict_image(img_path, top_k=top_k)
            best_pred = predictions[0]

            record = {
                "file_path": str(img_path.resolve()),
                "file_name": img_path.name,
                "predicted_class": best_pred["class_name"],
                "tis620_code": best_pred["tis620_code"],
                "predicted_char": best_pred["thai_char"],
                "confidence": best_pred["confidence"],
            }
            batch_records.append(record)

        df = pd.DataFrame(batch_records)

        if output_csv is not None:
            csv_path = Path(output_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"[*] บันทึกผลลัพธ์ Batch Inference เรียบร้อยที่: {csv_path.resolve()}")

        return df


def print_prediction_table(results: List[Dict[str, Any]], image_name: str = "") -> None:
    """แสดงตารางผลลัพธ์การทำนาย ตัวอักษรไทย, รหัส TIS-620 และค่า Confidence (%)"""
    print("\n" + "=" * 60)
    if image_name:
        print(f" ผลการทำนาย: {image_name}")
    else:
        print(" ผลการทำนายตัวอักษรไทย")
    print("=" * 60)
    print(f" {'อันดับ':<6} {'ตัวอักษรไทย':<14} {'รหัส TIS-620':<18} {'ค่า Confidence (%)':<15}")
    print("-" * 60)
    for res in results:
        rank_str = f"#{res['rank']}"
        char_str = res['thai_char']
        tis_str = res['tis620_code']
        conf_str = f"{res['confidence'] * 100:.2f}%"
        print(f" {rank_str:<6} {char_str:<14} {tis_str:<18} {conf_str:<15}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thai Character Recognition - Direct Inference")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path ของไฟล์รูปภาพที่ต้องการทำนาย (เช่น path/to/image.png)"
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
        help="Path ของโมเดล Checkpoint (ค่าเริ่มต้น: best_thai_character_finetuned.pth)"
    )

    args = parser.parse_args()

    try:
        predictor = ThaiCharacterPredictor(checkpoint_path=args.checkpoint)
        results = predictor.predict_image(args.image, top_k=args.top_k)
        print_prediction_table(results, image_name=Path(args.image).name)
    except Exception as e:
        print(f"[ข้อผิดพลาด]: {e}", file=sys.stderr)
        sys.exit(1)
