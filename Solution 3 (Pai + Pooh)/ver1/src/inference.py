"""
Inference Engine and Character Metadata Suite for Thai Character Classification.

Implements:
1. Complete Thai Character Metadata Dictionary (Thai Name, Romanization, Type).
2. Model weight loader and cached predictor.
3. Base64, PIL, and raw image preprocessors (Letterbox + Crop + Normalization).
4. Top-K confidence extraction.
"""

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn

try:
    from .dataset import build_split_dataframes, letterbox_pad, tis620_to_char
    from .models import build_model
    from .transforms import get_val_transform
except (ImportError, ValueError):
    from dataset import build_split_dataframes, letterbox_pad, tis620_to_char
    from models import build_model
    from transforms import get_val_transform

# Complete Thai Character Knowledge Base
THAI_CHAR_METADATA: Dict[int, Dict[str, str]] = {
    161: {"char": "ก", "name_th": "ก ไก่", "name_en": "Ko Kai", "type": "Consonant (Middle)", "meaning": "Chicken"},
    162: {"char": "ข", "name_th": "ข ไข่", "name_en": "Kho Khai", "type": "Consonant (High)", "meaning": "Egg"},
    163: {"char": "ฃ", "name_th": "ฃ ขวด", "name_en": "Kho Khuat", "type": "Consonant (Obsolete)", "meaning": "Bottle"},
    164: {"char": "ค", "name_th": "ค ควาย", "name_en": "Kho Khwai", "type": "Consonant (Low)", "meaning": "Water Buffalo"},
    167: {"char": "ง", "name_th": "ง งู", "name_en": "Ngo Ngu", "type": "Consonant (Low)", "meaning": "Snake"},
    168: {"char": "จ", "name_th": "จ จาน", "name_en": "Cho Chan", "type": "Consonant (Middle)", "meaning": "Plate"},
    169: {"char": "ฉ", "name_th": "ฉ ฉิ่ง", "name_en": "Cho Ching", "type": "Consonant (High)", "meaning": "Cymbals"},
    170: {"char": "ช", "name_th": "ช ช้าง", "name_en": "Cho Chang", "type": "Consonant (Low)", "meaning": "Elephant"},
    171: {"char": "ซ", "name_th": "ซ โซ่", "name_en": "So So", "type": "Consonant (Low)", "meaning": "Chain"},
    173: {"char": "ญ", "name_th": "ญ หญิง", "name_en": "Yo Ying", "type": "Consonant (Low)", "meaning": "Woman"},
    175: {"char": "ฏ", "name_th": "ฏ ปฏัก", "name_en": "To Patak", "type": "Consonant (Middle)", "meaning": "Goad / Spear"},
    176: {"char": "ฐ", "name_th": "ฐ ฐาน", "name_en": "Tho Than", "type": "Consonant (High)", "meaning": "Pedestal"},
    177: {"char": "ฑ", "name_th": "ฑ มณโฑ", "name_en": "Tho Montho", "type": "Consonant (Low)", "meaning": "Montho (Queen)"},
    178: {"char": "ฒ", "name_th": "ฒ ผู้เฒ่า", "name_en": "Tho Phuthao", "type": "Consonant (Low)", "meaning": "Elderly Person"},
    179: {"char": "ณ", "name_th": "ณ เณร", "name_en": "No Nen", "type": "Consonant (Low)", "meaning": "Novice Monk"},
    180: {"char": "ด", "name_th": "ด เด็ก", "name_en": "Do Dek", "type": "Consonant (Middle)", "meaning": "Child"},
    181: {"char": "ต", "name_th": "ต เต่า", "name_en": "To Tao", "type": "Consonant (Middle)", "meaning": "Turtle"},
    182: {"char": "ถ", "name_th": "ถ ถุง", "name_en": "Tho Thung", "type": "Consonant (High)", "meaning": "Bag / Sack"},
    183: {"char": "ท", "name_th": "ท ทหาร", "name_en": "Tho Thahan", "type": "Consonant (Low)", "meaning": "Soldier"},
    184: {"char": "ธ", "name_th": "ธ ธง", "name_en": "Tho Thong", "type": "Consonant (Low)", "meaning": "Flag"},
    185: {"char": "น", "name_th": "น หนู", "name_en": "No Nu", "type": "Consonant (Low)", "meaning": "Mouse"},
    186: {"char": "บ", "name_th": "บ ใบไม้", "name_en": "Bo Baimai", "type": "Consonant (Middle)", "meaning": "Leaf"},
    187: {"char": "ป", "name_th": "ป ปลา", "name_en": "Po Pla", "type": "Consonant (Middle)", "meaning": "Fish"},
    188: {"char": "ผ", "name_th": "ผ ผึ้ง", "name_en": "Pho Phueng", "type": "Consonant (High)", "meaning": "Bee"},
    189: {"char": "ฝ", "name_th": "ฝ ฝา", "name_en": "Fo Fa", "type": "Consonant (High)", "meaning": "Lid / Cover"},
    190: {"char": "พ", "name_th": "พ พาน", "name_en": "Pho Phan", "type": "Consonant (Low)", "meaning": "Tray on Pedestal"},
    191: {"char": "ฟ", "name_th": "ฟ ฟัน", "name_en": "Fo Fan", "type": "Consonant (Low)", "meaning": "Tooth"},
    192: {"char": "ภ", "name_th": "ภ สำเภา", "name_en": "Pho Samphao", "type": "Consonant (Low)", "meaning": "Sailboat"},
    193: {"char": "ม", "name_th": "ม ม้า", "name_en": "Mo Ma", "type": "Consonant (Low)", "meaning": "Horse"},
    194: {"char": "ย", "name_th": "ย ยักษ์", "name_en": "Yo Yak", "type": "Consonant (Low)", "meaning": "Giant"},
    195: {"char": "ร", "name_th": "ร เรือ", "name_en": "Ro Ruea", "type": "Consonant (Low)", "meaning": "Boat"},
    196: {"char": "ฤ", "name_th": "ตัว ฤ", "name_en": "Rue", "type": "Vocalic Consonant", "meaning": "Sanskrit Vocalic R"},
    197: {"char": "ล", "name_th": "ล ลิง", "name_en": "Lo Ling", "type": "Consonant (Low)", "meaning": "Monkey"},
    199: {"char": "ว", "name_th": "ว แหวน", "name_en": "Wo Waen", "type": "Consonant (Low)", "meaning": "Ring"},
    200: {"char": "ศ", "name_th": "ศ ศาลา", "name_en": "So Sala", "type": "Consonant (High)", "meaning": "Pavilion"},
    201: {"char": "ษ", "name_th": "ษ ฤๅษี", "name_en": "So Rusi", "type": "Consonant (High)", "meaning": "Hermit"},
    202: {"char": "ส", "name_th": "ส เสือ", "name_en": "So Suea", "type": "Consonant (High)", "meaning": "Tiger"},
    203: {"char": "ห", "name_th": "ห หีบ", "name_en": "Ho Hip", "type": "Consonant (High)", "meaning": "Chest / Box"},
    204: {"char": "ฬ", "name_th": "ฬ จุฬา", "name_en": "Lo Chula", "type": "Consonant (Low)", "meaning": "Kite"},
    205: {"char": "อ", "name_th": "อ อ่าง", "name_en": "O Ang", "type": "Consonant (Middle)", "meaning": "Basin"},
    206: {"char": "ฮ", "name_th": "ฮ นกฮูก", "name_en": "Ho Nokhuk", "type": "Consonant (Low)", "meaning": "Owl"},
    207: {"char": "ฯ", "name_th": "ไปยาลน้อย", "name_en": "Paiyannoi", "type": "Punctuation", "meaning": "Ellipsis / Abbreviation"},
    209: {"char": "ั", "name_th": "ไม้หันอากาศ", "name_en": "Mai Han-Akat", "type": "Upper Vowel", "meaning": "Short 'a' Vowel"},
    210: {"char": "า", "name_th": "สระ อา", "name_en": "Sara Aa", "type": "Following Vowel", "meaning": "Long 'aa' Vowel"},
    212: {"char": "ิ", "name_th": "สระ อิ", "name_en": "Sara I", "type": "Upper Vowel", "meaning": "Short 'i' Vowel"},
    213: {"char": "ี", "name_th": "สระ อี", "name_en": "Sara Ii", "type": "Upper Vowel", "meaning": "Long 'ee' Vowel"},
    214: {"char": "ึ", "name_th": "สระ อึ", "name_en": "Sara Ue", "type": "Upper Vowel", "meaning": "Short 'ue' Vowel"},
    215: {"char": "ื", "name_th": "สระ อือ", "name_en": "Sara Uee", "type": "Upper Vowel", "meaning": "Long 'uee' Vowel"},
    216: {"char": "ุ", "name_th": "สระ อุ", "name_en": "Sara U", "type": "Lower Vowel", "meaning": "Short 'u' Vowel"},
    217: {"char": "ู", "name_th": "สระ อู", "name_en": "Sara Uu", "type": "Lower Vowel", "meaning": "Long 'oo' Vowel"},
    224: {"char": "เ", "name_th": "สระ เอ", "name_en": "Sara E", "type": "Leading Vowel", "meaning": "Long 'e' Vowel"},
    225: {"char": "แ", "name_th": "สระ แอ", "name_en": "Sara Ae", "type": "Leading Vowel", "meaning": "Long 'ae' Vowel"},
    226: {"char": "โ", "name_th": "สระ โอ", "name_en": "Sara O", "type": "Leading Vowel", "meaning": "Long 'o' Vowel"},
    227: {"char": "ใ", "name_th": "สระ ไอไม้ม้วน", "name_en": "Sara Ai Maimuan", "type": "Leading Vowel", "meaning": "'ai' Vowel (Rolled)"},
    228: {"char": "ไ", "name_th": "สระ ไอไม้มลาย", "name_en": "Sara Ai Maimalai", "type": "Leading Vowel", "meaning": "'ai' Vowel"},
    229: {"char": "ๅ", "name_th": "ลากข้างยาว", "name_en": "Lakkhangyao", "type": "Vowel Length Sign", "meaning": "Vowel elongation"},
    230: {"char": "ๆ", "name_th": "ไม้ยมก", "name_en": "Maiyamok", "type": "Punctuation", "meaning": "Repetition Mark"},
    231: {"char": "็", "name_th": "ไม้ไต่คู้", "name_en": "Maitaikhu", "type": "Tone / Vowel Sign", "meaning": "Vowel shortener"},
    232: {"char": "่", "name_th": "ไม้เอก", "name_en": "Mai Ek", "type": "Tone Marker (1st)", "meaning": "Low Tone Mark"},
    233: {"char": "้", "name_th": "ไม้โท", "name_en": "Mai Tho", "type": "Tone Marker (2nd)", "meaning": "Falling Tone Mark"},
    234: {"char": "๊", "name_th": "ไม้ตรี", "name_en": "Mai Tri", "type": "Tone Marker (3rd)", "meaning": "High Tone Mark"},
    236: {"char": "์", "name_th": "ไม้ทัณฑฆาต (การันต์)", "name_en": "Thanthakhat", "type": "Tone / Silence Mark", "meaning": "Cancels sound of letter"},
    240: {"char": "๐", "name_th": "เลขศูนย์ไทย", "name_en": "Sun (0)", "type": "Thai Numeral", "meaning": "Digit 0"},
    241: {"char": "๑", "name_th": "เลขหนึ่งไทย", "name_en": "Nueng (1)", "type": "Thai Numeral", "meaning": "Digit 1"},
    242: {"char": "๒", "name_th": "เลขสองไทย", "name_en": "Song (2)", "type": "Thai Numeral", "meaning": "Digit 2"},
    243: {"char": "๓", "name_th": "เลขสามไทย", "name_en": "Sam (3)", "type": "Thai Numeral", "meaning": "Digit 3"},
    244: {"char": "๔", "name_th": "เลขสี่ไทย", "name_en": "Si (4)", "type": "Thai Numeral", "meaning": "Digit 4"},
    245: {"char": "๕", "name_th": "เลขห้าไทย", "name_en": "Ha (5)", "type": "Thai Numeral", "meaning": "Digit 5"},
    246: {"char": "๖", "name_th": "เลขหกไทย", "name_en": "Hok (6)", "type": "Thai Numeral", "meaning": "Digit 6"},
    247: {"char": "๗", "name_th": "เลขเจ็ดไทย", "name_en": "Chet (7)", "type": "Thai Numeral", "meaning": "Digit 7"},
    248: {"char": "๘", "name_th": "เลขแปดไทย", "name_en": "Paet (8)", "type": "Thai Numeral", "meaning": "Digit 8"},
    249: {"char": "๙", "name_th": "เลขเก้าไทย", "name_en": "Kao (9)", "type": "Thai Numeral", "meaning": "Digit 9"},
}


class ThaiCharacterInferenceEngine:
    """
    Inference Engine supporting model caching, letterbox preprocessing,
    and structured Top-K predictions.
    """

    def __init__(self, dataset_dir: Union[str, Path], device: Optional[torch.device] = None):
        self.dataset_dir = Path(dataset_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, nn.Module] = {}
        self.val_transform = get_val_transform()

        # Build class index lookup
        _, _, self.class_to_idx = build_split_dataframes(self.dataset_dir, train_ratio=0.8)
        self.idx_to_class = {idx: num for num, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)

    def load_model(self, model_name: str, checkpoint_path: Union[str, Path]) -> nn.Module:
        """Loads and caches a model checkpoint."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = build_model(model_name, num_classes=self.num_classes, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model = model.to(self.device).eval()

        self.models[model_name] = model
        return model

    def preprocess_image(
        self,
        image_input: Union[Image.Image, np.ndarray, str, bytes],
        target_size: Tuple[int, int] = (32, 32),
    ) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses various image formats (Base64 Data URL, PIL, Numpy) into
        aspect-preserving letterboxed tensor.
        """
        # Parse image
        if isinstance(image_input, str):
            if image_input.startswith("data:image"):
                # Base64 data URL
                header, encoded = image_input.split(",", 1)
                img_bytes = base64.b64decode(encoded)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Autocrop to bounding box of content if mostly white or transparent
        np_img = np.array(pil_img)
        gray = np.mean(np_img, axis=2)
        # Find dark pixels if dark-on-light
        if np.mean(gray) > 127:
            coords = np.argwhere(gray < 220)
        else: # Light on dark
            coords = np.argwhere(gray > 35)

        if len(coords) > 10:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            # Add subtle padding
            pad = 2
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(pil_img.width, x1 + pad)
            y1 = min(pil_img.height, y1 + pad)
            pil_img = pil_img.crop((x0, y0, x1, y1))

        # Letterbox pad to target square size
        letterboxed = letterbox_pad(pil_img, target_size=target_size)
        tensor = self.val_transform(letterboxed).unsqueeze(0).to(self.device)
        return tensor, letterboxed

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[Image.Image, np.ndarray, str, bytes],
        model_name: str = "custom_cnn",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Runs model inference on input image and returns structured predictions.
        """
        if model_name not in self.models:
            raise KeyError(f"Model '{model_name}' is not loaded. Available: {list(self.models.keys())}")

        model = self.models[model_name]
        tensor, letterboxed = self.preprocess_image(image_input, target_size=(32, 32))

        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        top_k_indices = np.argsort(probs)[::-1][:top_k]

        predictions: List[Dict[str, Any]] = []
        for idx in top_k_indices:
            class_num = self.idx_to_class[idx]
            meta = THAI_CHAR_METADATA.get(class_num, {
                "char": tis620_to_char(class_num),
                "name_th": f"รหัส {class_num}",
                "name_en": f"Class {class_num}",
                "type": "General",
                "meaning": "-",
            })

            predictions.append({
                "class_number": class_num,
                "class_idx": int(idx),
                "character": meta["char"],
                "name_th": meta["name_th"],
                "name_en": meta["name_en"],
                "type": meta["type"],
                "meaning": meta["meaning"],
                "probability": float(probs[idx]),
                "confidence_pct": round(float(probs[idx]) * 100, 2),
            })

        # Convert letterbox preview to base64
        buf = io.BytesIO()
        letterboxed.save(buf, format="PNG")
        preview_base64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "model_name": model_name,
            "top_prediction": predictions[0] if predictions else None,
            "predictions": predictions,
            "preprocessed_preview": preview_base64,
        }
